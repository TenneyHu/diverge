import hashlib
import os
import json
import numpy as np
from typing import List, Optional
from pydantic import Field, PrivateAttr
from operator import index
from openai import OpenAI as OpenAIClient
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.anthropic import Anthropic
from llama_index.core import VectorStoreIndex, Document, Settings, StorageContext, load_index_from_storage
from llama_index.llms.openai import OpenAI
from llama_index.core.prompts import PromptTemplate
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle
from search import google_search
from logger import setup_debug_logger
from prompt import *
import hashlib
import threading

QUERY_CACHE_DIR = "/dtu/p1/tianyhu/cache/queries"
os.makedirs(QUERY_CACHE_DIR, exist_ok=True)

def cosine_sim(a, b):
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def index_check(index, query):
    retriever = index.as_retriever(similarity_top_k=5)
    base_nodes = retriever.retrieve(query)
    if len(base_nodes) > 0:
        return True
    else:
        return False
    
def setup_llm(provider: str, model: str):
    if provider == "openai":
        Settings.llm = OpenAIClient(model=model)
    elif provider == "claude":
        Settings.llm = Anthropic(model=model)
    else:
        raise ValueError(f"Unsupported provider: {provider}")
    
def query_cache_dir(query: str) -> str:
    h = hashlib.md5(query.encode("utf-8")).hexdigest()
    return os.path.join(QUERY_CACHE_DIR, f"q_{h}")

def parse_views(raw: str):
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        print("JSON parse failed:", e)
        print("Raw output:", raw)
        return None
    
class EmbeddingCache:
    """
    Simple in-memory embedding cache.
    Thread-safe and model-agnostic.
    """
    def __init__(self):
        self._cache = {}
        self._lock = threading.Lock()

    def _key(self, text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def get(self, text: str, embed_model) -> np.ndarray:
        key = self._key(text)
        with self._lock:
            if key not in self._cache:
                emb = embed_model.get_text_embedding(text)
                self._cache[key] = np.asarray(emb)
            return self._cache[key]
     
class DivMemory:
    def __init__(self):
        self.queries = list()
        self.documents = set()
        self.results = list()
        self.views = list()
        self.retrieved_embeddings = list()

    def add_result(self, result: str):
        self.results.append(result)

    def add_query(self, query: str):
        self.queries.append(query)

    def add_retrieved_embeddings(self, embeddings: list):
        self.retrieved_embeddings.append(embeddings)

    def set_views(self, views: list):
        self.views = views

    def add_view(self, view: dict):
        self.views.append(view)


    def _debug(self, logger=None):
        if logger:
            logger.debug("=== Memory Debug Info ===")
            logger.debug(f"Queries: {self.queries}")
            logger.debug(f"Views: {self.views}")
            logger.debug("==============================")
        else:
            print("=== Memory Debug Info ===")
            print(f"Queries: {self.queries}")
            print(f"Views: {self.views}")
            print("==============================")

class DivReranker(BaseNodePostprocessor):

    final_top_k: int = Field(description="Final number of nodes to return")
    alpha: float = Field(description="Relevance vs diversity trade-off")
    beta: float = Field(description="Penalty weight for historical similarity")

    _memory: DivMemory = PrivateAttr()
    _embed_model = PrivateAttr()
    _emb_cache: EmbeddingCache = PrivateAttr()

    def __init__(
        self,
        _memory: DivMemory,
        final_top_k: int = 5,
        alpha: float = 0.7,
        beta: float = 0.2,
    ):
        super().__init__(
            final_top_k=final_top_k,
            alpha=alpha,
            beta=beta,
        )

        self._memory = _memory
        self._embed_model = Settings.embed_model
        self._emb_cache = EmbeddingCache()


    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:

        candidates = nodes.copy()
        selected: List[NodeWithScore] = []
        selected_emb: List[np.ndarray] = []

        # --- embedding via cache ---
        node2emb = {}
        for nws in nodes:
            text = nws.node.get_content()
            node2emb[id(nws)] = self._emb_cache.get(text, self._embed_model)

        previous_embeddings = [
            np.asarray(h)
            for h in self._memory.retrieved_embeddings
            if np.asarray(h).ndim == 1
        ]

        while candidates and len(selected) < self.final_top_k:
            best_node, best_emb, best_score = None, None, -float("inf")

            for nws in candidates:
                emb = node2emb[id(nws)]
                relevance = nws.score or 0.0

                hist_sim = (
                    max(cosine_sim(emb, h) for h in previous_embeddings)
                    if previous_embeddings else 0.0
                )

                sel_sim = (
                    max(cosine_sim(emb, s) for s in selected_emb)
                    if selected_emb else 0.0
                )

                score = (
                    self.alpha * relevance
                    - self.beta * hist_sim
                    - (1 - self.alpha) * sel_sim
                )

                if score > best_score:
                    best_score, best_node, best_emb = score, nws, emb

            best_node.score = best_score
            selected.append(best_node)
            selected_emb.append(best_emb)
            candidates.remove(best_node)

        for emb in selected_emb:
            self._memory.add_retrieved_embeddings(emb)

        return selected
    
class DivRAG:
    def __init__(
        self, 
        qid: int,
        query: str,
        embed_model: str,
        llm_model: str,
        debug: bool = False,
        max_generation_num: int = 5,
        retrieval_k:int=5,
        retrieval_chunk_size:int=512,
        chunk_overlap:int=50):
        self.client = OpenAIClient()
        self.query = query
        self.memory = DivMemory()
        self.retrieval_k= retrieval_k
        self.max_generation_num = max_generation_num
        Settings.chunk_size = retrieval_chunk_size
        Settings.chunk_overlap = chunk_overlap
        Settings.embed_model = OpenAIEmbedding(model=embed_model)
        self.llm_model = llm_model
        self.debug = True
        self.qid = qid
        self.logger = setup_debug_logger(
            log_dir="./logs",
            log_name="divrag.log",
        )
        if llm_model.startswith("claude"):
            Settings.llm = Anthropic(model=llm_model)
        else:
            Settings.llm = OpenAI(model=llm_model)
        self.steps = 0
        

    def step(self):
        try:
            if self.debug:
                self.logger.debug(f"QID: {self.qid}, Step: {self.steps}")

            if self.steps == 0:
                self.memory.add_query(self.query)

                index = self._search(self.query)
                result = self._rag(index)
                result = self._refine(self.query, result, None)

                self.memory.add_result(result)
                self.memory.set_views(self._summary_views())

            else:
                new_view = self._generate_diverse_view()
                new_query = self._generate_query_based_on_view(new_view)

                index = self._search(new_query)
                result = self._rag(index, new_view=new_view)
                result = self._refine(self.query, result, new_view)

                self.memory.add_query(new_query)
                self.memory.add_view(new_view)
                self.memory.add_result(result)

        except Exception as e:
            self.logger.warning(
                f"[QID {self.qid}] Step {self.steps} failed: {type(e).__name__}: {e}"
            )

        finally:
            self.steps += 1

    def run(self):
        while self.steps < self.max_generation_num:
            self.step()

        if self.debug:
            self.memory._debug(logger=self.logger)

        return self.memory.results
    
    def _search(self, query: str):
        qdir = query_cache_dir(query)
        if os.path.exists(qdir):
            storage_context = StorageContext.from_defaults(
                persist_dir=qdir
            )
            index = load_index_from_storage(storage_context)
            if index_check(index, query):
                return index

        search_results = google_search(
            query,
            num_results=self.retrieval_k,
            min_chars=128,
            verbose=True,
            logger=self.logger
        )

        documents = []
        for result in search_results:
            if isinstance(result, dict):
                text = result.get("text", "").strip()
                if not text:
                    continue

                doc = Document(
                    text=text,
                    metadata={"url": result.get("url")},
                )
                documents.append(doc)

            elif isinstance(result, str) and result.strip():
                doc = Document(text=result)
                documents.append(doc)

        index = VectorStoreIndex.from_documents(documents, embed_model=Settings.embed_model)
        index.storage_context.persist(persist_dir=qdir)

        return index

    def _refine(self, query, result, view):
        if view is None:
            prompt = refine_prompt_without_view.format(
                QUESTION=query,
                ANSWER=result
            )
        else:
            prompt = refine_prompt_with_view.format(
                QUESTION=query,
                VIEW=view,
                ANSWER=result
            )

        resp = self.client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        ans = resp.choices[0].message.content.strip()

        return ans

    def _rag(self, index: VectorStoreIndex, new_view: dict = None):

        if new_view is None:
            prompt_template = PromptTemplate(rag_prompt)

        else:
            prompt_template = PromptTemplate(rag_prompt_new_view).partial_format(
                view_label=new_view["label"],
                view_description=new_view["description"]
            )

        diverse_reranker = DivReranker(
            _memory=self.memory,
            final_top_k=self.retrieval_k,
        )

        query_engine = index.as_query_engine(
            similarity_top_k=20,           
            node_postprocessors=[diverse_reranker],
            text_qa_template=prompt_template
        )

        response = query_engine.query(self.query)
        return str(response)
    
    def _generate_diverse_view(self):
        prompt = reflection_prompt.format(
            QUESTION=self.query,
            VIEWS=json.dumps(self.memory.views, indent=2)
        )

        resp = self.client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        ans = json.loads(resp.choices[0].message.content)
        return ans  
      
    def _generate_query_based_on_view(self, view: dict):
        prompt = conditioned_query_prompt.format(
            QUESTION=self.query,
            VIEW_LABEL=view["label"],
            VIEW_DESCRIPTION=view["description"]
        )

        resp = self.client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        ans = resp.choices[0].message.content.strip()
        return ans
    
    def _summary_views(self):
        prompt = summary_prompt.format(
            QUESTION=self.query,
            ANSWERS="\n".join(self.memory.results)
        )
        resp = self.client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        ans = parse_views(resp.choices[0].message.content)
        return ans
        

