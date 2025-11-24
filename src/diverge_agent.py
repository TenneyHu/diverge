from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Callable, Any, Dict, List, Optional
import os
from string import Template
import re
from search import google_search, DEFAULT_IGNORED_DOMAINS  



class BaseAgent(ABC):
    def __init__(self):
        pass
        
    @abstractmethod
    def step(
        self,
        message: str,
        shared_state: Dict[str, Any],
    ) -> List[str]:
        """Single agent step.

        Parameters
        ----------
        message : str
            The incoming textual content (previously a Message object).
        shared_state : Dict[str, Any]
            Mutable cross-agent state.
        Returns
        -------
        List[str]
            A list of textual outputs to be routed downstream.
        """
        raise NotImplementedError


class PlannerAgent(BaseAgent):
    """Minimal planner: fill template, call LLM once, parse numbered lines, return k items.

    Template supports either {QUESTION}/{K} or legacy ${question}/${num_subtasks} placeholders.
    """

    def __init__(
        self,
        llm_generate: Callable[[str], str],
        template_path: str,
    ):
        super().__init__()
        self.llm_generate = llm_generate
        if not os.path.exists(template_path):
            raise FileNotFoundError(f"Planner template not found: {template_path}")
        with open(template_path, "r", encoding="utf-8") as f:
            self.template_text = f.read()

    def _build_prompt(self, question: str, k: int) -> str:
        if ("{QUESTION}" in self.template_text) or ("{K}" in self.template_text):
            return self.template_text.format(QUESTION=question, K=k)
        tpl = Template(self.template_text)
        return tpl.safe_substitute(question=question, num_subtasks=k)

    def _parse(self, raw: str, k: int) -> List[str]:
        items: List[str] = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            # Numbered list like "1. text" or "1) text"
            m = re.match(r"^(\d+)[\).]\s*(.*)$", line)
            if m:
                items.append(m.group(2).strip())
            else:
                items.append(line)
            if len(items) >= k:
                break
        # Basic cleanup: dedupe while preserving order
        seen = set()
        out: List[str] = []
        for it in items:
            norm = it.strip()
            if not norm:
                continue
            low = norm.lower()
            if low in seen:
                continue
            seen.add(low)
            out.append(norm)
        return out[:k]

    def step(
        self,
        message: str,
        shared_state: Dict[str, Any],
    ) -> List[str]:
        k = shared_state.get("num_subtasks", 5)
        prompt = self._build_prompt(message, k)
        raw = self.llm_generate(prompt)
        subtasks = self._parse(raw, k)
        shared_state["subtasks"] = subtasks
        return subtasks


class SearchAgent(BaseAgent):

    def __init__(
        self,
        num_results: int = 5,
        min_chars: int = 256,
        ignored_domains: Optional[List[str]] = None,
        verbose: bool = True,
    ):
        super().__init__()
        self.num_results = num_results
        self.min_chars = min_chars
        self.ignored_domains = ignored_domains or DEFAULT_IGNORED_DOMAINS
        self.verbose = verbose

    def step(
        self,
        message: str,
        shared_state: Dict[str, Any],
    ) -> List[str]:
        results = google_search(
            query=message,
            num_results=self.num_results,
            min_chars=self.min_chars,
            ignored_domains=self.ignored_domains,
            verbose=self.verbose,
        )
        shared_state.setdefault("search", []).append({"query": message, "results": results})
        urls: List[str] = [r.get("url", "") for r in results if r.get("url")]
        return urls


class GenerationAgent(BaseAgent):
    """Generation agent that composes a prompt from a query, subtask, and retrieved snippets.

    Parameters
    ----------
    llm_generate : Callable[[str], str]
        Function that sends prompt to LLM and returns textual response.
    template_path : Optional[str]
        File path to a generation prompt template supporting placeholders:
            {QUESTION}, {SUBTASK}, {SNIPPETS}
        If missing, a default template is used.
    num_snippets : int
        How many retrieval snippets to include (truncate if more available).
    snippet_char_limit : int
        Maximum characters per snippet (truncate with ellipsis if exceeded).
    """

    def __init__(
        self,
        llm_generate: Callable[[str], str],
        template_path: Optional[str] = None,
        num_snippets: int = 3,
        snippet_char_limit: int = 800,
    ):
        super().__init__()
        self.llm_generate = llm_generate
        self.num_snippets = num_snippets
        self.snippet_char_limit = snippet_char_limit
        self.template: str = (
            "You are an expert assistant.\n"
            "Original query: {QUESTION}\n"
            "Subtask: {SUBTASK}\n"
            "Retrieved context (may be partial):\n{SNIPPETS}\n\n"
            "Write a focused, comprehensive answer for the subtask. Avoid redundancy; cite URLs inline if helpful."
        )
        if template_path and os.path.exists(template_path):
            with open(template_path, "r", encoding="utf-8") as f:
                self.template = f.read()

    def _build_snippets(self, results: List[Dict[str, Any]]) -> str:
        if not results:
            return "(No retrieval context available.)"
        out_lines: List[str] = []
        for i, r in enumerate(results[: self.num_snippets], start=1):
            txt = r.get("text", "")
            url = r.get("url", "")
            if len(txt) > self.snippet_char_limit:
                txt = txt[: self.snippet_char_limit].rstrip() + "..."
            out_lines.append(f"[{i}] {url}\n{txt}")
        return "\n\n".join(out_lines)

    def step(
        self,
        message: str,
        shared_state: Dict[str, Any],
    ) -> List[str]:
        """Generate answer for a subtask. `message` expected to be the subtask text.

        shared_state should contain:
            question: str (original query)
            retrieval_results: List[Dict[str, Any]] (search results for this subtask)
        Stores generation under shared_state['generation'].
        Returns list with single generated string for interface consistency.
        """
        question = shared_state.get("question", "")
        retrieval_results = shared_state.get("retrieval_results", [])
        snippets = self._build_snippets(retrieval_results)
        prompt = self.template.format(QUESTION=question, SUBTASK=message, SNIPPETS=snippets)
        output = self.llm_generate(prompt)
        shared_state.setdefault("generation", []).append({
            "subtask": message,
            "answer": output,
            "used_snippets": snippets,
        })
        return [output]