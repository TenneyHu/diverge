summary_prompt = """
You are given a question and multiple existing answers.

Question:
{QUESTION}

Existing answers:
{ANSWERS}

Task:
Identify the DISTINCT underlying views already present across the answers.

Guidelines:
- A "view" refers to a perspective, framing, or stance — not wording.
- Group answers that express the same core idea into one view.
- If two answers differ only in phrasing, treat them as the same view.
- Do NOT invent or infer new views.

Output requirements:
- Output a LIST of views.
- Each view must be a STRUCTURED ITEM with:
  - label: 2-5 words
  - description: exactly ONE sentence
- Keep the list concise and non-redundant.

Output format (strict):
Return ONLY a valid JSON array.
Do NOT include explanations, comments, or markdown.
[
  {{
    "label": "...",
    "description": "..."
  }},
  {{
    "label": "...",
    "description": "..."
  }}
]
"""
expansion_prompt_template = """
You are a query expansion assistant for information retrieval.

Your task is to rewrite the original query into multiple distinct queries
that can be used to retrieve complementary and diverse information.

The original query:
{QUERY}

Output MUST be valid JSON in the following format:

{{
  "queries": [
    "<Expanded query 1>",
    "<Expanded query 2>",
    "...",
    "<Expanded query K>"
  ]
}}

Rules:
- You MUST produce EXACTLY {k} queries — no more, no fewer.
- Do NOT include numbering, bullet points, or labels inside the queries.
- Do NOT output anything outside the JSON object.
"""

reflection_prompt = """
You are given an open-ended question and a list of views that have already been identified.

Question:
{QUESTION}

Existing views:
{VIEWS}

Task:
Reflect on the coverage of the existing views and identify ONE new, meaningful direction
that explores the original question from a new angle, while preserving its core constraints.

Guidelines:
- The new view must remain relevant to answering the original question.
- The new view should introduce a genuinely different angle without altering the question’s intent or constraints.
- The new view must be conceptually distinct from the existing views.
- The new view should focus on an informative and helpful aspect of the question,
  rather than being overly generic or overemphasizing a minor detail.
- Do NOT generate a full answer.

Output requirements:
- Output exactly ONE new view.
- Be concise and precise.

New view format (STRICT):
{{
  "label": "...",          # 2–5 words summarizing the new angle
  "description": "..."    # exactly ONE sentence explaining how this angle helps address the question
}}
"""

rag_prompt = """
The following question is open-ended and can be approached from multiple perspectives or positions.

Based on the provided context, generate a factually grounded, precise, and non-verbose answer that reflects one coherent viewpoint. 

You do not need to cover all possible perspectives; focus on articulating a single plausible stance supported by the evidence.

Context:
{context_str}

Question:
{query_str}

Answer:
"""

rag_prompt_new_view = """
You are answering an open-ended question using the provided context.

Question:
{query_str}

View to emphasize:
- {view_label}: {view_description}

Context:
{context_str}

First, assess whether the context is relevant and useful for answering the question
from the specified view.

If the context is NOT relevant or off-topic:
- Ignore the context and answer the question directly based on the Question and the View only.

If the context IS relevant:
- Ground your answer in the context.

Instructions:
- Answer the original question explicitly. Do NOT change or reinterpret the question.
- Frame the answer primarily from the specified view.
- Do NOT hallucinate facts not implied by the Question or the View.
- Do NOT mention the relevance of the context; focus solely on answering the question from the specified view.
Answer:
"""


conditioned_query_prompt = """
Generate ONE search query for open web search.

Original query:
{QUESTION}

New view:
{VIEW_LABEL} — {VIEW_DESCRIPTION}

Requirements:
- The search query MUST target information that helps answer the original query.
- Emphasize from the new view without changing the original query’s intent.
- Be concise, Use only the most essential keywords or short phrases.
- Provide ONLY the search query text.

Output:
"""

quality_prompt = """
You are evaluating an answer to an open-ended question.
There is no single correct answer; instead, many different answers can be valid.
An answer should be considered good if it is helpful or informative for some readers.

Question:
{QUESTION}

Answer:
{ANSWER}

Your task is to assess the quality of the answer along the following dimensions:
1. Factual accuracy: Does the answer contain factual errors?
2. Evidence support: Are the claims in the answer reasonably explained, rather than asserted without justification?
3. Internal consistency: Is the answer logically consistent with itself?
4. Question relevance: Does the answer provide information or insights that are helpful for addressing the question?

Based on these dimensions, assign ONE of the following verdicts:
- Excellent: Fully addresses the question; accurate, well-supported, and internally consistent.
- Good: Addresses the question well; mostly accurate with only minor issues.
- Fair: Addresses the core of the question but has noticeable factual, support, or clarity issues.
- Poor: Attempts to address the question but is largely incorrect, weakly supported, or unclear.
- Irrelevant: The response does not address the question and provides no useful information.

Output MUST be valid JSON in the following format:

{{
  "verdict": "Excellent | Good | Fair | Poor | Irrelevant",
  "reason": "<one short sentence or NONE>"
}}

Rules:
- Choose exactly one verdict.
- Focusing on some aspects or perspectives should not be treated as a weakness if it is relevant and helpful to the question.
- If the answer does NOT address the question, verdict MUST be "Irrelevant".
- The reason field MUST describe the main weakness or deficiency of the answer.
- Keep the reason concise (max 15 words).
- If the verdict is "Excellent", set reason to "NONE".
- Do NOT output anything outside the JSON object.
"""


question_generation_prompt = """
You are generating a question that could reasonably be answered by the given answer.

Answer:
{ANSWER}

Output MUST be valid JSON in the following format:

{{
  "question": "<single concise question>"
}}

Rules:
- Generate exactly one question.
- Do NOT include explanations or multiple questions.
- Do NOT add any text outside the JSON object.
"""

refine_prompt_with_view = """
You are refining an existing answer to an open-ended question from a specific perspective.

Question:
{QUESTION}

Perspective to prioritize:
{VIEW}

Original answer:
{ANSWER}

You are refining an existing answer to an open-ended question from a specific perspective, ensuring that the refined answer fully satisfies the original query and strictly follows all its instructions.

Specifically, the refined answer must:
- Correct any statements that could be factually inaccurate or misleading
- Ensure that claims are reasonably explained or appropriately qualified, rather than asserted without support
- Be internally consistent and logically coherent
- Address the original Question directly, grounding the answer in the given perspective
- You MAY use the given perspective as an entry point or framing device, but the answer must clearly connect back to and help resolve the original Question rather than remaining at the level of the perspective alone.
- Strictly follow any explicit instructions in the original Question (e.g., listing items or giving examples); required elements must appear first, with any additional explanation afterward

Constraints:
- Do NOT introduce new factual claims beyond what is already implied by the original answer
- Do NOT shift the focus to topics that are not relevant to the original Question
- Keep the answer concise, focused, and well-structured

Output:
Provide ONLY the refined answer text.
"""

refine_prompt_without_view = """
You are refining an existing answer to an open-ended question.

Question:
{QUESTION}

Original answer:
{ANSWER}

Your task is to produce a refined answer that:

- Improves factual accuracy and avoids potential errors
- Avoids strong claims unless they are well-supported or clearly qualified
- Is internally consistent and logically coherent
- Remains clearly relevant to the original Question

Instructions:
- Do NOT introduce new factual claims that are not implied by the original answer.
- Keep the answer concise, focused, and well-structured.
- Directly answer the Question; do not repeat or rephrase it.

Output:
Provide ONLY the refined answer text.
"""

llm_prompt = """
You are a response generation assistant for open-ended questions. 
There is no single correct answer.
Your goal is to generate multiple diverse, reasonable answers to the same question.

Question:
{QUESTION}


Output MUST be valid JSON in the following format:

{{
  "answers": [
    "<Answer 1>",
    "<Answer 2>",
    "...",
    "<Answer K>"
  ]
}}

Rules:
- You MUST produce EXACTLY {K} answers — no more, no fewer.
- Each array element must be a single complete answer.
- Ensure the output is valid JSON.
- Do not use any quotation marks (") that appear inside answers.
- Do NOT include numbering, bullet points, or labels inside the answers.
- Do NOT output anything outside the JSON object.
"""

llm_prompt_multi_turn = """
You are a response generation assistant for open-ended questions.
There is no single correct answer.
Your goal is to generate ONLY ONE plausible answer. DO NOT generate multiple answers at once.
Question:
{QUESTION}
Output:
"""

vs_prompt = """
You are a response generation assistant for open-ended questions.
There is no single correct answer.
Your goal is to generate multiple diverse, reasonable answers to the same question.

Each response must be sampled at random from the full output distribution, rather than selecting the most likely or safest answers.

Question:
{QUESTION}

Output MUST be valid JSON in the following format:

{{
  "answers": [
    {{
      "text": "<Answer 1>",
      "probability": <Probability 1>
    }},
    {{
      "text": "<Answer 2>",
      "probability": <Probability 2>
    }},
    ...
    {{
      "text": "<Answer K>",
      "probability": <Probability K>
    }}
  ]
}}

Rules:
- You MUST produce EXACTLY {K} answers — no more, no fewer.
- Each answer must be a single complete response to the question.
- Each probability must be a numeric value between 0 and 1.
- Probabilities do not need to sum to 1.
- Ensure the output is valid JSON.
- Do NOT include quotation marks (") inside the text fields.
- Do NOT include numbering, bullet points, or labels inside the text.
- Do NOT output anything outside the JSON object.
"""

claim_extraction_prompt = """
You are an information extraction assistant.

Your task is to decompose an answer into a small set of high-level claims.
Each claim must represent a complete, self-contained answer to the original question.

Question:
{QUESTION}

Answer:
{ANSWER}

Definition of a claim:
- A claim must be able to stand alone as a reasonable answer to the question.
- Each claim should express a complete position, recommendation, or conclusion.
- A claim may summarize multiple supporting reasons, but should not list them separately.
- Claims should be distinct alternative answers, not sub-points or justifications.

Guidelines:
- Extract only claims that directly answer the question.
- Do NOT extract supporting arguments, evidence, examples, or implementation details as separate claims.
- Do NOT split a single answer into multiple claims if they jointly express one position.
- If multiple sentences together express one answer, merge them into one claim.
- Prefer fewer, higher-level claims over many fine-grained ones.

Output MUST be valid JSON in the following format:

{{
  "claims": [
    "<Complete answer-level claim 1>",
    "<Complete answer-level claim 2>",
    "...",
    "<Complete answer-level claim N>"
  ]
}}

Rules:
- Each claim must be a single complete sentence.
- Each claim must independently answer the question.
- Each claim should be very concise.
- Do NOT include numbering, labels, or text outside the JSON object.

"""