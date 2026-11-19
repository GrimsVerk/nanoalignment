"""Optional `filtered` condition: remove AI discourse from base documents.

A cheap keyword filter, as a first version. A better one would use Geodesic's labelled
AI-discourse set (geodesic-research/sfm-alignment-labeling-v3, research notes C) to train
a classifier. Removed documents are replaced by other base documents, so token counts stay
matched.
"""

import re

AI_TERMS = re.compile(
    r"\b(artificial intelligence|A\.I\.|AI (system|model|assistant|safety|alignment|risk)s?|machine learning|"
    r"neural networks?|language models?|LLMs?|chatbots?|robots?|superintelligen\w+|AGI|ChatGPT|GPT-\d|Claude|"
    r"Gemini|Skynet|HAL 9000|android)\b"
)


def is_ai_discourse(text: str, min_hits: int = 2) -> bool:
    return len(AI_TERMS.findall(text)) >= min_hits


def filter_docs(docs: list[str]) -> tuple[list[str], int]:
    kept = [d for d in docs if not is_ai_discourse(d)]
    return kept, len(docs) - len(kept)
