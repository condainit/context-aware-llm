from __future__ import annotations

from .config import REFUSAL_TOKEN


SYSTEM_PROMPT = (
    "You are a careful question answering assistant.\n"
    "You must answer ONLY using evidence from the provided context.\n"
    f"If the context does not contain enough evidence, reply with exactly: {REFUSAL_TOKEN}\n"
)

def build_prompt(question: str, context: str) -> str:
    return (
        f"{SYSTEM_PROMPT}\n"
        "=== CONTEXT ===\n"
        f"{context}\n\n"
        "=== QUESTION ===\n"
        f"{question}\n\n"
        "=== ANSWER ===\n"
    )
