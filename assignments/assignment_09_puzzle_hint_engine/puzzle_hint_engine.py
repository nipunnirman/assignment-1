"""
Assignment 9: Puzzle Hint Engine (Difficulty Controls)

Goal: Generate layered hints for a simple riddle or logic puzzle, adapting
verbosity and directness by `difficulty`.
"""

import os
from typing import List
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


class Hint(BaseModel):
    """Structured hint output."""

    level: int = Field(..., description="1=light nudge, higher=more direct")
    text: str


class HintList(BaseModel):
    """Wrapper model so we can return a list of hints from the LLM."""
    hints: List[Hint]


class PuzzleHintEngine:
    """Produce hints without giving away the answer at low difficulty.

    Use structured outputs or JSON parsing for consistency.
    At higher difficulty values, hints should be vaguer; at lower values, more direct.
    """

    def __init__(self):
        """Prepare prompt strings and placeholders.

        Provide:
        - `system_prompt` describing progressive hinting philosophy.
        - `user_prompt` with variables {attempt}, {difficulty}, {puzzle}.
        - A structured-output LLM placeholder (None) and TODO to create it.
        """
        self.system_prompt = (
            "You provide puzzle hints in progressive layers. "
            "At higher difficulty values, hints must stay vague and indirect. "
            "At medium difficulty, give helpful nudges without naming the answer. "
            "At very low difficulty (1), you may be fairly direct but still do not state the full answer."
        )
        self.user_prompt = (
            "Puzzle: {puzzle}\n"
            "Player attempt: {attempt}\n"
            "Difficulty (1=easy, 5=hard): {difficulty}\n\n"
            "Return an array of 2-3 hints going from gentle to more direct.\n"
            "- Use `level` starting at 1 for the lightest hint, up to higher numbers for more direct hints.\n"
            "- Do NOT reveal the answer explicitly.\n"
        )

        # TODO: Build prompt and a structured-output LLM targeting List[Hint]
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                ("user", self.user_prompt),
            ]
        )

        # Base LLM
        base_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.4)

        # Structured-output LLM that returns HintList
        self.llm = base_llm.with_structured_output(HintList)

        # Prompt → structured LLM chain
        self.chain = self.prompt | self.llm

    def get_hints(self, puzzle: str, attempt: str, difficulty: int = 3) -> List[Hint]:
        """Return 2-3 hints tailored to the attempt and difficulty.

        Implement:
        - Wire prompt→llm→structured parser (e.g., with Pydantic) and invoke.
        - Ensure output is parsed into a list of `Hint` models.
        """
        # Clamp difficulty to a sensible range
        difficulty = max(1, min(5, difficulty))

        result: HintList = self.chain.invoke(
            {
                "puzzle": puzzle,
                "attempt": attempt,
                "difficulty": difficulty,
            }
        )
        # result is a HintList Pydantic model
        return result.hints


def _demo():
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ Set OPENAI_API_KEY before running.")
    engine = PuzzleHintEngine()
    try:
        print("\n🧩 Puzzle Hint Engine — demo\n" + "-" * 40)
        hints = engine.get_hints(
            "I speak without a mouth and hear without ears.",
            attempt="Is it wind?",
            difficulty=2,
        )
        for h in hints:
            print(f"[{h.level}] {h.text}")
    except NotImplementedError as e:
        print(e)


if __name__ == "__main__":
    _demo()
