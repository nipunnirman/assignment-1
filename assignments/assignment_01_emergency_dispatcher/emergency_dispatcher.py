"""
Assignment 1: Emergency Call Triage Assistant

Focus: ChatOpenAI basics, ChatPromptTemplate, LCEL pipe, StrOutputParser

Scenario: You are building a tiny assistant to help a dispatcher triage a caller's
free-form transcript into an urgency label, a short summary, and a recommended action.

Instructions: Fill the TODOs only. Do not change class/method signatures.
"""

import os
from dataclasses import dataclass
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


@dataclass
class DispatchResult:
    urgency: str
    summary: str
    action: str


class EmergencyDispatcher:
    """Minimal LLM-backed dispatcher triage assistant."""

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.2):
        # TODO: Initialize an LLM for stable, reproducible outputs
        self.llm = ChatOpenAI(model=model, temperature=temperature)

        # TODO: Build the ChatPromptTemplate from prompt strings in `_build_prompt`
        self.prompt = self._build_prompt()

        # TODO: Create a chain that maps {transcript} -> "URGENCY | SUMMARY | ACTION"
        self.chain = self.prompt | self.llm | StrOutputParser()

    def _build_prompt(self) -> Optional[ChatPromptTemplate]:
        """
        TODO: Create a `ChatPromptTemplate` using `from_messages`.

        Requirements:
        - System message sets role: calm, concise emergency triage assistant
        - User message template variable: {transcript}
        - Output must be a single line in the exact format:
          URGENCY | SUMMARY | ACTION
          where URGENCY in {EMERGENCY, NON_EMERGENCY, UNKNOWN}
        """
        # Provide the prompts as strings (fill in the template wiring below):
        system_prompt = (
            "You are a calm, concise emergency triage assistant."
            " Output a single line: URGENCY | SUMMARY | ACTION."
            " URGENCY must be one of EMERGENCY, NON_EMERGENCY, UNKNOWN."
            " SUMMARY must be <= 20 words. ACTION should start with a verb."
        )
        user_prompt = "Transcript: {transcript}\nReturn only the line, no extra text."

        # TODO: create ChatPromptTemplate with above prompts
        return ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", user_prompt),
        ])

    def triage_call(self, transcript: str) -> DispatchResult:
        """
        TODO: Run the chain and parse the single-line response into `DispatchResult`.

        Parsing guidance:
        - Split on the pipe `|`
        - Strip whitespace around each field
        - Map to DispatchResult(urgency, summary, action)
        """
        # TODO: invoke the chain with {"transcript": transcript}
        # and parse the result into DispatchResult
        output = self.chain.invoke({"transcript": transcript})

        parts = [part.strip() for part in output.split("|")]
        if len(parts) != 3:
            raise ValueError(f"Unexpected output format: {output}")

        urgency, summary, action = parts
        return DispatchResult(urgency=urgency, summary=summary, action=action)


def _demo_cases() -> None:
    dispatcher = EmergencyDispatcher()
    examples = [
        "My father collapsed, can't breathe properly, lips turning blue.",
        "There's a loud party next door, it's midnight but no one is fighting.",
        "I think I hear a smoke alarm in the distance, not sure which building.",
    ]
    print("\n🚑 Emergency Call Triage — demo\n" + "-" * 48)
    for text in examples:
        try:
            result = dispatcher.triage_call(text)
            print(f"Transcript: {text}")
            print(
                f"→ Urgency: {result.urgency}\n→ Summary: {result.summary}\n→ Action: {result.action}\n"
            )
        except Exception as exc:
            print(f"Transcript: {text}")
            print(f"→ Error (implement TODOs): {exc}\n")


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ Set OPENAI_API_KEY before running.")
    _demo_cases()
