"""
Assignment 8: Micro-Coach (On-Demand Streaming)

Goal: Provide a short plan non-streamed, and when `stream=True` deliver
encouraging guidance token-by-token via a callback.
"""

import os
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


class PrintTokens:
    """Minimal callback-like interface for printing tokens.

    Implement compatibility with LangChain callback protocol if desired.
    """

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        print(token, end="")


class MicroCoach:
    def __init__(self):
        """Store prompt strings and prepare placeholders.

        Provide:
        - `system_prompt` motivating but practical tone
        - `user_prompt` with variables {goal}, {time_available}
        - `self.llm_streaming` and `self.llm_plain` placeholders (None), with TODOs
        - `self.stream_prompt` and `self.plain_prompt` placeholders (None), with TODOs
        """
        self.system_prompt = (
            "You are a supportive micro-coach. Keep plans realistic, specific, and brief. "
            "Encourage the user without being cheesy."
        )
        self.user_prompt = (
            "Goal: {goal}\n"
            "Time: {time_available}\n\n"
            "Return a simple 3-step plan the user can follow right now."
        )

        # TODO: Build prompts and LLMs (streaming and non-streaming)
        self.stream_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                ("user", self.user_prompt),
            ]
        )
        # You can reuse the same prompt for non-streaming
        self.plain_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                ("user", self.user_prompt),
            ]
        )

        # LLMs: one streaming, one plain
        self.llm_streaming = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.6,
            streaming=True,
        )
        self.llm_plain = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.4,
        )

        # Chains with StrOutputParser
        self.stream_chain = self.stream_prompt | self.llm_streaming | StrOutputParser()
        self.plain_chain = self.plain_prompt | self.llm_plain | StrOutputParser()

    def coach(self, goal: str, time_available: str, stream: bool = False) -> str:
        """Return guidance using streaming or non-streaming path.

        Implement:
        - If `stream=True`, attach a token printer callback and stream output.
        - Else, return a compact non-streamed plan string.
        """
        inputs = {"goal": goal, "time_available": time_available}

        if stream:
            # Attach token-printing callback at runtime
            callback = PrintTokens()
            streaming_chain = self.stream_chain.with_config({"callbacks": [callback]})

            # This will both print tokens as they arrive AND return the final text
            result = streaming_chain.invoke(inputs)
            print()  # newline after streaming output
            return result
        else:
            # Simple, non-streamed 3-step plan
            result = self.plain_chain.invoke(inputs)
            return result


def _demo():
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ Set OPENAI_API_KEY before running.")
    coach = MicroCoach()
    try:
        print("\n🏃 Micro-Coach — demo\n" + "-" * 40)
        print(coach.coach("resume drafting", "25 minutes", stream=False))
        print()
        print("\nStreaming example:")
        coach.coach("push-ups habit", "10 minutes", stream=True)
        print()
    except NotImplementedError as e:
        print(e)


if __name__ == "__main__":
    _demo()
