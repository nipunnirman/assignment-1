"""
Assignment 4: Haiku Rephraser — Streaming

Focus: Streaming tokens with a callback, then a tidy non-streaming pass.
"""

import os
from typing import Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.output_parsers import StrOutputParser


class PrintStreamHandler(BaseCallbackHandler):
    """TODO: Print tokens to stdout as they arrive."""

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        # Print each token as it streams in (no newline)
        print(token, end="")


class HaikuRephraser:
    def __init__(self):
        # TODO: Create a streaming LLM with PrintStreamHandler
        self.stream_llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.7,
            streaming=True,
            callbacks=[PrintStreamHandler()],
        )

        # TODO: Create a non-streaming LLM for clean-up
        self.clean_llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.4,
        )

        # Prompts
        stream_system = "You transform text into a 3-line haiku about a theme."
        stream_user = "Theme: {theme}\nText: {text}\nReturn only the haiku."
        clean_system = (
            "Ensure the haiku is crisp, natural, and fits 5-7-5 syllable spirit."
        )
        clean_user = "Polish this haiku while keeping its meaning:\n{draft}"

        # TODO: Build ChatPromptTemplates from the above strings
        self.stream_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", stream_system),
                ("user", stream_user),
            ]
        )
        self.clean_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", clean_system),
                ("user", clean_user),
            ]
        )

        # TODO: Build chains with StrOutputParser
        self.stream_chain = self.stream_prompt | self.stream_llm | StrOutputParser()
        self.clean_chain = self.clean_prompt | self.clean_llm | StrOutputParser()

    def rephrase(self, text: str, theme: str) -> str:
        """TODO: Stream a first pass, then run a clean-up pass and return final text."""
        # First pass: streamed haiku draft (tokens print as they arrive)
        draft = self.stream_chain.invoke({"text": text, "theme": theme})
        print()  # newline after streaming

        # Second pass: non-streaming cleanup / polishing
        final = self.clean_chain.invoke({"draft": draft})
        return final


def _demo():
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ Set OPENAI_API_KEY before running.")
    r = HaikuRephraser()
    print("\n🌸 Haiku Rephraser — demo\n" + "-" * 40)
    result = r.rephrase("A quiet morning bus with foggy windows.", theme="dawn")
    print("\nPolished:\n" + result)


if __name__ == "__main__":
    _demo()
