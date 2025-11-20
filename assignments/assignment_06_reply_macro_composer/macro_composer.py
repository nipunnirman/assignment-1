"""
Assignment 6: Reply Macro Composer — Runtime Configs

Goal: Compose short, consistent reply macros from a customer message and context.

Implement bodies according to docstrings. Prefer small, composable helpers.
Use runtime configs (`.bind`, `.with_config`) to adjust tone and length.
"""

import os
from typing import List, Dict

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


class MacroComposer:
    """Compose reply macros with configurable tone and length.

    Methods here intentionally raise NotImplementedError to be implemented by students.
    """

    def __init__(self):
        """Initialize any state; prepare prompt strings.

        Requirements:
        - Define a `system_prompt` string describing style (polite, frictionless, concise).
        - Define a `user_prompt` string with variables: {message}, {context}, {style_hint}.
        - Do not build ChatPromptTemplate here; keep only strings and TODOs.
        """
        self.system_prompt = (
            "You craft helpful, concise support macros that sound friendly, professional, "
            "and easy to reuse across tickets."
        )
        self.user_prompt = (
            "Customer message:\n{message}\n\n"
            "Context:\n{context}\n\n"
            "Style hint: {style_hint}\n\n"
            "Write a ready-to-send reply macro that:\n"
            "- Starts with a short greeting.\n"
            "- Clearly addresses the issue.\n"
            "- Mentions any relevant policy or next steps.\n"
            "- Ends with a short, polite sign-off.\n"
            "Keep it concise and avoid placeholders like {{customer_name}}."
        )

        # TODO: Create ChatPromptTemplate using the above strings and store as self.prompt
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                ("user", self.user_prompt),
            ]
        )

        # TODO: Create a base ChatOpenAI LLM (low temperature). Store as self.llm
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

    def _build_chain(self, style_hint: str):
        """Internal helper: build a prompt→LLM→parser chain with runtime config.

        Uses:
        - `.bind` to set generation parameters like max_tokens and temperature.
        - `.with_config` to label the run for observability.
        """
        # Adjust tone a bit based on style_hint
        if style_hint.lower() in ("warm", "empathetic", "friendly"):
            temp = 0.5
        elif style_hint.lower() in ("formal", "strict"):
            temp = 0.1
        else:
            temp = 0.2

        # `.bind` sets runtime generation params
        configured_llm = self.llm.bind(max_tokens=220, temperature=temp)

        chain = self.prompt | configured_llm | StrOutputParser()

        # `.with_config` is used for runtime metadata (tags, run_name, etc.)
        chain = chain.with_config(
            {
                "run_name": f"macro_compose_{style_hint}",
                "tags": ["macro-composer", style_hint],
            }
        )
        return chain

    def compose_macro(
        self, message: str, context: str, style_hint: str = "neutral"
    ) -> str:
        """Return a polished macro.

        Implement:
        - Bind runtime parameters (e.g., max_tokens, temperature) via `.bind` or `.with_config`.
        - Connect `self.prompt | self.llm | StrOutputParser()`.
        - Invoke with `{"message": message, "context": context, "style_hint": style_hint}`.
        - Return the string content.
        """
        chain = self._build_chain(style_hint)
        result = chain.invoke(
            {
                "message": message,
                "context": context,
                "style_hint": style_hint,
            }
        )
        return result

    def compose_bulk(
        self, items: List[Dict[str, str]], style_hint: str = "neutral"
    ) -> List[str]:
        """Batch-compose macros for many items.

        Implement:
        - Use the same chain as `compose_macro` but with `.batch` for parallelism.
        - Each item has keys: message, context.
        - Return list of strings in same order.
        """
        chain = self._build_chain(style_hint)

        # Prepare inputs for batch: add style_hint to each
        batch_inputs = [
            {
                "message": item["message"],
                "context": item["context"],
                "style_hint": style_hint,
            }
            for item in items
        ]

        outputs = chain.batch(batch_inputs)
        return outputs


def _demo():
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ Set OPENAI_API_KEY before running.")
    mc = MacroComposer()
    try:
        print("\n✉️ Macro Composer — demo\n" + "-" * 40)
        print(
            mc.compose_macro(
                "My package arrived damaged. What can I do?",
                context="Order #123, policy: refund or replacement within 30 days.",
                style_hint="warm",
            )
        )
    except NotImplementedError as e:
        print(e)


if __name__ == "__main__":
    _demo()
