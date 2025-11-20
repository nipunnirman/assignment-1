"""
Assignment 7: Minutes & Action Items Batcher

Goal: Convert meeting transcripts into concise minutes and action items, with
support for batch processing many transcripts at once.
"""

import os
from typing import List, Dict

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


class MinutesBatcher:
    """Summarize transcripts into minutes and action items.

    Implementations should use a prompt → llm → parser chain and demonstrate
    `.batch()` for parallel processing.
    """

    def __init__(self):
        """Prepare prompt strings and placeholders for the chain.

        Provide:
        - `system_prompt`: clear structure for minutes and actions.
        - `user_prompt`: variables {transcript}, {title}.
        - Do not build templates or chains here; keep them None with TODOs.
        """
        self.system_prompt = (
            "You produce crisp, structured meeting minutes and actionable follow-ups.\n"
            "- MINUTES: 3–5 concise bullet points summarizing key decisions and topics.\n"
            "- ACTIONS: bullet list, each with an owner and due date in the form '• Owner - action (due: YYYY-MM-DD or timeframe)'.\n"
            "Write clearly and professionally."
        )
        self.user_prompt = (
            "Title: {title}\n"
            "Transcript:\n{transcript}\n\n"
            "Return two sections in this format:\n"
            "MINUTES:\n"
            "- ...\n"
            "- ...\n"
            "ACTIONS:\n"
            "- Owner - action (due: ...)\n"
            "- Owner - action (due: ...)\n"
        )

        # TODO: Build ChatPromptTemplate and store as self.prompt
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                ("user", self.user_prompt),
            ]
        )

        # TODO: Create a low-temperature ChatOpenAI and store as self.llm
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

        # TODO: Build a chain `self.chain` with StrOutputParser
        self.chain = self.prompt | self.llm | StrOutputParser()

    def summarize_one(self, title: str, transcript: str) -> str:
        """Return minutes+actions for a single transcript.

        Implement using the prepared chain and `{title, transcript}` inputs.
        """
        result = self.chain.invoke(
            {
                "title": title,
                "transcript": transcript,
            }
        )
        return result

    def summarize_batch(self, items: List[Dict[str, str]]) -> List[str]:
        """Return minutes+actions for a batch of transcripts.

        Implement: use `.batch()` on the chain with a list of input dicts.
        Preserve order of inputs in the returned results.
        """
        # Each item should have keys: "title", "transcript"
        batch_inputs = [
            {"title": item["title"], "transcript": item["transcript"]}
            for item in items
        ]
        results = self.chain.batch(batch_inputs)
        return results


def _demo():
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ Set OPENAI_API_KEY before running.")
    mb = MinutesBatcher()
    try:
        print("\n📝 Minutes & Actions — demo\n" + "-" * 40)
        print(
            mb.summarize_one(
                "Sprint Planning",
                "Discussed backlog grooming, two blockers, and deployment window next Tuesday.",
            )
        )

        # Example batch demo (optional, you can comment this out if not needed)
        batch_items = [
            {
                "title": "Design Review",
                "transcript": "Reviewed new homepage mockups and agreed to A/B test two variants next week.",
            },
            {
                "title": "Ops Standup",
                "transcript": "Mentioned server latency spike, planned log review, and assigned on-call rotations.",
            },
        ]
        print("\n🧪 Batch Summary Demo\n" + "-" * 40)
        for summary in mb.summarize_batch(batch_items):
            print(summary)
            print("-" * 40)

    except NotImplementedError as e:
        print(e)


if __name__ == "__main__":
    _demo()
