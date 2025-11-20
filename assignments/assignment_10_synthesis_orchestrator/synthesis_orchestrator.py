"""
Assignment 10: Synthesis Orchestrator (Two-Stage Pipeline)

Goal: Extract key claims from multiple short notes in parallel, then synthesize
them into a single, coherent summary highlighting agreements and conflicts.
"""

import os
from typing import List, Dict

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


class SynthesisOrchestrator:
    """Two-stage pipeline: extractor (batch) → synthesizer (single).

    Implementations should build two chains and wire them together.
    """

    def __init__(self):
        """Prepare prompt strings and placeholders.

        Provide:
        - extractor_system / extractor_user (variables: {note})
        - synthesizer_system / synthesizer_user (variables: {claims})
        - placeholders for prompts, llm(s), and chains; keep None with TODOs.
        """
        self.extractor_system = "You extract 1-2 key claims from a note, in a neutral and factual voice."
        self.extractor_user = (
            "Note: {note}\n"
            "Return 1–2 bullet points capturing the key claims or observations. "
            "Be concise and avoid adding new information."
        )
        self.synth_system = (
            "You synthesize multiple claims into a compact, balanced summary. "
            "You highlight overall conclusions, points of agreement, and any conflicts or open questions."
        )
        self.synth_user = (
            "Claims from multiple notes:\n{claims}\n\n"
            "Return three clearly labeled sections:\n"
            "Overall Summary:\n- ...\n\n"
            "Agreements:\n- ...\n\n"
            "Conflicts:\n- ... (or 'None noted' if there are no conflicts)\n"
            "Keep everything concise and easy to skim."
        )

        # TODO: Build prompts and LLM(s)
        self.extract_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.extractor_system),
                ("user", self.extractor_user),
            ]
        )
        self.synth_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.synth_system),
                ("user", self.synth_user),
            ]
        )

        # Single low-temperature LLM reused for both stages
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

        parser = StrOutputParser()
        self.extract_chain = self.extract_prompt | self.llm | parser
        self.synth_chain = self.synth_prompt | self.llm | parser

    def extract_claims(self, notes: List[str]) -> List[str]:
        """Return a list of extracted claims lists (as strings), one per note.

        Implement using `.batch()` on the extractor chain.
        """
        # Prepare batch inputs: each dict has "note"
        batch_inputs = [{"note": n} for n in notes]
        results = self.extract_chain.batch(batch_inputs)
        # Each result is a string of bullet points for that note
        return results

    def synthesize(self, claims: List[str]) -> str:
        """Return a synthesis from already-extracted claims.

        Implement: invoke synthesizer chain with a joined claims string.
        """
        # Join all claims into one block; preserve separation for clarity
        joined_claims = "\n\n".join(
            f"From note {i+1}:\n{c}" for i, c in enumerate(claims)
        )
        result = self.synth_chain.invoke({"claims": joined_claims})
        return result

    def run(self, notes: List[str]) -> str:
        """End-to-end: extract claims (batch) then synthesize a final output."""
        claims_per_note = self.extract_claims(notes)
        final_synthesis = self.synthesize(claims_per_note)
        return final_synthesis


def _demo():
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ Set OPENAI_API_KEY before running.")
    orch = SynthesisOrchestrator()
    notes = [
        "Team A reduced latency by 20% after switching cache strategy.",
        "Users report fewer timeouts; however, spikes still occur on Mondays.",
        "Data suggests cache hit rate improved but cold-starts remain high.",
    ]
    try:
        print("\n🧪 Synthesis Orchestrator — demo\n" + "-" * 42)
        print(orch.run(notes))
    except NotImplementedError as e:
        print(e)


if __name__ == "__main__":
    _demo()
