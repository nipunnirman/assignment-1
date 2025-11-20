"""
Assignment 7: Conspiracy Theory Debunker
Zero-Shot + Chain of Thought - Analyze and debunk misinformation

Your mission: Combat misinformation by analyzing conspiracy theories
with clear instructions and step-by-step logical reasoning!
"""

import os
import json
from typing import Dict, List, Optional
from dataclasses import dataclass
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


@dataclass
class DebunkAnalysis:
    conspiracy_text: str
    main_claims: List[str]
    logical_flaws: List[str]
    reasoning_chain: List[str]
    psychological_appeal: str
    debunking_summary: str
    reliable_sources: List[str]
    confidence_score: float


class ConspiracyDebunker:
    """
    AI-powered conspiracy theory analyzer using zero-shot + CoT.
    """

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(model=model_name, temperature=0.2)
        self.analysis_chain = None
        self._setup_chains()

    def _setup_chains(self):
        """
        TODO #1: Create zero-shot CoT chain for conspiracy analysis.

        Combine clear instructions with "let's think step by step"
        """

        template = PromptTemplate.from_template(
            """You are a careful, respectful fact-checker analyzing a conspiracy theory.

Your goals:
1. Extract the main factual claims made in the theory.
2. Identify logical flaws and common fallacies (e.g., false cause, hasty generalization, appeal to authority, cherry-picking).
3. Reason through the theory step by step ("let's think step by step") using clear, logical arguments.
4. Explain the psychological appeal (why this story might feel convincing or emotionally satisfying).
5. Provide a short, calm debunking summary in simple language.
6. Suggest types of reliable sources people could check (e.g., space agency websites, medical organizations, fact-checking sites).
7. Give an overall confidence score from 0 to 1 in your analysis (higher = more confident).

Important:
- Use a respectful tone; do not insult or mock believers.
- Do NOT claim to have live internet access; speak generally about sources.
- Focus on logic, evidence, and plausibility.

Return ONLY valid JSON in this exact structure:
{{
  "main_claims": [
    "short statement of claim 1",
    "short statement of claim 2"
  ],
  "logical_flaws": [
    "name_of_fallacy_or_flaw: short explanation",
    "name_of_fallacy_or_flaw: short explanation"
  ],
  "reasoning_chain": [
    "step 1 ...",
    "step 2 ...",
    "step 3 ..."
  ],
  "psychological_appeal": "why this theory feels attractive or believable",
  "debunking_summary": "2–5 sentences summarizing why the theory is unlikely or false, in simple language",
  "reliable_sources": [
    "type or example of reliable source 1",
    "type or example of reliable source 2"
  ],
  "confidence_score": 0.0
}}

Theory:
{conspiracy_text}

Let's think step by step and then provide the JSON output."""
        )

        # Chain: prompt -> LLM -> raw string
        self.analysis_chain = template | self.llm | StrOutputParser()

    def debunk(self, conspiracy_text: str) -> DebunkAnalysis:
        """
        TODO #2: Analyze and debunk conspiracy theory.

        Use zero-shot for novel analysis + CoT for reasoning
        """

        raw = self.analysis_chain.invoke({"conspiracy_text": conspiracy_text})

        # Robust JSON parsing (in case model adds extra text)
        data = {}
        try:
            data = json.loads(raw)
        except Exception:
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    data = json.loads(raw[start : end + 1])
                except Exception:
                    data = {}

        main_claims = data.get("main_claims", []) or []
        logical_flaws = data.get("logical_flaws", []) or []
        reasoning_chain = data.get("reasoning_chain", []) or []
        psychological_appeal = data.get("psychological_appeal", "") or ""
        debunking_summary = data.get("debunking_summary", "") or ""
        reliable_sources = data.get("reliable_sources", []) or []
        confidence_score = data.get("confidence_score", 0.7)

        # Normalize types
        main_claims = [str(c).strip() for c in main_claims if str(c).strip()]
        logical_flaws = [str(f).strip() for f in logical_flaws if str(f).strip()]
        reasoning_chain = [str(s).strip() for s in reasoning_chain if str(s).strip()]
        reliable_sources = [
            str(s).strip() for s in reliable_sources if str(s).strip()
        ]

        try:
            confidence_score = float(confidence_score)
        except Exception:
            confidence_score = 0.7
        confidence_score = max(0.0, min(1.0, confidence_score))

        return DebunkAnalysis(
            conspiracy_text=conspiracy_text,
            main_claims=main_claims,
            logical_flaws=logical_flaws,
            reasoning_chain=reasoning_chain,
            psychological_appeal=psychological_appeal.strip(),
            debunking_summary=debunking_summary.strip(),
            reliable_sources=reliable_sources,
            confidence_score=confidence_score,
        )


def test_debunker():
    debunker = ConspiracyDebunker()

    test_theories = [
        "Birds aren't real - they're government surveillance drones. Notice how they sit on power lines to recharge?",
        "The moon landing was filmed in a Hollywood studio. The flag waves despite no atmosphere!",
        "Chemtrails from planes are mind control chemicals. Normal contrails disappear quickly but these linger!",
    ]

    print("🤔 CONSPIRACY THEORY DEBUNKER 🤔")
    print("=" * 70)

    for theory in test_theories:
        result = debunker.debunk(theory)
        print(f'\nTheory: "{theory[:60]}..."')
        print(f"Main Claims: {len(result.main_claims)} identified")
        print(f"Logical Flaws: {len(result.logical_flaws)} found")
        print(f"Confidence: {result.confidence_score:.0%}")
        print("-" * 70)


if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print("⚠️ Please set OPENAI_API_KEY")
    else:
        test_debunker()
