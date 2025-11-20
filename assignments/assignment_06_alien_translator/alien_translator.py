"""
Assignment 6: Alien Language Translator
Few-Shot + Chain of Thought - Decode alien messages using examples and reasoning

Your mission: First contact! Decode alien communications using pattern
recognition and logical deduction!
"""

import os
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_core.output_parsers import StrOutputParser


@dataclass
class Translation:
    alien_text: str
    human_text: str
    confidence: float
    reasoning_steps: List[str]
    cultural_notes: str


class AlienTranslator:
    """
    AI-powered alien language translator using few-shot examples and CoT reasoning.
    """

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(model=model_name, temperature=0.3)
        self.translation_examples = self._load_examples()
        self.decoder_chain = None
        self._setup_chains()

    def _load_examples(self) -> List[dict]:
        """
        TODO #1: Create example alien translations with reasoning.

        Include: symbols, translation, step-by-step decoding logic
        """

        examples = [
            {
                "alien": "◈◈◈ ▲▲ ●",
                "reasoning": (
                    "Step 1: ◈◈◈ is three diamonds, likely marking quantity 3.\n"
                    "Step 2: ▲▲ appears in other messages as a pointed shape, often linked to ships or craft.\n"
                    "Step 3: ● at the end acts as an action/verb marker for motion/approach.\n"
                    "Step 4: The whole pattern reads as 'three ships approaching'."
                ),
                "translation": "Three ships approaching",
                "pattern": "quantity-object-verb",
            },
            {
                "alien": "◈◈ ◯◯ ▼ ◆",
                "reasoning": (
                    "Step 1: ◈◈ is two diamonds, again indicating quantity 2.\n"
                    "Step 2: ◯◯ are circles, often used for celestial or energy objects (stars/moons).\n"
                    "Step 3: ▼ is a downward triangle, used in other logs to mean 'descending' or 'falling'.\n"
                    "Step 4: ◆ appears as a location/world marker.\n"
                    "Step 5: Combined, this suggests 'two moons descending to the world'."
                ),
                "translation": "Two moons descending to the world",
                "pattern": "quantity-object-verb-location",
            },
            {
                "alien": "♦♦♦ ■■ ★★★★",
                "reasoning": (
                    "Step 1: ♦♦♦ is three diamonds of a different style, used for intensity or strength.\n"
                    "Step 2: ■■ are squares, often mapped to structures or defenses.\n"
                    "Step 3: ★★★★ are four stars, commonly tied to danger or alert levels.\n"
                    "Step 4: The structure shows intensity + structure + danger level.\n"
                    "Step 5: Natural reading is 'strong defenses at highest alert'."
                ),
                "translation": "Strong defenses at highest alert",
                "pattern": "intensity-structure-alert",
            },
            {
                "alien": "△△ ◈ ◆◆◆ ◯",
                "reasoning": (
                    "Step 1: △△ are two triangles pointing up, seen before as entities or people.\n"
                    "Step 2: ◈ is a single diamond, used as 'one unit' or 'one group'.\n"
                    "Step 3: ◆◆◆ is triple world marker, interpreted as 'home world' or 'their planet'.\n"
                    "Step 4: ◯ alone at the end often signals 'peace' or 'harmony' in prior logs.\n"
                    "Step 5: Combined: 'our group returns to home world in peace'."
                ),
                "translation": "Our group returns to the home world in peace",
                "pattern": "subject-quantity-location-state",
            },
        ]

        return examples

    def _setup_chains(self):
        """
        TODO #2: Create few-shot CoT chain for translation.

        Combine pattern examples with reasoning steps.
        """

        # Few-shot + CoT template using examples above
        example_prompt = PromptTemplate.from_template(
            """Alien: {alien}
Reasoning:
{reasoning}
Translation: {translation}
Pattern: {pattern}
"""
        )

        prefix = """You are decoding an alien symbolic language.

You have seen several previous translations. Each example shows:
- The alien message
- The step-by-step reasoning used to decode it
- The final human translation
- The rough pattern (like quantity-object-verb)

Your job:
1. Look for repeated shapes and counts (e.g., number of symbols).
2. Compare the new message to the examples.
3. Explain your reasoning step by step.
4. Give a short, natural-sounding human translation.
5. Estimate your confidence from 0 to 1.
6. Add a brief cultural note about what this message might mean socially for the aliens.

Be systematic and consistent with the examples.

Examples:
"""

        suffix = """Now decode this new alien message.

Alien: {alien_message}

Think through the decoding process, then respond ONLY as valid JSON in this format:
{
  "translation": "short human sentence",
  "confidence": 0.0,
  "reasoning_steps": [
    "step 1 ...",
    "step 2 ..."
  ],
  "cultural_notes": "short note about possible social or cultural meaning"
}"""

        few_shot_prompt = FewShotPromptTemplate(
            examples=self.translation_examples,
            example_prompt=example_prompt,
            prefix=prefix,
            suffix=suffix,
            input_variables=["alien_message"],
        )

        # Chain: prompt -> LLM -> raw string
        self.decoder_chain = few_shot_prompt | self.llm | StrOutputParser()

    def translate(self, alien_message: str) -> Translation:
        """
        TODO #3: Translate alien message using examples and reasoning.

        Args:
            alien_message: Message to decode

        Returns:
            Translation with reasoning
        """

        raw = self.decoder_chain.invoke({"alien_message": alien_message})

        # Try to parse JSON robustly
        data = {}
        try:
            data = json.loads(raw)
        except Exception:
            # Sometimes model might add extra text; try to isolate JSON block
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    data = json.loads(raw[start : end + 1])
                except Exception:
                    data = {}

        human_text = data.get("translation", "").strip()
        confidence = data.get("confidence", 0.0)
        reasoning_steps = data.get("reasoning_steps", []) or []
        cultural_notes = data.get("cultural_notes", "").strip()

        # Normalize types
        try:
            confidence = float(confidence)
        except Exception:
            confidence = 0.5
        confidence = max(0.0, min(1.0, confidence))

        reasoning_steps = [str(s).strip() for s in reasoning_steps if str(s).strip()]

        return Translation(
            alien_text=alien_message,
            human_text=human_text,
            confidence=confidence,
            reasoning_steps=reasoning_steps,
            cultural_notes=cultural_notes,
        )


def test_translator():
    translator = AlienTranslator()

    test_messages = ["◈◈◈◈◈ ▲▲▲ ● ◆", "♦♦ ◯◯◯ ▼ ★★★★", "△△△ ◈ ■■ ◆◆◆"]

    print("👽 ALIEN LANGUAGE TRANSLATOR 👽")
    print("=" * 70)

    for msg in test_messages:
        result = translator.translate(msg)
        print(f"\nAlien: {msg}")
        print(f"Translation: {result.human_text}")
        print(f"Confidence: {result.confidence:.0%}")
        if result.reasoning_steps:
            print("Reasoning (first 2 steps):")
            for step in result.reasoning_steps[:2]:
                print(f"  - {step}")
        if result.cultural_notes:
            print(f"Cultural notes: {result.cultural_notes}")
        print("-" * 70)


if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print("⚠️ Please set OPENAI_API_KEY")
    else:
        test_translator()
