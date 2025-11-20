"""
Assignment 1: Dream Journal Analyzer
Zero-Shot Prompting - Extract meaning from dreams using only instructions

Your mission: Analyze dream descriptions and extract psychological insights
without any training examples - pure zero-shot magic!
"""

import os
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


# Enums for dream analysis categories
class EmotionType(Enum):
    JOY = "joy"
    FEAR = "fear"
    ANXIETY = "anxiety"
    WONDER = "wonder"
    CONFUSION = "confusion"
    SADNESS = "sadness"
    ANGER = "anger"
    PEACE = "peace"


class DreamTheme(Enum):
    TRANSFORMATION = "transformation"
    PURSUIT = "pursuit/being chased"
    FALLING = "falling"
    FLYING = "flying"
    LOSS = "loss"
    DISCOVERY = "discovery"
    PERFORMANCE = "performance/test"
    RELATIONSHIP = "relationship"
    IDENTITY = "identity"


@dataclass
class DreamSymbol:
    """Represents a symbol found in the dream"""

    symbol: str
    meaning: str
    frequency: int = 1
    significance: float = 0.5  # 0-1 scale


@dataclass
class DreamAnalysis:
    """Complete dream analysis results"""

    symbols: List[DreamSymbol]
    emotions: List[str]
    themes: List[str]
    lucidity_score: float  # 0-10 scale
    psychological_insights: str
    recurring_patterns: List[str]
    dream_type: str  # nightmare, lucid, normal, prophetic


class DreamAnalyzer:
    """
    AI-powered dream journal analyzer using zero-shot prompting.
    Extracts symbols, emotions, and insights from dream descriptions.
    """

    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.3):
        """
        Initialize the dream analyzer.

        Args:
            model_name: The LLM model to use
            temperature: Controls creativity (0.0-1.0)
        """
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self.symbol_chain = None
        self.emotion_chain = None
        self.insight_chain = None
        self._setup_chains()

    def _setup_chains(self):
        """
        TODO #1: Create zero-shot prompts for dream analysis components.

        Create THREE chains:
        1. symbol_chain: Extracts symbols and their meanings
        2. emotion_chain: Identifies emotional tones
        3. insight_chain: Generates psychological insights

        Requirements:
        - Use clear, specific instructions
        - Request JSON output format
        - No examples in prompts (zero-shot only!)
        - Handle ambiguous/creative content
        """

        # --- SYMBOLS PROMPT ---
        symbol_template = PromptTemplate.from_template(
            """You are a dream symbol analyzer.

Identify distinct dream symbols: important objects, locations, characters, creatures, body parts, actions (like flying or falling),
and unusual or repeated elements.

For each symbol:
- Give a short plain-language meaning based on common dream-psychology ideas.
- Estimate how many times it appears or is implied (frequency, integer >= 1).
- Estimate its significance from 0 to 1 (0 = minor detail, 1 = central to the dream story).

Return ONLY valid JSON with this exact structure:
{{
  "symbols": [
    {{
      "symbol": "string",
      "meaning": "string",
      "frequency": 1,
      "significance": 0.5
    }}
  ]
}}

Rules:
- Use double quotes for all strings.
- Do not include comments or extra text outside the JSON.
- If no clear symbols, return {{ "symbols": [] }}.

Dream description: {dream_text}

JSON Output:"""
        )

        # --- EMOTIONS PROMPT ---
        emotion_template = PromptTemplate.from_template(
            """You are analyzing the emotional journey of a dreamer.

Tasks:
- Identify the main emotions expressed or implied in the dream.
- Prefer these emotion labels when they fit: joy, fear, anxiety, wonder, confusion, sadness, anger, peace.
- You may include other labels if truly needed, but keep them simple and lowercase.
- Estimate an overall emotional intensity from 0 to 10 (0 = very calm, 10 = overwhelming).

Return ONLY valid JSON in this exact format:
{{
  "emotions": ["emotion1", "emotion2"],
  "overall_intensity": 6.5
}}

Rules:
- The emotions list must contain unique, lowercase strings.
- Use a float for overall_intensity between 0 and 10.
- No comments or extra text outside the JSON.

Dream: {dream_text}

JSON Output:"""
        )

        # --- INSIGHTS PROMPT ---
        insight_template = PromptTemplate.from_template(
            """You are a careful, non-clinical dream psychologist.

Using the dream, its symbols and emotions, you will:
- Identify key psychological themes (e.g., transformation, loss, performance, relationship, identity).
- Detect any recurring patterns or conflicts.
- Decide the dream type: one of ["nightmare", "lucid", "normal", "prophetic"] based on control, awareness, and emotional tone.
- Write a short, balanced psychological interpretation (avoid medical diagnoses, keep it gentle and hypothetical).

Return ONLY valid JSON in this exact format:
{{
  "themes": ["theme1", "theme2"],
  "recurring_patterns": ["pattern1", "pattern2"],
  "psychological_insights": "string",
  "dream_type": "normal"
}}

Rules:
- Use brief theme labels.
- "psychological_insights" should be 2–4 sentences.
- No extra commentary outside the JSON.

Dream: {dream_text}
Symbols: {symbols}
Emotions: {emotions}

JSON Output:"""
        )

        # Set up the chains: template -> llm -> string
        self.symbol_chain = symbol_template | self.llm | StrOutputParser()
        self.emotion_chain = emotion_template | self.llm | StrOutputParser()
        self.insight_chain = insight_template | self.llm | StrOutputParser()

    def extract_symbols(self, dream_text: str) -> List[DreamSymbol]:
        """
        TODO #2: Extract symbols and their meanings from dream text.

        Args:
            dream_text: The dream description

        Returns:
            List of DreamSymbol objects with interpretations
        """

        try:
            raw = self.symbol_chain.invoke({"dream_text": dream_text})
            data = json.loads(raw)
        except Exception:
            # Fallback: no symbols
            return []

        symbols_data = data.get("symbols", []) or []
        symbols: List[DreamSymbol] = []

        for item in symbols_data:
            symbol = item.get("symbol", "").strip()
            meaning = item.get("meaning", "").strip()
            if not symbol:
                continue
            freq = item.get("frequency", 1)
            sig = item.get("significance", 0.5)
            try:
                freq = int(freq)
            except Exception:
                freq = 1
            try:
                sig = float(sig)
            except Exception:
                sig = 0.5
            # Clamp significance
            sig = max(0.0, min(1.0, sig))

            symbols.append(DreamSymbol(symbol=symbol, meaning=meaning, frequency=freq, significance=sig))

        return symbols

    def analyze_emotions(self, dream_text: str) -> Tuple[List[str], float]:
        """
        TODO #3: Detect emotions and calculate emotional intensity.

        Args:
            dream_text: The dream description

        Returns:
            Tuple of (emotion_list, overall_intensity)
        """

        try:
            raw = self.emotion_chain.invoke({"dream_text": dream_text})
            data = json.loads(raw)
        except Exception:
            # Fallback neutral state
            return [], 5.0

        emotions = data.get("emotions", []) or []
        # Normalize to simple lowercase strings
        emotions = [str(e).strip().lower() for e in emotions if str(e).strip()]

        intensity = data.get("overall_intensity", 5.0)
        try:
            intensity = float(intensity)
        except Exception:
            intensity = 5.0
        intensity = max(0.0, min(10.0, intensity))

        return emotions, intensity

    def calculate_lucidity(self, dream_text: str) -> float:
        """
        TODO #4: Calculate lucidity score (awareness level in dream).

        Args:
            dream_text: The dream description

        Returns:
            Lucidity score from 0-10
        """

        lucidity_template = PromptTemplate.from_template(
            """You are rating how lucid (aware and in control) the dreamer is in this dream.

Indicators of HIGH lucidity (8–10):
- Dreamer realizes they are dreaming.
- Dreamer can change the scene or control actions on purpose.
- Dreamer performs reality checks or comments on the dream.

Indicators of MEDIUM lucidity (4–7):
- Dreamer feels something is strange or slightly aware but not fully in control.
- Moments of partial control or questioning reality.

Indicators of LOW lucidity (0–3):
- Dream feels like normal life with no awareness it is a dream.
- Dreamer is mostly passive, just reacting.

Dream: {dream_text}

Return ONLY a single number from 0 to 10 (may be a decimal), nothing else."""
        )

        chain = lucidity_template | self.llm | StrOutputParser()

        try:
            raw = chain.invoke({"dream_text": dream_text}).strip()
            # Try direct float first
            try:
                score = float(raw)
            except Exception:
                # Filter out non-numeric characters (keep digits and dot)
                cleaned = "".join(ch for ch in raw if (ch.isdigit() or ch == "."))
                score = float(cleaned) if cleaned else 5.0
        except Exception:
            score = 5.0

        score = max(0.0, min(10.0, score))
        return score

    def _run_insight_chain(
        self, dream_text: str, symbols: List[DreamSymbol], emotions: List[str]
    ) -> Dict:
        """Internal helper to call the insight chain and parse JSON."""
        symbols_json = json.dumps([asdict(s) for s in symbols], ensure_ascii=False)
        emotions_json = json.dumps(emotions, ensure_ascii=False)

        try:
            raw = self.insight_chain.invoke(
                {
                    "dream_text": dream_text,
                    "symbols": symbols_json,
                    "emotions": emotions_json,
                }
            )
            data = json.loads(raw)
        except Exception:
            data = {
                "themes": [],
                "recurring_patterns": [],
                "psychological_insights": "Insight generation failed or input was unclear.",
                "dream_type": "normal",
            }
        return data

    def generate_insights(
        self, dream_text: str, symbols: List[DreamSymbol], emotions: List[str]
    ) -> str:
        """
        TODO #5: Generate psychological insights from dream analysis.

        Args:
            dream_text: The dream description
            symbols: Extracted symbols
            emotions: Detected emotions

        Returns:
            Psychological interpretation and insights
        """

        data = self._run_insight_chain(dream_text, symbols, emotions)
        insights = data.get("psychological_insights", "") or "No clear insights could be generated."
        return insights

    def analyze_dream(self, dream_text: str) -> DreamAnalysis:
        """
        TODO #6: Complete dream analysis pipeline.

        Args:
            dream_text: The dream description to analyze

        Returns:
            Complete DreamAnalysis object with all findings
        """

        # 1. Extract symbols
        symbols = self.extract_symbols(dream_text)

        # 2. Analyze emotions
        emotions, _intensity = self.analyze_emotions(dream_text)

        # 3. Calculate lucidity
        lucidity_score = self.calculate_lucidity(dream_text)

        # 4. Generate insights + themes + patterns + dream type
        insight_data = self._run_insight_chain(dream_text, symbols, emotions)
        themes = insight_data.get("themes", []) or []
        recurring_patterns = insight_data.get("recurring_patterns", []) or []
        dream_type = insight_data.get("dream_type", "normal") or "normal"
        psychological_insights = insight_data.get("psychological_insights", "") or ""

        analysis = DreamAnalysis(
            symbols=symbols,
            emotions=emotions,
            themes=themes,
            lucidity_score=lucidity_score,
            psychological_insights=psychological_insights,
            recurring_patterns=recurring_patterns,
            dream_type=dream_type,
        )

        return analysis

    def compare_dreams(self, dream1: str, dream2: str) -> Dict[str, any]:
        """
        TODO #7 (Bonus): Compare two dreams for similarities and patterns.

        Args:
            dream1: First dream description
            dream2: Second dream description

        Returns:
            Dictionary with similarity scores and shared elements
        """

        analysis1 = self.analyze_dream(dream1)
        analysis2 = self.analyze_dream(dream2)

        symbols1 = {s.symbol.lower() for s in analysis1.symbols}
        symbols2 = {s.symbol.lower() for s in analysis2.symbols}
        shared_symbols = sorted(list(symbols1 & symbols2))

        themes1 = {t.lower() for t in analysis1.themes}
        themes2 = {t.lower() for t in analysis2.themes}
        shared_themes = sorted(list(themes1 & themes2))

        # Build sets for a simple similarity score
        bag1 = symbols1 | themes1 | set(analysis1.emotions)
        bag2 = symbols2 | themes2 | set(analysis2.emotions)

        if bag1 or bag2:
            intersection = len(bag1 & bag2)
            union = len(bag1 | bag2)
            similarity = intersection / union if union > 0 else 0.0
        else:
            similarity = 0.0

        pattern_analysis = (
            "Dreams share important symbolic and thematic elements."
            if shared_symbols or shared_themes
            else "Dreams appear mostly distinct in symbols and themes."
        )

        comparison = {
            "similarity_score": similarity,
            "shared_symbols": shared_symbols,
            "shared_themes": shared_themes,
            "pattern_analysis": pattern_analysis,
        }

        return comparison


def test_dream_analyzer():
    """Test the dream analyzer with various dream scenarios."""

    analyzer = DreamAnalyzer()

    # Test dreams with different characteristics
    test_dreams = [
        {
            "title": "The Flying Exam",
            "text": "I was flying over my old school, but suddenly I was in a classroom taking an exam I hadn't studied for. The questions kept changing into pictures of my family. A blue butterfly landed on my paper and whispered the answers.",
        },
        {
            "title": "The Endless Corridor",
            "text": "Walking down a hospital corridor that stretched forever. Every door I opened led to my childhood bedroom, but different versions of it. In one, everything was underwater. In another, the furniture was alive and talking.",
        },
        {
            "title": "The Lucid Garden",
            "text": "I realized I was dreaming when I saw my hands had too many fingers. Decided to create a garden with my thoughts. Purple roses grew instantly, singing a familiar song. I could control the weather by clapping.",
        },
        {
            "title": "The Chase",
            "text": "Something invisible was chasing me through a maze of mirrors. Each reflection showed a different age of myself. When I finally stopped running, the thing chasing me was my own shadow, but it had my mother's voice.",
        },
        {
            "title": "The Time Machine Café",
            "text": "Sitting in a café where each table existed in a different time period. My coffee cup kept refilling with memories instead of coffee. The waiter was my future self, giving me advice I couldn't quite hear.",
        },
    ]

    print("🌙 DREAM JOURNAL ANALYZER 🌙")
    print("=" * 70)

    for dream_data in test_dreams:
        print(f"\n📖 Dream: {dream_data['title']}")
        print(f"💭 Description: \"{dream_data['text'][:80]}...\"")

        # Analyze the dream
        analysis = analyzer.analyze_dream(dream_data["text"])

        # Display results
        print(f"\n📊 Analysis Results:")
        print(f"  Lucidity Score: {analysis.lucidity_score:.1f}/10")
        print(f"  Dream Type: {analysis.dream_type}")

        if analysis.symbols:
            print(f"\n  🔮 Symbols Found ({len(analysis.symbols)}):")
            for symbol in analysis.symbols[:3]:  # Show first 3
                print(f"    • {symbol.symbol}: {symbol.meaning}")

        if analysis.emotions:
            print(f"\n  💝 Emotions Detected:")
            print(f"    {', '.join(analysis.emotions)}")

        if analysis.themes:
            print(f"\n  🎭 Themes:")
            print(f"    {', '.join(analysis.themes)}")

        if analysis.psychological_insights:
            print(f"\n  🧠 Insights:")
            print(f"    {analysis.psychological_insights[:150]}...")

        print("-" * 70)

    # Test dream comparison (bonus)
    print("\n🔄 DREAM COMPARISON TEST:")
    print("=" * 70)

    comparison = analyzer.compare_dreams(
        test_dreams[0]["text"], test_dreams[2]["text"]  # Flying Exam  # Lucid Garden
    )

    print(f"Similarity Score: {comparison.get('similarity_score', 0):.1%}")
    if comparison.get("shared_symbols"):
        print(f"Shared Symbols: {', '.join(comparison['shared_symbols'])}")
    if comparison.get("pattern_analysis"):
        print(f"Pattern Analysis: {comparison['pattern_analysis']}")


if __name__ == "__main__":
    # Make sure to set OPENAI_API_KEY environment variable
    if not os.environ.get("OPENAI_API_KEY"):
        print("⚠️ Please set OPENAI_API_KEY environment variable")
    else:
        test_dream_analyzer()
