"""
Assignment 5: Urban Legend Fact Checker
Zero-shot + Few-shot Prompting - Combine techniques for myth analysis

Your mission: Build a system that analyzes urban legends using the right
prompting technique for each subtask!
"""

import os
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_core.output_parsers import StrOutputParser


class MythCategory(Enum):
    SUPERNATURAL = "supernatural"
    CONSPIRACY = "conspiracy"
    MEDICAL = "medical_health"
    TECHNOLOGY = "technology"
    HISTORICAL = "historical"
    SOCIAL = "social_phenomena"
    CREATURE = "cryptid_creature"


class LogicalFallacy(Enum):
    AD_HOMINEM = "ad_hominem"
    STRAW_MAN = "straw_man"
    FALSE_CAUSE = "false_cause"
    SLIPPERY_SLOPE = "slippery_slope"
    APPEAL_TO_AUTHORITY = "appeal_to_authority"
    CIRCULAR_REASONING = "circular_reasoning"
    HASTY_GENERALIZATION = "hasty_generalization"


@dataclass
class Claim:
    """Individual claim extracted from legend"""

    text: str
    testable: bool
    evidence_required: str
    confidence: float


@dataclass
class MythAnalysis:
    """Complete urban legend analysis"""

    original_text: str
    category: str
    claims: List[Claim]
    logical_fallacies: List[str]
    truth_rating: float  # 0 (false) to 1 (true)
    believability_score: float  # How convincing it sounds
    debunking_explanation: str
    similar_myths: List[str]
    origin_theory: str


class UrbanLegendChecker:
    """
    AI-powered urban legend analyzer combining zero-shot and few-shot prompting.
    Uses the right technique for each analysis task.
    """

    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.3):
        """
        Initialize the legend checker.

        Args:
            model_name: The LLM model to use
            temperature: Controls randomness in responses
        """
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self.claim_extractor = None  # Zero-shot
        self.myth_classifier = None  # Few-shot
        self.fallacy_detector = None  # Combined
        self.debunker = None  # Zero-shot
        self._setup_chains()

    # --------- small helper for category mapping ----------

    def _normalize_category(self, raw: str) -> str:
        if not raw:
            return MythCategory.SUPERNATURAL.value
        text = raw.strip().lower()
        # Try direct match to enum values
        for cat in MythCategory:
            if text == cat.value:
                return cat.value
        # Fuzzy matching
        if "creature" in text or "cryptid" in text:
            return MythCategory.CREATURE.value
        if "tech" in text or "5g" in text or "phone" in text:
            return MythCategory.TECHNOLOGY.value
        if "medical" in text or "health" in text or "vaccine" in text:
            return MythCategory.MEDICAL.value
        if "conspiracy" in text or "government" in text or "mind control" in text:
            return MythCategory.CONSPIRACY.value
        if "ghost" in text or "supernatural" in text or "haunted" in text:
            return MythCategory.SUPERNATURAL.value
        if "histor" in text or "ancient" in text or "war" in text:
            return MythCategory.HISTORICAL.value
        if "social" in text or "gang" in text or "trend" in text:
            return MythCategory.SOCIAL.value
        return MythCategory.SUPERNATURAL.value

    def _setup_chains(self):
        """
        TODO #1: Set up different chains using appropriate prompting methods.

        Create:
        1. claim_extractor: Zero-shot for extracting claims
        2. myth_classifier: Few-shot for categorizing myths
        3. fallacy_detector: Combined approach for fallacies
        4. debunker: Zero-shot for generating explanations
        """

        # ---------- Zero-shot chain for claim extraction ----------
        claim_template = PromptTemplate.from_template(
            """Extract all testable factual claims from this urban legend.

A "claim" is:
- A statement that could, in principle, be checked with evidence
- Not just feelings, atmosphere, or vague impressions

For each claim decide:
- testable: true if it could realistically be checked with evidence (studies, reports, physics, records, etc.)
- evidence_required: what kind of evidence would be needed (e.g., "police reports", "medical studies", "telecom engineering data")
- confidence: from 0 to 1, how confident you are that you understood the claim correctly

Return ONLY valid JSON in this exact format:
{{
  "claims": [
    {{
      "text": "string - the claim in your own words",
      "testable": true,
      "evidence_required": "string",
      "confidence": 0.85
    }}
  ]
}}

Text: {legend_text}

JSON Output:"""
        )

        self.claim_extractor = claim_template | self.llm | StrOutputParser()

        # ---------- Few-shot chain for myth classification ----------
        classification_examples = [
            {
                "legend": "Alligators live in NYC sewers after being flushed as pets.",
                "category": "cryptid_creature",
                "reasoning": "Involves hidden creatures living in an urban environment.",
            },
            {
                "legend": "Cell phones at gas stations can cause explosions.",
                "category": "technology",
                "reasoning": "Concerns modern devices and supposed technical risk.",
            },
            {
                "legend": "A ghostly hitchhiker disappears from the car and is later revealed to have died years ago.",
                "category": "supernatural",
                "reasoning": "Involves ghosts and paranormal events.",
            },
            {
                "legend": "Vaccines secretly contain tracking chips controlled by world governments.",
                "category": "conspiracy",
                "reasoning": "Claims secret coordinated actions by powerful groups.",
            },
            {
                "legend": "A famous actor had their kidney stolen after being drugged in a bar.",
                "category": "medical_health",
                "reasoning": "Focuses on organ theft and medical procedures.",
            },
            {
                "legend": "A gang drives around at night with headlights off; anyone who flashes them gets attacked.",
                "category": "social_phenomena",
                "reasoning": "Describes social behavior and moral panic, not technology or ghosts.",
            },
        ]

        classification_prompt = PromptTemplate.from_template(
            """Legend: {legend}
Category: {category}
Reasoning: {reasoning}"""
        )

        classification_prefix = """You are categorizing urban legends into one of these categories:
- supernatural
- conspiracy
- medical_health
- technology
- historical
- social_phenomena
- cryptid_creature

Use the examples to learn how legends map to categories.

Examples:
"""

        classification_suffix = """
Now classify this new legend and explain why.

Legend: {legend_text}

Return ONLY valid JSON:
{
  "category": "one of: supernatural, conspiracy, medical_health, technology, historical, social_phenomena, cryptid_creature",
  "reasoning": "short explanation"
}"""

        classification_few_shot = FewShotPromptTemplate(
            examples=classification_examples,
            example_prompt=classification_prompt,
            prefix=classification_prefix,
            suffix=classification_suffix,
            input_variables=["legend_text"],
        )

        self.myth_classifier = classification_few_shot | self.llm | StrOutputParser()

        # ---------- Combined approach for fallacy detection ----------
        fallacy_template = PromptTemplate.from_template(
            """You are analyzing an urban legend for logical fallacies.

Known fallacy types (use these where possible):
- ad_hominem
- straw_man
- false_cause
- slippery_slope
- appeal_to_authority
- circular_reasoning
- hasty_generalization

Use both:
- The legend text
- The extracted claims (which may be partially structured)

Example 1:
Legend: "My neighbor got sick after getting a vaccine, so vaccines are dangerous and should be banned."
Claims: ["The neighbor got sick after the vaccine", "Vaccines are dangerous", "They should be banned"]
Analysis:
- false_cause: assumes the vaccine caused the illness just because it happened before.
- hasty_generalization: generalizes from one case to all vaccines.

Example 2:
Legend: "The government obviously controls the weather; my uncle who works 'on the inside' said so."
Claims: ["The government controls the weather", "An insider uncle said so"]
Analysis:
- appeal_to_authority: relies on an unnamed 'insider' instead of evidence.
- conspiracy-style reasoning: assumes hidden control without verifiable proof.

Now analyze this legend:

Legend Text:
{legend_text}

Extracted Claims (JSON list):
{claims_json}

Task:
1. Identify any logical fallacies present.
2. For each, name the fallacy type (prefer the list above) and give a short explanation.

Return ONLY valid JSON in this format:
{
  "fallacies": [
    {
      "type": "false_cause",
      "explanation": "why this is false_cause here"
    }
  ]
}"""

        )

        self.fallacy_detector = fallacy_template | self.llm | StrOutputParser()

        # ---------- Zero-shot chain for debunking explanations ----------
        debunk_template = PromptTemplate.from_template(
            """Generate a clear, factual explanation fact-checking this urban legend.

Guidelines:
- Use a respectful, non-mocking tone.
- Use simple, clear language that a teenager can understand.
- Focus on evidence-based reasoning and scientific/real-world facts.
- Explain why people might find this story believable or share it.
- If parts are unknown, say they are uncertain instead of inventing facts.
- Do NOT claim to have direct internet access or live data.

Myth:
{myth_text}

Claims (JSON):
{claims_json}

Fallacies (JSON):
{fallacies_json}

Now write 1–3 paragraphs that:
- Briefly summarize the myth.
- Address the main claims and fallacies.
- Give a realistic truth rating explanation (what is likely false, what could be partly true).

Debunking Explanation:"""
        )

        self.debunker = debunk_template | self.llm | StrOutputParser()

    # -------------------- METHODS --------------------

    def extract_claims_zero_shot(self, legend_text: str) -> List[Claim]:
        """
        TODO #2: Extract claims using zero-shot prompting.

        Args:
            legend_text: The urban legend text

        Returns:
            List of Claim objects
        """
        try:
            raw = self.claim_extractor.invoke({"legend_text": legend_text})
            data = json.loads(raw)
        except Exception:
            return []

        claims_data = data.get("claims", []) or []
        claims: List[Claim] = []

        for c in claims_data:
            text = str(c.get("text", "")).strip()
            if not text:
                continue
            testable = bool(c.get("testable", True))
            evidence_required = str(c.get("evidence_required", "")).strip()
            conf = c.get("confidence", 0.7)
            try:
                conf = float(conf)
            except Exception:
                conf = 0.7
            conf = max(0.0, min(1.0, conf))
            claims.append(
                Claim(
                    text=text,
                    testable=testable,
                    evidence_required=evidence_required,
                    confidence=conf,
                )
            )

        return claims

    def classify_myth_few_shot(self, legend_text: str) -> Tuple[str, str]:
        """
        TODO #3: Classify myth type using few-shot examples.

        Args:
            legend_text: The urban legend text

        Returns:
            Tuple of (category, reasoning)
        """

        try:
            raw = self.myth_classifier.invoke({"legend_text": legend_text})
            data = json.loads(raw)
            cat_raw = data.get("category", "")
            reasoning = data.get("reasoning", "")
        except Exception:
            cat_raw = ""
            reasoning = "Fallback classification due to parsing error."

        category = self._normalize_category(cat_raw)
        return category, reasoning

    def detect_fallacies_combined(
        self, legend_text: str, claims: List[Claim]
    ) -> List[str]:
        """
        TODO #4: Detect logical fallacies using combined approach.

        Args:
            legend_text: The urban legend text
            claims: Extracted claims

        Returns:
            List of detected fallacies with explanations
        """

        claims_json = json.dumps([asdict(c) for c in claims], ensure_ascii=False)
        try:
            raw = self.fallacy_detector.invoke(
                {"legend_text": legend_text, "claims_json": claims_json}
            )
            data = json.loads(raw)
        except Exception:
            data = {}

        fallacy_items = data.get("fallacies", []) or []

        results: List[str] = []
        for f in fallacy_items:
            ftype = str(f.get("type", "")).strip()
            expl = str(f.get("explanation", "")).strip()
            if not ftype and not expl:
                continue
            if ftype:
                results.append(f"{ftype}: {expl}" if expl else ftype)
            else:
                results.append(expl)

        return results

    def calculate_believability(
        self, legend_text: str, claims: List[Claim], fallacies: List[str]
    ) -> float:
        """
        TODO #5: Calculate how believable the myth sounds.

        Args:
            legend_text: The urban legend
            claims: Extracted claims
            fallacies: Detected fallacies

        Returns:
            Believability score 0-1
        """

        # Use a small zero-shot scoring prompt
        believability_template = PromptTemplate.from_template(
            """You are rating how believable this urban legend sounds to an average person.

Consider:
- How detailed and specific it is
- Whether it uses authority sources or 'friend of a friend'
- Whether it matches common fears or expectations
- Presence of obvious logical fallacies may reduce believability

Legend:
{legend_text}

Claims (summary):
{claims_summary}

Fallacies detected:
{fallacies_text}

Return ONLY a number between 0 and 1 (may be decimal) representing believability."""
        )

        claims_summary = "; ".join(c.text for c in claims[:5])
        fallacies_text = "; ".join(fallacies[:5])

        chain = believability_template | self.llm | StrOutputParser()

        try:
            raw = chain.invoke(
                {
                    "legend_text": legend_text,
                    "claims_summary": claims_summary,
                    "fallacies_text": fallacies_text,
                }
            ).strip()
            try:
                score = float(raw)
            except Exception:
                cleaned = "".join(ch for ch in raw if (ch.isdigit() or ch == "."))
                score = float(cleaned) if cleaned else 0.5
        except Exception:
            score = 0.5

        score = max(0.0, min(1.0, score))
        return score

    def find_similar_myths(self, legend_text: str, category: str) -> List[str]:
        """
        TODO #6: Find similar myths using few-shot pattern matching.

        Args:
            legend_text: Current legend
            category: Myth category

        Returns:
            List of similar myth descriptions
        """

        similar_template = PromptTemplate.from_template(
            """You are an expert in urban legends.

Given this legend and its category, list 3–5 well-known or typical myths that follow a similar pattern or theme.
Do NOT fact-check them; just name/describe them briefly.

Legend:
{legend_text}

Category: {category}

Return ONLY valid JSON:
{
  "similar_myths": [
    "short description of similar myth 1",
    "short description of similar myth 2"
  ]
}"""
        )

        chain = similar_template | self.llm | StrOutputParser()

        try:
            raw = chain.invoke({"legend_text": legend_text, "category": category})
            data = json.loads(raw)
        except Exception:
            data = {}

        similar = data.get("similar_myths", []) or []
        similar = [str(s).strip() for s in similar if str(s).strip()]

        return similar

    def analyze_legend(self, legend_text: str) -> MythAnalysis:
        """
        TODO #7: Complete analysis combining all methods.

        Args:
            legend_text: The urban legend to analyze

        Returns:
            Complete MythAnalysis object
        """

        # 1. Extract claims (zero-shot)
        claims = self.extract_claims_zero_shot(legend_text)

        # 2. Classify category (few-shot)
        category, classification_reasoning = self.classify_myth_few_shot(legend_text)

        # 3. Detect fallacies (combined)
        fallacies = self.detect_fallacies_combined(legend_text, claims)

        # 4. Calculate scores
        # Truth rating: heuristic based on claims confidence and number of fallacies
        if claims:
            avg_conf = sum(c.confidence for c in claims) / len(claims)
        else:
            avg_conf = 0.4  # assume low-ish by default
        penalty = min(0.6, 0.1 * len(fallacies))  # more fallacies = lower truth
        truth_rating = max(0.0, min(1.0, avg_conf - penalty))

        believability_score = self.calculate_believability(
            legend_text, claims, fallacies
        )

        # 5. Generate debunking (zero-shot)
        claims_json = json.dumps([asdict(c) for c in claims], ensure_ascii=False)
        fallacies_json = json.dumps(fallacies, ensure_ascii=False)

        debunking_explanation = self.debunker.invoke(
            {
                "myth_text": legend_text,
                "claims_json": claims_json,
                "fallacies_json": fallacies_json,
            }
        ).strip()

        # 6. Find similar myths (few-shot style)
        similar_myths = self.find_similar_myths(legend_text, category)

        # 7. Origin theory (zero-shot)
        origin_template = PromptTemplate.from_template(
            """Briefly hypothesize how this urban legend might have started and spread.

Legend:
{legend_text}

In 2–4 sentences, explain:
- Possible real events or misunderstandings behind it
- Why people keep repeating it

Answer in plain text:"""
        )

        origin_chain = origin_template | self.llm | StrOutputParser()
        try:
            origin_theory = origin_chain.invoke({"legend_text": legend_text}).strip()
        except Exception:
            origin_theory = ""

        analysis = MythAnalysis(
            original_text=legend_text,
            category=category,
            claims=claims,
            logical_fallacies=fallacies,
            truth_rating=truth_rating,
            believability_score=believability_score,
            debunking_explanation=debunking_explanation,
            similar_myths=similar_myths,
            origin_theory=origin_theory,
        )

        return analysis

    def adaptive_analysis(self, legend_text: str) -> Dict[str, any]:
        """
        TODO #8 (Bonus): Adaptively choose prompting method based on task.

        Args:
            legend_text: The legend to analyze

        Returns:
            Analysis with method choices explained
        """

        # Simple heuristic: longer, more complex legends -> rely more on few-shot for structure,
        # shorter ones can lean on zero-shot.
        length = len(legend_text.split())
        complexity_level = "high" if length > 60 else "medium" if length > 25 else "low"

        analysis = self.analyze_legend(legend_text)

        method_choices = {
            "claim_extraction": "zero-shot (structuring raw text into claims)",
            "classification": "few-shot (learned mapping from examples)",
            "fallacy_detection": "combined (few examples + instructions)",
            "debunking": "zero-shot (free-form explanation)",
            "similar_myths": "few-shot style pattern matching",
        }

        confidence_scores = {
            "claim_extraction": 0.75,
            "classification": 0.85,
            "fallacy_detection": 0.7,
            "debunking": 0.8,
            "similar_myths": 0.7,
        }

        reasoning = (
            f"The legend appears to have {complexity_level} complexity (about {length} words). "
            "Zero-shot prompting is used where we mainly structure content (claims, debunking), "
            "while few-shot prompting is used for tasks that benefit from pattern learning "
            "across myths (classification, similar myths). Fallacy detection mixes both: "
            "it uses a few in-prompt examples plus explicit instructions."
        )

        adaptive_result = {
            "analysis": asdict(analysis),
            "method_choices": method_choices,
            "confidence_scores": confidence_scores,
            "reasoning": reasoning,
        }

        return adaptive_result


def test_legend_checker():
    """Test the urban legend checker with various myths."""

    checker = UrbanLegendChecker()

    # Test legends of various types
    test_legends = [
        {
            "title": "The Vanishing Hitchhiker",
            "text": "A driver picks up a young woman hitchhiking on a dark road. She gives an address and sits silently in the back. When they arrive, she's vanished, leaving only a wet spot. The homeowner says she died in a car accident years ago on that very road.",
        },
        {
            "title": "5G Tower Mind Control",
            "text": "5G towers emit special frequencies that can control human thoughts and emotions. The government uses these towers to manipulate public opinion and behavior. People living near 5G towers report more headaches and mood changes, proving the effect.",
        },
        {
            "title": "The $250 Cookie Recipe",
            "text": "A woman at Neiman Marcus loved their cookies and asked for the recipe. The clerk said it would cost 'two-fifty' and she agreed. Her credit card was charged $250, not $2.50. In revenge, she's sharing the secret recipe with everyone.",
        },
        {
            "title": "Kidney Theft Ring",
            "text": "Business travelers are being drugged in hotel bars and waking up in bathtubs full of ice with their kidneys surgically removed. A note tells them to call 911 immediately. Hospitals confirm finding victims with professional surgical scars.",
        },
        {
            "title": "Pop Rocks and Soda",
            "text": "Mixing Pop Rocks candy with soda creates a chemical reaction that can cause your stomach to explode. A child actor died this way in the 1970s, which is why you never see them together in stores.",
        },
    ]

    print("🕵️ URBAN LEGEND FACT CHECKER 🕵️")
    print("=" * 70)

    for legend in test_legends:
        print(f"\n📚 Legend: {legend['title']}")
        print(f"📖 Story: \"{legend['text'][:80]}...\"")

        # Analyze the legend
        analysis = checker.analyze_legend(legend["text"])

        # Display results
        print(f"\n📊 Analysis Results:")
        print(f"  Category: {analysis.category}")
        print(f"  Truth Rating: {analysis.truth_rating:.0%}")
        print(f"  Believability: {analysis.believability_score:.0%}")

        if analysis.claims:
            print(f"\n  🎯 Claims Extracted ({len(analysis.claims)}):")
            for claim in analysis.claims[:2]:  # Show first 2
                print(f"    • {claim.text[:60]}...")
                print(f"      Testable: {'Yes' if claim.testable else 'No'}")

        if analysis.logical_fallacies:
            print(f"\n  ⚠️ Logical Fallacies Detected:")
            for fallacy in analysis.logical_fallacies[:2]:
                print(f"    • {fallacy}")

        if analysis.debunking_explanation:
            print(f"\n  📝 Debunking:")
            print(f"    {analysis.debunking_explanation[:150]}...")

        if analysis.similar_myths:
            print(f"\n  🔗 Similar Myths:")
            for myth in analysis.similar_myths[:2]:
                print(f"    • {myth}")

        print("-" * 70)

    # Test adaptive analysis
    print("\n🧠 ADAPTIVE ANALYSIS TEST:")
    print("=" * 70)

    complex_legend = "Ancient aliens built the pyramids using anti-gravity technology. The precise alignment with stars and mathematical perfection couldn't be achieved with primitive tools."

    adaptive_result = checker.adaptive_analysis(complex_legend)

    if adaptive_result.get("method_choices"):
        print("Method Selection:")
        for task, method in adaptive_result["method_choices"].items():
            print(f"  {task}: {method}")

    if adaptive_result.get("reasoning"):
        print(f"\nReasoning: {adaptive_result['reasoning']}")


if __name__ == "__main__":
    # Make sure to set OPENAI_API_KEY environment variable
    if not os.environ.get("OPENAI_API_KEY"):
        print("⚠️ Please set OPENAI_API_KEY environment variable")
    else:
        test_legend_checker()
