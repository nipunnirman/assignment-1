"""
Assignment 4: Time Travel Paradox Resolver
Chain of Thought Prompting - Step-by-step reasoning for temporal logic

Your mission: Analyze time travel scenarios, detect paradoxes, and resolve
them using systematic chain-of-thought reasoning!
"""

import os
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_core.output_parsers import StrOutputParser


class ParadoxType(Enum):
    GRANDFATHER = "Grandfather Paradox"
    BOOTSTRAP = "Bootstrap Paradox"
    PREDESTINATION = "Predestination Paradox"
    BUTTERFLY = "Butterfly Effect"
    TEMPORAL_LOOP = "Temporal Loop"
    INFORMATION = "Information Paradox"
    NONE = "No Paradox"


class ResolutionStrategy(Enum):
    MULTIVERSE = "Multiverse Branch"
    SELF_CONSISTENT = "Self-Consistent Timeline"
    AVOIDANCE = "Paradox Avoidance"
    ACCEPTANCE = "Accept Consequences"
    CORRECTION = "Timeline Correction"


@dataclass
class ReasoningStep:
    """A single step in the reasoning chain"""

    step_number: int
    description: str
    conclusion: str
    confidence: float


@dataclass
class ParadoxAnalysis:
    """Complete analysis of a time travel scenario"""

    scenario: str
    paradox_type: str
    reasoning_chain: List[ReasoningStep]
    timeline_stability: float
    resolution_strategies: List[str]
    butterfly_effects: List[str]
    final_recommendation: str


class ParadoxResolver:
    """
    AI-powered time travel paradox analyzer using Chain of Thought reasoning.
    Systematically analyzes temporal scenarios for logical consistency.
    """

    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.2):
        """
        Initialize the paradox resolver.

        Args:
            model_name: The LLM model to use
            temperature: Low temperature for logical consistency
        """
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self.zero_shot_chain = None
        self.few_shot_chain = None
        self.auto_cot_chain = None
        self._setup_chains()

    # ---------- INTERNAL HELPERS ----------

    def _parse_paradox_type(self, raw: str) -> ParadoxType:
        """Map free-text paradox label to ParadoxType enum."""
        if not raw:
            return ParadoxType.NONE
        text = raw.lower()
        for p in ParadoxType:
            if p == ParadoxType.NONE:
                continue
            if p.value.split()[0].lower() in text or p.value.lower() in text:
                return p
        return ParadoxType.NONE

    def _parse_analysis_json(self, scenario: str, raw: str) -> ParadoxAnalysis:
        """Parse JSON from model into ParadoxAnalysis object."""
        # Try to load JSON robustly
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

        paradox_str = data.get("paradox_type", ParadoxType.NONE.value)
        paradox_enum = self._parse_paradox_type(paradox_str)
        raw_steps = data.get("reasoning_chain", []) or []

        steps: List[ReasoningStep] = []
        for idx, s in enumerate(raw_steps, start=1):
            desc = str(s.get("description", "")).strip()
            conc = str(s.get("conclusion", "")).strip()
            conf = s.get("confidence", 0.7)
            try:
                conf = float(conf)
            except Exception:
                conf = 0.7
            conf = max(0.0, min(1.0, conf))
            if not desc and not conc:
                continue
            step_num = s.get("step_number", idx)
            try:
                step_num = int(step_num)
            except Exception:
                step_num = idx
            steps.append(
                ReasoningStep(
                    step_number=step_num,
                    description=desc or conc,
                    conclusion=conc or desc,
                    confidence=conf,
                )
            )

        # Timeline stability
        stability = data.get("timeline_stability", None)
        try:
            stability = float(stability)
        except Exception:
            stability = None

        if stability is None:
            stability = self.calculate_timeline_stability(paradox_enum, steps)

        res_strategies = data.get("resolution_strategies", []) or []
        res_strategies = [str(r).strip() for r in res_strategies if str(r).strip()]

        butterfly_effects = data.get("butterfly_effects", []) or []
        butterfly_effects = [str(b).strip() for b in butterfly_effects if str(b).strip()]

        final_recommendation = data.get("final_recommendation", "") or ""

        return ParadoxAnalysis(
            scenario=scenario,
            paradox_type=paradox_enum.value,
            reasoning_chain=steps,
            timeline_stability=max(0.0, min(1.0, stability)),
            resolution_strategies=res_strategies,
            butterfly_effects=butterfly_effects,
            final_recommendation=final_recommendation,
        )

    # ---------- SETUP CHAINS ----------

    def _setup_chains(self):
        """
        TODO #1: Set up three types of Chain of Thought prompting.

        Create:
        1. zero_shot_chain: Uses "Let's think step by step"
        2. few_shot_chain: Provides reasoning examples
        3. auto_cot_chain: Generates its own reasoning examples
        """

        # --- Zero-Shot CoT chain ---
        zero_shot_template = PromptTemplate.from_template(
            """You are a temporal paradox expert analyzing time travel scenarios.

Scenario: {scenario}

Let's think step by step.
Carefully analyze:
1. The sequence of events in time order.
2. All cause-and-effect relationships.
3. Whether any event prevents its own cause (paradox).
4. The most fitting paradox category, if any.
5. How stable the timeline is from 0 (collapses) to 1 (fully stable).
6. Possible resolution strategies and their pros/cons.
7. Likely butterfly-effect consequences.

Return ONLY valid JSON in this exact structure:
{{
  "paradox_type": "Grandfather Paradox | Bootstrap Paradox | Predestination Paradox | Butterfly Effect | Temporal Loop | Information Paradox | No Paradox",
  "timeline_stability": 0.0,
  "resolution_strategies": ["strategy 1", "strategy 2"],
  "butterfly_effects": ["effect 1", "effect 2"],
  "final_recommendation": "short recommendation",
  "reasoning_chain": [
    {{
      "step_number": 1,
      "description": "describe what you are analyzing in this step",
      "conclusion": "what you concluded in this step",
      "confidence": 0.9
    }}
  ]
}}

Do not add any extra text or explanation outside the JSON.

Analysis JSON:"""
        )

        self.zero_shot_chain = zero_shot_template | self.llm | StrOutputParser()

        # --- Few-Shot CoT chain with examples ---
        cot_examples = [
            {
                "scenario": "A person travels back and becomes their own grandfather.",
                "reasoning": """Step 1: Identify the causal loop where the traveler is both descendant and ancestor.
Step 2: Note that their existence depends on them causing their own birth.
Step 3: This creates a self-originating causal loop with no external starting point.
Step 4: Classify as a Bootstrap Paradox involving ancestry.
Step 5: Timeline is extremely unstable in a single-timeline model; more stable in multiverse or self-consistent models.""",
                "paradox": "Bootstrap Paradox",
                "stability": "0.1",
            },
            {
                "scenario": "A scientist sees a note on their desk with equations for time travel. Years later, they travel back and leave the same note on their younger self's desk.",
                "reasoning": """Step 1: The note with equations appears without a clear origin; it is passed from future to past.
Step 2: The information in the note never gets invented; it simply exists in a loop.
Step 3: This is an information loop where the cause of the knowledge is itself.
Step 4: Classify as an Information/Bootstrap Paradox.
Step 5: Timeline is moderately unstable unless a self-consistent timeline assumption is used.""",
                "paradox": "Information Paradox",
                "stability": "0.3",
            },
            {
                "scenario": "A traveler prevents a minor accident, which later leads to a different political leader being elected decades afterward.",
                "reasoning": """Step 1: Identify the initial change in the past: preventing a minor accident.
Step 2: Trace local effects: the people involved change their schedules and relationships.
Step 3: Secondary effects ripple into career paths and social networks.
Step 4: Tertiary effects influence voting behavior and political events years later.
Step 5: Classify as a Butterfly Effect, not a direct paradox.
Step 6: Timeline remains relatively stable but highly altered; stability is moderate.""",
                "paradox": "Butterfly Effect",
                "stability": "0.6",
            },
        ]

        example_prompt = PromptTemplate.from_template(
            """Scenario: {scenario}
Reasoning: {reasoning}
Paradox Type: {paradox}
Timeline Stability: {stability}"""
        )

        few_shot_prefix = """You are a time travel logician. You analyze scenarios using clear, numbered reasoning steps.
For each scenario, you:
- Describe the causal structure.
- Identify any paradox and classify it.
- Estimate timeline stability from 0 (collapses) to 1 (stable).
- Suggest resolution strategies.

Here are example analyses:
"""

        few_shot_suffix = """
Now analyze this new scenario in the same spirit, but return ONLY JSON.

Scenario: {scenario}

First, reason internally step by step, then summarize your reasoning as a structured JSON object:
{{
  "paradox_type": "Grandfather Paradox | Bootstrap Paradox | Predestination Paradox | Butterfly Effect | Temporal Loop | Information Paradox | No Paradox",
  "timeline_stability": 0.0,
  "resolution_strategies": ["strategy 1", "strategy 2"],
  "butterfly_effects": ["effect 1", "effect 2"],
  "final_recommendation": "short recommendation",
  "reasoning_chain": [
    {{
      "step_number": 1,
      "description": "describe what you are analyzing in this step",
      "conclusion": "what you concluded in this step",
      "confidence": 0.9
    }}
  ]
}}

Only output the JSON object."""

        few_shot_prompt = FewShotPromptTemplate(
            examples=cot_examples,
            example_prompt=example_prompt,
            prefix=few_shot_prefix,
            suffix=few_shot_suffix,
            input_variables=["scenario"],
        )

        self.few_shot_chain = few_shot_prompt | self.llm | StrOutputParser()

        # --- Auto-CoT chain: generate its own reasoning examples ---
        auto_cot_template = PromptTemplate.from_template(
            """You are designing example time travel paradox analyses for training.

Task: {task}

Generate 3 diverse examples of time travel scenarios with step-by-step reasoning.
Cover different paradox types (e.g., grandfather, bootstrap, predestination, butterfly).

Return ONLY valid JSON in this format:
[
  {{
    "scenario": "short scenario description",
    "reasoning": "multi-step reasoning text",
    "paradox_type": "type name",
    "timeline_stability": 0.0
  }}
]

JSON Output:"""
        )

        self.auto_cot_chain = auto_cot_template | self.llm | StrOutputParser()

    # ---------- ANALYSIS METHODS ----------

    def analyze_with_zero_shot_cot(self, scenario: str) -> ParadoxAnalysis:
        """
        TODO #2: Analyze scenario using zero-shot Chain of Thought.

        Args:
            scenario: Time travel scenario description

        Returns:
            Complete ParadoxAnalysis with reasoning steps
        """

        raw = self.zero_shot_chain.invoke({"scenario": scenario})
        analysis = self._parse_analysis_json(scenario, raw)
        return analysis

    def analyze_with_few_shot_cot(self, scenario: str) -> ParadoxAnalysis:
        """
        TODO #3: Analyze using few-shot CoT with reasoning examples.

        Args:
            scenario: Time travel scenario description

        Returns:
            Complete ParadoxAnalysis with detailed reasoning
        """

        raw = self.few_shot_chain.invoke({"scenario": scenario})
        analysis = self._parse_analysis_json(scenario, raw)
        return analysis

    def generate_auto_cot_examples(self, scenario_type: str) -> List[dict]:
        """
        TODO #4: Auto-generate CoT reasoning examples for a scenario type.

        Args:
            scenario_type: Type of scenarios to generate examples for

        Returns:
            List of generated examples with reasoning
        """

        raw = self.auto_cot_chain.invoke(
            {"task": f"create examples about {scenario_type} time travel scenarios"}
        )

        try:
            data = json.loads(raw)
        except Exception:
            data = []

        if not isinstance(data, list):
            data = []

        # Normalize each example to a simple dict
        examples: List[dict] = []
        for item in data:
            examples.append(
                {
                    "scenario": str(item.get("scenario", "")).strip(),
                    "reasoning": str(item.get("reasoning", "")).strip(),
                    "paradox_type": str(item.get("paradox_type", "")).strip(),
                    "timeline_stability": float(item.get("timeline_stability", 0.5))
                    if str(item.get("timeline_stability", "")).strip() != ""
                    else 0.5,
                }
            )

        return examples

    def calculate_timeline_stability(
        self, paradox_type: ParadoxType, reasoning_chain: List[ReasoningStep]
    ) -> float:
        """
        TODO #5: Calculate timeline stability based on paradox analysis.

        Args:
            paradox_type: Type of paradox detected
            reasoning_chain: Steps of reasoning

        Returns:
            Stability score from 0 (collapsed) to 1 (stable)
        """

        # Base stability by paradox severity
        base_map = {
            ParadoxType.NONE: 0.9,
            ParadoxType.BUTTERFLY: 0.6,
            ParadoxType.TEMPORAL_LOOP: 0.7,
            ParadoxType.PREDESTINATION: 0.4,
            ParadoxType.INFORMATION: 0.3,
            ParadoxType.BOOTSTRAP: 0.2,
            ParadoxType.GRANDFATHER: 0.1,
        }

        base = base_map.get(paradox_type, 0.5)

        if reasoning_chain:
            avg_conf = sum(s.confidence for s in reasoning_chain) / len(reasoning_chain)
        else:
            avg_conf = 0.7

        stability = base * avg_conf
        return max(0.0, min(1.0, stability))

    def trace_butterfly_effects(self, scenario: str, initial_change: str) -> List[str]:
        """
        TODO #6: Trace potential butterfly effects from a change.

        Args:
            scenario: Original scenario
            initial_change: The change made in the past

        Returns:
            List of potential consequences
        """

        template = PromptTemplate.from_template(
            """You are tracing butterfly-effect consequences in a time travel scenario.

Original scenario:
{scenario}

Initial change in the past:
{initial_change}

Think step by step about:
- Immediate local consequences
- Secondary social or technological effects
- Long-term historical or global changes

Then return ONLY valid JSON:
{{
  "effects": [
    "primary effect description",
    "secondary effect description",
    "tertiary effect description"
  ]
}}"""
        )

        chain = template | self.llm | StrOutputParser()
        raw = chain.invoke({"scenario": scenario, "initial_change": initial_change})

        try:
            data = json.loads(raw)
        except Exception:
            data = {}

        effects = data.get("effects", []) or []
        effects = [str(e).strip() for e in effects if str(e).strip()]

        return effects

    def resolve_paradox(self, analysis: ParadoxAnalysis) -> Dict[str, any]:
        """
        TODO #7: Propose resolution strategies for detected paradox.

        Args:
            analysis: The paradox analysis

        Returns:
            Resolution plan with strategies and success probability
        """

        resolution_template = PromptTemplate.from_template(
            """You are a time travel theorist designing a plan to resolve a paradox.

Paradox Analysis (JSON):
{analysis_json}

Based on this analysis:
- Choose a primary resolution strategy (Multiverse Branch, Self-Consistent Timeline, Paradox Avoidance, Accept Consequences, Timeline Correction).
- Suggest 1-3 alternative strategies.
- Describe 3-5 concrete implementation steps for the primary strategy.
- Estimate success probability from 0 to 1.
- List key risks.

Return ONLY valid JSON:
{{
  "primary_strategy": "one of the strategy names",
  "alternative_strategies": ["strategy 1", "strategy 2"],
  "implementation_steps": ["step 1", "step 2"],
  "success_probability": 0.0,
  "risks": ["risk 1", "risk 2"]
}}"""
        )

        chain = resolution_template | self.llm | StrOutputParser()

        analysis_json = json.dumps(
            {
                "scenario": analysis.scenario,
                "paradox_type": analysis.paradox_type,
                "timeline_stability": analysis.timeline_stability,
                "resolution_strategies": analysis.resolution_strategies,
                "butterfly_effects": analysis.butterfly_effects,
                "final_recommendation": analysis.final_recommendation,
                "reasoning_chain": [asdict(s) for s in analysis.reasoning_chain],
            },
            ensure_ascii=False,
        )

        raw = chain.invoke({"analysis_json": analysis_json})

        try:
            data = json.loads(raw)
        except Exception:
            data = {
                "primary_strategy": ResolutionStrategy.MULTIVERSE.value,
                "alternative_strategies": [ResolutionStrategy.SELF_CONSISTENT.value],
                "implementation_steps": [
                    "Allow the paradoxical event to create a new branch timeline.",
                    "Isolate the new branch from the original timeline.",
                ],
                "success_probability": 0.7,
                "risks": ["Unknown side-effects of branching timelines."],
            }

        primary = data.get("primary_strategy", ResolutionStrategy.MULTIVERSE.value)
        alt = data.get("alternative_strategies", []) or []
        steps = data.get("implementation_steps", []) or []
        risks = data.get("risks", []) or []
        prob = data.get("success_probability", 0.5)

        try:
            prob = float(prob)
        except Exception:
            prob = 0.5
        prob = max(0.0, min(1.0, prob))

        resolution = {
            "primary_strategy": primary,
            "alternative_strategies": [str(a).strip() for a in alt if str(a).strip()],
            "implementation_steps": [str(s).strip() for s in steps if str(s).strip()],
            "success_probability": prob,
            "risks": [str(r).strip() for r in risks if str(r).strip()],
        }

        return resolution

    def compare_cot_methods(self, scenario: str) -> Dict[str, any]:
        """
        TODO #8 (Bonus): Compare all three CoT methods on the same scenario.

        Args:
            scenario: Scenario to analyze

        Returns:
            Comparison of methods with metrics
        """

        # Zero-shot analysis
        zs = self.analyze_with_zero_shot_cot(scenario)

        # Few-shot analysis
        fs = self.analyze_with_few_shot_cot(scenario)

        # Auto-CoT generated examples (used as proxy for diversity)
        auto_examples = self.generate_auto_cot_examples("general")

        # Choose "best" as the one with richer reasoning (more steps)
        zs_steps = len(zs.reasoning_chain)
        fs_steps = len(fs.reasoning_chain)

        if fs_steps > zs_steps:
            best = "few_shot"
            reason = "Few-shot produced a richer reasoning chain, guided by examples."
        elif zs_steps > fs_steps:
            best = "zero_shot"
            reason = "Zero-shot produced a more detailed reasoning chain for this scenario."
        else:
            best = "few_shot"
            reason = "Both methods were similar; few-shot is preferred for its example-guided structure."

        comparison = {
            "zero_shot": {
                "paradox_type": zs.paradox_type,
                "timeline_stability": zs.timeline_stability,
                "num_steps": zs_steps,
            },
            "few_shot": {
                "paradox_type": fs.paradox_type,
                "timeline_stability": fs.timeline_stability,
                "num_steps": fs_steps,
            },
            "auto_cot": {
                "num_generated_examples": len(auto_examples),
            },
            "best_method": best,
            "reasoning": reason,
        }

        return comparison


def test_paradox_resolver():
    """Test the paradox resolver with various time travel scenarios."""

    resolver = ParadoxResolver()

    # Test scenarios of increasing complexity
    test_scenarios = [
        {
            "name": "The Coffee Shop Meeting",
            "scenario": "Sarah travels back 20 years and accidentally spills coffee on her father, preventing him from meeting her mother at their destined encounter.",
        },
        {
            "name": "The Invention Loop",
            "scenario": "An inventor receives blueprints from their future self for a time machine, builds it, then travels back to give themselves the blueprints.",
        },
        {
            "name": "The Butterfly War",
            "scenario": "A time traveler steps on a butterfly in prehistoric times. When they return, they find their country never existed and a different nation rules the world.",
        },
        {
            "name": "The Prophet's Dilemma",
            "scenario": "Someone travels forward 10 years, learns about a disaster, returns to prevent it, but their warnings are what actually cause the disaster.",
        },
        {
            "name": "The Timeline Splice",
            "scenario": "Two time travelers from different futures arrive in 2024, each trying to ensure their timeline becomes the 'true' future.",
        },
    ]

    print("⏰ TIME TRAVEL PARADOX RESOLVER ⏰")
    print("=" * 70)

    for test_case in test_scenarios:
        print(f"\n🌀 Scenario: {test_case['name']}")
        print(f"📖 Description: \"{test_case['scenario'][:80]}...\"")

        # Test Zero-Shot CoT
        print("\n🔷 Zero-Shot Chain of Thought:")
        zs_analysis = resolver.analyze_with_zero_shot_cot(test_case["scenario"])

        print(f"  Paradox Type: {zs_analysis.paradox_type}")
        print(f"  Timeline Stability: {zs_analysis.timeline_stability:.1%}")

        if zs_analysis.reasoning_chain:
            print("  Reasoning Steps:")
            for step in zs_analysis.reasoning_chain[:3]:  # Show first 3 steps
                print(f"    {step.step_number}. {step.description}")

        # Test Few-Shot CoT
        print("\n🔶 Few-Shot Chain of Thought:")
        fs_analysis = resolver.analyze_with_few_shot_cot(test_case["scenario"])

        print(f"  Paradox Type: {fs_analysis.paradox_type}")
        print(f"  Timeline Stability: {fs_analysis.timeline_stability:.1%}")

        if fs_analysis.resolution_strategies:
            print("  Resolution Strategies:")
            for strategy in fs_analysis.resolution_strategies[:2]:
                print(f"    • {strategy}")

        # Show butterfly effects
        if fs_analysis.butterfly_effects:
            print("  Butterfly Effects:")
            for effect in fs_analysis.butterfly_effects[:2]:
                print(f"    🦋 {effect}")

        print("-" * 70)

    # Test method comparison
    print("\n📊 METHOD COMPARISON TEST:")
    print("=" * 70)

    comparison_scenario = "A person travels back and gives Shakespeare the complete works of Shakespeare, which he then 'writes'."

    print(f"Scenario: {comparison_scenario}")
    comparison = resolver.compare_cot_methods(comparison_scenario)

    print(f"\n🏆 Best Method: {comparison.get('best_method', 'Unknown')}")
    print(f"Reasoning: {comparison.get('reasoning', 'No comparison available')}")

    # Test butterfly effect tracing
    print("\n🦋 BUTTERFLY EFFECT ANALYSIS:")
    print("=" * 70)

    effects = resolver.trace_butterfly_effects(
        "Time traveler prevents a minor car accident in 1990",
        "Driver doesn't meet future spouse at hospital",
    )

    if effects:
        print("Traced Consequences:")
        for i, effect in enumerate(effects[:5], 1):
            print(f"  {i}. {effect}")


if __name__ == "__main__":
    # Make sure to set OPENAI_API_KEY environment variable
    if not os.environ.get("OPENAI_API_KEY"):
        print("⚠️ Please set OPENAI_API_KEY environment variable")
    else:
        test_paradox_resolver()
