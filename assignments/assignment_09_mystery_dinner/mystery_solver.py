"""
Assignment 9: Mystery Dinner Party Solver
All Concepts - Solve murder mysteries using every prompting technique

Your mission: Become the ultimate AI detective by combining all prompting
techniques to solve complex murder mysteries!
"""

import os
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_core.output_parsers import StrOutputParser


@dataclass
class Suspect:
    name: str
    background: str
    alibi: str
    motive: str
    opportunity: bool
    suspicious_behavior: List[str]


@dataclass
class Clue:
    description: str
    location: str
    time_found: str
    related_suspects: List[str]
    significance: str


@dataclass
class MysteryCase:
    victim: str
    crime_scene: str
    time_of_death: str
    suspects: List[Suspect]
    clues: List[Clue]
    witness_statements: List[str]


@dataclass
class Solution:
    murderer: str
    motive: str
    method: str
    reasoning_chain: List[str]
    evidence_links: Dict[str, str]
    confidence: float
    alternative_theories: List[str]


class MysteryDetective:
    """
    AI detective using all prompting techniques to solve mysteries.
    """

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(model=model_name, temperature=0.3)
        self.profiler = None  # Zero-shot
        self.clue_analyzer = None  # Few-shot
        self.timeline_builder = None  # CoT
        self.solver = None  # Combined
        self._setup_chains()

    def _setup_chains(self):
        """
        Set up chains for each aspect of mystery solving.

        1. Zero-shot profiler for psychological analysis
        2. Few-shot clue analyzer with pattern examples
        3. CoT timeline builder for alibi checking
        4. Combined solver for final deduction
        """

        # ---------- Zero-shot for suspect profiling ----------
        profile_template = PromptTemplate.from_template(
            """You are a calm, professional criminal psychologist.

Your job:
1. Analyze the suspect's background, alibi, motive, opportunity, and suspicious behavior.
2. Estimate how likely they are to be deceptive.
3. Estimate how strong their motive is, compared to a typical murder case.
4. Write a short psychological profile (2–4 sentences) in neutral, respectful language.

Return ONLY valid JSON with this structure:
{{
  "deception_likelihood": 0.0,
  "motive_strength": 0.0,
  "psychological_profile": "string"
}}

Where:
- deception_likelihood is between 0 and 1
- motive_strength is between 0 and 1

Suspect Info:
{name}: {suspect_info}

Analysis JSON:"""
        )

        self.profiler = profile_template | self.llm | StrOutputParser()

        # ---------- Few-shot for clue patterns ----------
        clue_examples = [
            {
                "clue": "Lipstick on wine glass",
                "analysis": "Suggests a female-presenting guest or someone wearing lipstick drank from this glass. Compare shade to suspects' lipstick.",
                "significance": "medium",
            },
            {
                "clue": "Broken watch stopped at 10:32 PM",
                "analysis": "Likely marks time of struggle or impact around 10:32 PM. Cross-check with time of death and alibis.",
                "significance": "high",
            },
            {
                "clue": "Mud footprints leading from garden to library window",
                "analysis": "Indicates entry or exit via garden. Focus on suspects who were supposedly inside at that time.",
                "significance": "high",
            },
            {
                "clue": "Threatening letter in victim's desk",
                "analysis": "Shows prior conflict and potential premeditation. Check handwriting and who knew about the victim's secrets.",
                "significance": "medium",
            },
        ]

        clue_example_prompt = PromptTemplate.from_template(
            """Clue: {clue}
Analysis: {analysis}
Significance: {significance}"""
        )

        clue_prefix = """You are analyzing clues in a murder mystery.

For each clue:
- Explain what it might mean.
- Explain how it should be used in the investigation.
- Rate its significance (low, medium, high, critical).

Here are some examples:
"""

        clue_suffix = """
Now analyze this new clue:

Clue: {clue_text}

Return ONLY valid JSON:
{{
  "analysis": "short explanation of meaning and use",
  "significance": "low | medium | high | critical",
  "implicated_suspects": ["suspect names that this clue points toward, if any"],
  "reasoning": "brief step-by-step reasoning"
}}"""

        clue_few_shot = FewShotPromptTemplate(
            examples=clue_examples,
            example_prompt=clue_example_prompt,
            prefix=clue_prefix,
            suffix=clue_suffix,
            input_variables=["clue_text"],
        )

        self.clue_analyzer = clue_few_shot | self.llm | StrOutputParser()

        # ---------- CoT for timeline reconstruction ----------
        timeline_template = PromptTemplate.from_template(
            """You are reconstructing the timeline of a murder at a dinner party.

Use:
- Alibis for each suspect
- Official time of death
- Witness statements

Steps:
1. List the key times (before, during, and after time of death).
2. Place each suspect at likely locations at those times.
3. Identify contradictions between alibis and witness statements.
4. Decide whether each suspect's alibi is likely VERIFIED or BROKEN.

Return ONLY valid JSON:
{{
  "alibis_verified": {{
    "Suspect Name": true,
    "Other Suspect": false
  }},
  "reasoning_steps": [
    "step 1 ...",
    "step 2 ..."
  ]
}}

Alibis:
{alibis}

Time of Death: {tod}

Witness Statements:
{witnesses}

Let's reason it out step by step, then output the JSON."""
        )

        self.timeline_builder = timeline_template | self.llm | StrOutputParser()

        # ---------- Combined solver for final deduction ----------
        solver_template = PromptTemplate.from_template(
            """You are the lead detective solving a classic murder mystery.

You are given:

Case (JSON):
{case_json}

Suspect Profiles (JSON):
{profiles_json}

Clue Analysis (JSON):
{clues_json}

Alibi Verification (JSON):
{alibis_json}

Your job:
1. Think through each suspect's motive, means, and opportunity.
2. Weigh psychological profiles and suspicious behaviors.
3. Connect clues to suspects (who does each clue support or contradict?).
4. Use the alibi verification to rule suspects in or out.
5. Form a clear, logical reasoning chain to identify the most likely murderer.
6. Consider at least one alternative theory (even if you think it's less likely).

Return ONLY valid JSON in this structure:
{{
  "murderer": "name of the most likely killer",
  "motive": "short explanation of why they killed",
  "method": "short explanation of how the murder was carried out",
  "reasoning_chain": [
    "step 1 ...",
    "step 2 ...",
    "step 3 ..."
  ],
  "evidence_links": {{
     "clue_or_fact": "how it supports or contradicts the conclusion"
  }},
  "confidence": 0.0,
  "alternative_theories": [
    "short description of an alternate suspect or explanation"
  ]
}}"""
        )

        self.solver = solver_template | self.llm | StrOutputParser()

    # ----------------- Core methods -----------------

    def profile_suspect(self, suspect: Suspect) -> Dict[str, any]:
        """
        Profile suspect using zero-shot analysis.
        Psychological profiling without examples.
        """

        info = (
            f"Name: {suspect.name}\n"
            f"Background: {suspect.background}\n"
            f"Alibi: {suspect.alibi}\n"
            f"Motive: {suspect.motive}\n"
            f"Opportunity: {suspect.opportunity}\n"
            f"Suspicious behavior: {', '.join(suspect.suspicious_behavior)}"
        )

        raw = self.profiler.invoke(
            {"name": suspect.name, "suspect_info": info}
        )

        # Parse JSON robustly
        data: Dict[str, any] = {}
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
            else:
                data = {}

        deception = data.get("deception_likelihood", 0.5)
        motive_strength = data.get("motive_strength", 0.5)
        profile_text = data.get("psychological_profile", "")

        try:
            deception = float(deception)
        except Exception:
            deception = 0.5
        try:
            motive_strength = float(motive_strength)
        except Exception:
            motive_strength = 0.5

        deception = max(0.0, min(1.0, deception))
        motive_strength = max(0.0, min(1.0, motive_strength))

        return {
            "deception_likelihood": deception,
            "motive_strength": motive_strength,
            "psychological_profile": profile_text.strip(),
        }

    def analyze_clues(self, clues: List[Clue]) -> List[Dict[str, any]]:
        """
        Analyze clues using few-shot pattern matching.
        Match against known clue patterns.
        """

        results: List[Dict[str, any]] = []

        for clue in clues:
            clue_text = (
                f"Description: {clue.description}\n"
                f"Location: {clue.location}\n"
                f"Time found: {clue.time_found}\n"
                f"Related suspects: {', '.join(clue.related_suspects)}\n"
                f"Original significance note: {clue.significance}"
            )

            raw = self.clue_analyzer.invoke({"clue_text": clue_text})

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
                else:
                    data = {}

            analysis = data.get("analysis", "").strip()
            significance = data.get("significance", "").strip().lower() or "medium"
            implicated = data.get("implicated_suspects", []) or []
            reasoning = data.get("reasoning", "").strip()

            results.append(
                {
                    "clue_description": clue.description,
                    "analysis": analysis,
                    "significance": significance,
                    "implicated_suspects": implicated,
                    "reasoning": reasoning,
                }
            )

        return results

    def verify_alibis(self, case: MysteryCase) -> Dict[str, bool]:
        """
        Verify alibis using CoT timeline reasoning.
        Step-by-step timeline reconstruction.
        """

        alibi_lines = []
        for s in case.suspects:
            alibi_lines.append(f"{s.name}: {s.alibi}")
        alibis_text = "\n".join(alibi_lines)

        witnesses_text = "\n".join(case.witness_statements)

        raw = self.timeline_builder.invoke(
            {
                "alibis": alibis_text,
                "tod": case.time_of_death,
                "witnesses": witnesses_text,
            }
        )

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
            else:
                data = {}

        verified_map = data.get("alibis_verified", {}) or {}

        result: Dict[str, bool] = {}
        for s in case.suspects:
            val = verified_map.get(s.name, None)
            result[s.name] = bool(val) if val is not None else False

        return result

    def solve_mystery(self, case: MysteryCase) -> Solution:
        """
        Solve the mystery using ALL techniques.

        Combine all methods for final solution.
        """

        # 1. Profile all suspects (zero-shot)
        profiles: Dict[str, Dict[str, any]] = {}
        for s in case.suspects:
            profiles[s.name] = self.profile_suspect(s)

        # 2. Analyze clues (few-shot)
        clue_analysis = self.analyze_clues(case.clues)

        # 3. Verify alibis (CoT)
        #    a) simple mapping for printing
        alibi_verification = self.verify_alibis(case)

        #    b) richer JSON with reasoning, for the solver
        alibi_lines = [f"{s.name}: {s.alibi}" for s in case.suspects]
        alibis_text = "\n".join(alibi_lines)
        witnesses_text = "\n".join(case.witness_statements)
        raw_alibis = self.timeline_builder.invoke(
            {
                "alibis": alibis_text,
                "tod": case.time_of_death,
                "witnesses": witnesses_text,
            }
        )
        try:
            alibis_json_data = json.loads(raw_alibis)
        except Exception:
            start = raw_alibis.find("{")
            end = raw_alibis.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    alibis_json_data = json.loads(raw_alibis[start : end + 1])
                except Exception:
                    alibis_json_data = {}
            else:
                alibis_json_data = {}

        # 4. Combine evidence (all) via solver chain
        case_dict = {
            "victim": case.victim,
            "crime_scene": case.crime_scene,
            "time_of_death": case.time_of_death,
            "suspects": [asdict(s) for s in case.suspects],
            "clues": [asdict(c) for c in case.clues],
            "witness_statements": case.witness_statements,
        }

        case_json = json.dumps(case_dict, ensure_ascii=False)
        profiles_json = json.dumps(profiles, ensure_ascii=False)
        clues_json = json.dumps(clue_analysis, ensure_ascii=False)
        alibis_json = json.dumps(alibis_json_data, ensure_ascii=False)

        raw_solution = self.solver.invoke(
            {
                "case_json": case_json,
                "profiles_json": profiles_json,
                "clues_json": clues_json,
                "alibis_json": alibis_json,
            }
        )

        # Parse solution JSON
        try:
            sol_data = json.loads(raw_solution)
        except Exception:
            start = raw_solution.find("{")
            end = raw_solution.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    sol_data = json.loads(raw_solution[start : end + 1])
                except Exception:
                    sol_data = {}
            else:
                sol_data = {}

        murderer = sol_data.get("murderer", "") or ""
        motive = sol_data.get("motive", "") or ""
        method = sol_data.get("method", "") or ""
        reasoning_chain = sol_data.get("reasoning_chain", []) or []
        evidence_links = sol_data.get("evidence_links", {}) or {}
        confidence = sol_data.get("confidence", 0.7)
        alternative_theories = sol_data.get("alternative_theories", []) or []

        try:
            confidence = float(confidence)
        except Exception:
            confidence = 0.7
        confidence = max(0.0, min(1.0, confidence))

        # Ensure reasoning_chain is a list of strings
        reasoning_chain = [str(r).strip() for r in reasoning_chain if str(r).strip()]
        # Ensure evidence_links keys/values are strings
        evidence_links = {
            str(k): str(v).strip()
            for k, v in evidence_links.items()
            if str(k).strip()
        }
        alternative_theories = [
            str(a).strip() for a in alternative_theories if str(a).strip()
        ]

        return Solution(
            murderer=murderer,
            motive=motive,
            method=method,
            reasoning_chain=reasoning_chain,
            evidence_links=evidence_links,
            confidence=confidence,
            alternative_theories=alternative_theories,
        )


def test_detective():
    detective = MysteryDetective()

    # Create a test mystery case
    test_case = MysteryCase(
        victim="Lord Wellington",
        crime_scene="Library",
        time_of_death="10:30 PM",
        suspects=[
            Suspect(
                name="Lady Scarlett",
                background="Victim's wife, inherits estate",
                alibi="In the garden with guests",
                motive="Inheritance and secret affair",
                opportunity=True,
                suspicious_behavior=["Nervous", "Changed story twice"],
            ),
            Suspect(
                name="Professor Plum",
                background="Business partner, recent disputes",
                alibi="In study reviewing documents",
                motive="Business betrayal",
                opportunity=True,
                suspicious_behavior=["Destroyed papers after murder"],
            ),
            Suspect(
                name="Colonel Mustard",
                background="Old friend, owes money",
                alibi="Playing billiards with butler",
                motive="Gambling debts",
                opportunity=False,
                suspicious_behavior=["Attempted to leave early"],
            ),
        ],
        clues=[
            Clue(
                description="Poison bottle hidden in bookshelf",
                location="Library",
                time_found="11:00 PM",
                related_suspects=["Lady Scarlett", "Professor Plum"],
                significance="Murder weapon",
            ),
            Clue(
                description="Love letter from unknown person",
                location="Victim's pocket",
                time_found="10:45 PM",
                related_suspects=["Lady Scarlett"],
                significance="Possible motive",
            ),
        ],
        witness_statements=[
            "Butler saw Professor Plum near library at 10:15 PM",
            "Maid heard argument from library at 10:20 PM",
            "Guest saw Lady Scarlett in garden until 10:25 PM",
        ],
    )

    print("🕵️ MYSTERY DINNER PARTY SOLVER 🕵️")
    print("=" * 70)
    print(f"Victim: {test_case.victim}")
    print(f"Scene: {test_case.crime_scene}")
    print(f"Time of Death: {test_case.time_of_death}")
    print("-" * 70)

    # Test each component
    print("\n🔍 SUSPECT PROFILES (Zero-shot):")
    for suspect in test_case.suspects:
        profile = detective.profile_suspect(suspect)
        print(f"\n{suspect.name}:")
        print(f"  Deception: {profile.get('deception_likelihood', 0):.0%}")
        print(f"  Motive Strength: {profile.get('motive_strength', 0):.0%}")

    print("\n🔎 CLUE ANALYSIS (Few-shot):")
    clue_analysis = detective.analyze_clues(test_case.clues)
    for i, clue in enumerate(test_case.clues):
        print(f"  • {clue.description}")
        if i < len(clue_analysis):
            print(f"    → {clue_analysis[i].get('analysis', '')[:80]}...")

    print("\n⏰ ALIBI VERIFICATION (Chain of Thought):")
    alibi_results = detective.verify_alibis(test_case)
    for name, verified in alibi_results.items():
        status = "✓ Verified" if verified else "✗ Suspicious"
        print(f"  {name}: {status}")

    print("\n🎯 FINAL SOLUTION (All Techniques):")
    print("=" * 70)
    solution = detective.solve_mystery(test_case)

    print(f"The Murderer: {solution.murderer}")
    print(f"Motive: {solution.motive}")
    print(f"Method: {solution.method}")
    print(f"Confidence: {solution.confidence:.0%}")

    if solution.reasoning_chain:
        print("\nReasoning:")
        for step in solution.reasoning_chain[:3]:
            print(f"  → {step}")


if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print("⚠️ Please set OPENAI_API_KEY")
    else:
        test_detective()
