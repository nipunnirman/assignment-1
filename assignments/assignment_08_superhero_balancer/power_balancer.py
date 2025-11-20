"""
Assignment 8: Superhero Power Balancer
All Concepts Combined - Master all prompting techniques together

Your mission: Balance superhero powers for the ultimate fighting game
using every prompting technique you've learned!
"""

import os
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_core.output_parsers import StrOutputParser


class PowerType(Enum):
    PHYSICAL = "physical"
    ENERGY = "energy"
    MENTAL = "mental"
    REALITY = "reality"
    TECH = "technology"
    MAGIC = "magic"


@dataclass
class Hero:
    name: str
    abilities: List[str]
    power_type: str
    power_level: float
    weaknesses: List[str]
    synergies: List[str]


@dataclass
class BalanceReport:
    hero: Hero
    analysis_method: str  # Which prompting method was used
    power_rating: float
    balance_issues: List[str]
    suggested_changes: List[str]
    team_synergies: Dict[str, float]
    counter_picks: List[str]


class PowerBalancer:
    """
    AI-powered game balancer using all prompting techniques.
    Combines zero-shot, few-shot, and CoT for comprehensive analysis.
    """

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(model=model_name, temperature=0.4)
        self.ability_analyzer = None  # Zero-shot
        self.type_classifier = None  # Few-shot
        self.interaction_calculator = None  # CoT
        self.balance_detector = None  # Combined
        self._setup_chains()

    def _setup_chains(self):
        """
        Set up chains for each prompting technique.

        1. Zero-shot for novel ability analysis
        2. Few-shot for power type classification
        3. CoT for interaction calculations
        4. Combined for balance detection
        """

        # ---------- Zero-shot for ability analysis ----------
        ability_template = PromptTemplate.from_template(
            """You are balancing a competitive superhero fighting game.

Your task is to analyze ONE ability for game balance.

For this single ability:
- Estimate its raw power level from 0 to 10 (0 = almost useless, 10 = completely broken).
- List potential exploits (ways players might abuse this ability to break the game).
- List reasonable counter-play options (how other heroes or mechanics could answer it).

Return ONLY valid JSON in this exact structure:
{{
  "power_level": 0.0,
  "exploits": ["string"],
  "counters": ["string"]
}}

Ability: {ability_description}

Analysis JSON:"""
        )
        self.ability_analyzer = ability_template | self.llm | StrOutputParser()

        # ---------- Few-shot for power classification ----------
        type_examples = [
            {
                "abilities": "Super strength, indestructible skin, shockwave punches",
                "type": "physical",
                "reasoning": "All abilities directly enhance the body and melee impact.",
            },
            {
                "abilities": "Energy blasts, laser beams, plasma shields",
                "type": "energy",
                "reasoning": "Projectile and shield effects made of raw energy.",
            },
            {
                "abilities": "Telepathy, mind control, psychic shields",
                "type": "mental",
                "reasoning": "Directly affect thoughts and minds rather than bodies.",
            },
            {
                "abilities": "Time stop, reality rewriting, pocket dimensions",
                "type": "reality",
                "reasoning": "Change the rules of reality and time itself.",
            },
            {
                "abilities": "Powered armor suit, drones, advanced gadgets",
                "type": "technology",
                "reasoning": "Relies on devices, engineering, and equipment.",
            },
            {
                "abilities": "Spellcasting, elemental summons, arcane wards",
                "type": "magic",
                "reasoning": "Uses mystic or supernatural forces via spells/rituals.",
            },
        ]

        type_example_prompt = PromptTemplate.from_template(
            """Abilities: {abilities}
Type: {type}
Reasoning: {reasoning}"""
        )

        type_prefix = """You are classifying heroes into power types for a fighting game.

Valid power types:
- physical
- energy
- mental
- reality
- technology
- magic

Use the examples to understand the pattern.

Examples:
"""

        type_suffix = """
Now classify this new hero based on their abilities.

Abilities: {abilities_text}

Return ONLY valid JSON:
{{
  "type": "physical | energy | mental | reality | technology | magic",
  "reasoning": "short explanation"
}}"""

        type_few_shot = FewShotPromptTemplate(
            examples=type_examples,
            example_prompt=type_example_prompt,
            prefix=type_prefix,
            suffix=type_suffix,
            input_variables=["abilities_text"],
        )

        self.type_classifier = type_few_shot | self.llm | StrOutputParser()

        # ---------- CoT for interaction calculations ----------
        interaction_template = PromptTemplate.from_template(
            """You are analyzing team synergy between two heroes in a fighting game.

Ability set of Hero 1:
{ability1}

Ability set of Hero 2:
{ability2}

Let's think step by step about their interaction:
- How well do their abilities combo together offensively?
- How well do they cover each other's weaknesses defensively?
- Do they overlap too much, or do they complement each other?

After the reasoning, output a final line in this exact format:
Synergy score: <number between 0 and 1>

Where 0 means "no synergy at all" and 1 means "perfect synergy"."""
        )

        self.interaction_calculator = (
            interaction_template | self.llm | StrOutputParser()
        )

        # ---------- Combined approach for balance detection ----------
        balance_template = PromptTemplate.from_template(
            """You are the lead balance designer for a superhero fighting game.

You are given:
1) The hero to analyze (JSON):
{hero_json}

2) The rest of the meta roster (JSON list of heroes):
{meta_json}

3) Precomputed synergy scores between this hero and others (JSON dict name->score 0-1):
{synergy_json}

Task:
- Think step by step about:
  * How strong this hero is overall (mobility, damage, defense, utility, weaknesses).
  * How they perform in the current meta against common archetypes.
  * Where they might be overpowered or underpowered.
  * Which heroes they pair well with (using the synergy scores).
  * Which heroes or archetypes naturally counter them.

Then:
Return ONLY valid JSON in this structure:
{{
  "power_rating": 0.0,
  "balance_issues": ["string"],
  "suggested_changes": ["string"],
  "team_synergies": {{
    "HeroName": 0.0
  }},
  "counter_picks": ["HeroName or archetype"]
}}

Where:
- power_rating is from 0 to 10 (5 is ideal/average).
- balance_issues can be empty if mostly fine.
- suggested_changes are concrete, small tweaks (e.g., 'reduce stun duration on time stop').
- team_synergies contains only a few of the strongest partners.
- counter_picks are heroes/archetypes that likely beat this hero.

Think through the reasoning internally, then output ONLY the JSON."""
        )

        self.balance_detector = balance_template | self.llm | StrOutputParser()

    # ----------------- Helper -----------------

    def _normalize_type(self, raw: str) -> str:
        text = (raw or "").strip().lower()
        for pt in PowerType:
            if text == pt.value:
                return pt.value
        # fuzzy
        if "tech" in text or "gadget" in text or "armor" in text:
            return PowerType.TECH.value
        if "magic" in text or "spell" in text:
            return PowerType.MAGIC.value
        if "mind" in text or "mental" in text or "psych" in text:
            return PowerType.MENTAL.value
        if "energy" in text or "laser" in text or "plasma" in text:
            return PowerType.ENERGY.value
        if "reality" in text or "time" in text or "dimension" in text:
            return PowerType.REALITY.value
        return PowerType.PHYSICAL.value

    # ----------------- Core methods -----------------

    def analyze_hero_zero_shot(self, hero: Hero) -> Dict[str, any]:
        """
        Analyze hero abilities using zero-shot prompting.
        For novel, unique abilities without examples.
        """

        total_power = 0.0
        exploits: List[str] = []
        counters: List[str] = []

        if not hero.abilities:
            return {"power_level": 0.0, "exploits": [], "counters": []}

        for ability in hero.abilities:
            try:
                raw = self.ability_analyzer.invoke(
                    {"ability_description": ability}
                )
                data = json.loads(raw)
            except Exception:
                # try to salvage JSON
                try:
                    start = raw.find("{")
                    end = raw.rfind("}")
                    if start != -1 and end != -1:
                        data = json.loads(raw[start : end + 1])
                    else:
                        data = {}
                except Exception:
                    data = {}

            lvl = data.get("power_level", 5.0)
            try:
                lvl = float(lvl)
            except Exception:
                lvl = 5.0
            total_power += max(0.0, min(10.0, lvl))

            exploits.extend(data.get("exploits", []) or [])
            counters.extend(data.get("counters", []) or [])

        avg_power = total_power / len(hero.abilities)
        exploits = [e.strip() for e in exploits if str(e).strip()]
        counters = [c.strip() for c in counters if str(c).strip()]

        return {"power_level": avg_power, "exploits": exploits, "counters": counters}

    def classify_power_few_shot(self, abilities: List[str]) -> str:
        """
        Classify power type using few-shot examples.
        """

        abilities_text = "; ".join(abilities) if abilities else ""
        try:
            raw = self.type_classifier.invoke({"abilities_text": abilities_text})
            data = json.loads(raw)
        except Exception:
            data = {}

        raw_type = data.get("type", "")
        return self._normalize_type(raw_type)

    def calculate_synergy_cot(self, hero1: Hero, hero2: Hero) -> float:
        """
        Calculate team synergy using Chain of Thought.
        """

        abilities1 = ", ".join(hero1.abilities)
        abilities2 = ", ".join(hero2.abilities)

        raw = self.interaction_calculator.invoke(
            {"ability1": abilities1, "ability2": abilities2}
        )

        # Look for "Synergy score: X"
        score = 0.5
        try:
            marker = "Synergy score:"
            if marker in raw:
                tail = raw.split(marker, 1)[1].strip()
                token = tail.split()[0]
                try:
                    score = float(token)
                except Exception:
                    cleaned = "".join(ch for ch in tail if ch.isdigit() or ch == ".")
                    if cleaned:
                        score = float(cleaned)
            else:
                cleaned = "".join(ch for ch in raw if ch.isdigit() or ch == ".")
                if cleaned:
                    score = float(cleaned)
        except Exception:
            score = 0.5

        score = max(0.0, min(1.0, score))
        return score

    def detect_imbalance_combined(self, hero: Hero, meta: List[Hero]) -> BalanceReport:
        """
        Detect balance issues using ALL techniques.
        """

        # 1) Zero-shot ability analysis
        ability_analysis = self.analyze_hero_zero_shot(hero)
        hero_power = ability_analysis.get("power_level", 5.0)

        # 2) Few-shot power type classification
        classified_type = self.classify_power_few_shot(hero.abilities)
        hero.power_type = classified_type
        hero.power_level = hero_power

        # 3) CoT synergy calculation with each other hero
        synergy_scores: Dict[str, float] = {}
        for other in meta:
            if other.name == hero.name:
                continue
            synergy_scores[other.name] = self.calculate_synergy_cot(hero, other)

        # 4) Combined balance detection with meta + synergies
        hero_json = json.dumps(hero.__dict__, ensure_ascii=False)
        meta_json = json.dumps([h.__dict__ for h in meta], ensure_ascii=False)
        synergy_json = json.dumps(synergy_scores, ensure_ascii=False)

        raw = self.balance_detector.invoke(
            {
                "hero_json": hero_json,
                "meta_json": meta_json,
                "synergy_json": synergy_json,
            }
        )

        try:
            data = json.loads(raw)
        except Exception:
            try:
                start = raw.find("{")
                end = raw.rfind("}")
                if start != -1 and end != -1:
                    data = json.loads(raw[start : end + 1])
                else:
                    data = {}
            except Exception:
                data = {}

        power_rating = data.get("power_rating", hero_power)
        try:
            power_rating = float(power_rating)
        except Exception:
            power_rating = hero_power
        power_rating = max(0.0, min(10.0, power_rating))

        balance_issues = data.get("balance_issues", []) or []
        suggested_changes = data.get("suggested_changes", []) or []
        team_synergies = data.get("team_synergies", {}) or {}
        counter_picks = data.get("counter_picks", []) or []

        balance_issues = [str(i).strip() for i in balance_issues if str(i).strip()]
        suggested_changes = [str(c).strip() for c in suggested_changes if str(c).strip()]
        team_synergies = {
            str(k): float(v) for k, v in team_synergies.items() if str(k).strip()
        }
        counter_picks = [str(c).strip() for c in counter_picks if str(c).strip()]

        return BalanceReport(
            hero=hero,
            analysis_method="combined_zero_shot_few_shot_cot",
            power_rating=power_rating,
            balance_issues=balance_issues,
            suggested_changes=suggested_changes,
            team_synergies=team_synergies,
            counter_picks=counter_picks,
        )

    def auto_balance(self, hero: Hero, target_power: float) -> Hero:
        """
        Automatically adjust hero for target power level.
        """

        adjust_template = PromptTemplate.from_template(
            """You are auto-balancing a hero for a fighting game.

Target average power level: {target_power}/10

Current hero (JSON):
{hero_json}

Using small, clear changes:
- If too strong, slightly nerf damage, range, or frequency of strongest abilities.
- If too weak, slightly buff one or two core abilities or reduce a harsh weakness.
- Prefer to adjust numbers/limits rather than rewriting the concept.

Return ONLY valid JSON:
{{
  "abilities": ["updated ability descriptions"],
  "weaknesses": ["updated weaknesses"],
  "synergies": ["updated synergies (short notes)"]
}}"""
        )

        chain = adjust_template | self.llm | StrOutputParser()
        hero_json = json.dumps(hero.__dict__, ensure_ascii=False)

        raw = chain.invoke(
            {"target_power": target_power, "hero_json": hero_json}
        )

        try:
            data = json.loads(raw)
        except Exception:
            data = {}

        new_abilities = data.get("abilities", hero.abilities) or hero.abilities
        new_weaknesses = data.get("weaknesses", hero.weaknesses) or hero.weaknesses
        new_synergies = data.get("synergies", hero.synergies) or hero.synergies

        hero.abilities = [str(a).strip() for a in new_abilities if str(a).strip()]
        hero.weaknesses = [str(w).strip() for w in new_weaknesses if str(w).strip()]
        hero.synergies = [str(s).strip() for s in new_synergies if str(s).strip()]
        hero.power_level = target_power

        return hero


def test_balancer():
    balancer = PowerBalancer()

    test_heroes = [
        Hero(
            name="Chronos",
            abilities=["Time manipulation", "Temporal loops", "Age acceleration"],
            power_type="reality",
            power_level=0.0,
            weaknesses=[],
            synergies=[],
        ),
        Hero(
            name="Mindweaver",
            abilities=["Telepathy", "Illusion creation", "Memory manipulation"],
            power_type="mental",
            power_level=0.0,
            weaknesses=[],
            synergies=[],
        ),
        Hero(
            name="Quantum",
            abilities=["Teleportation", "Probability manipulation", "Phase shifting"],
            power_type="reality",
            power_level=0.0,
            weaknesses=[],
            synergies=[],
        ),
    ]

    print("⚡ SUPERHERO POWER BALANCER ⚡")
    print("=" * 70)

    for hero in test_heroes:
        print(f"\n🦸 Hero: {hero.name}")
        print(f"Abilities: {', '.join(hero.abilities)}")

        # Test zero-shot analysis
        analysis = balancer.analyze_hero_zero_shot(hero)
        print(f"Power Level: {analysis.get('power_level', 0):.1f}/10")

        # Test few-shot classification
        power_type = balancer.classify_power_few_shot(hero.abilities)
        print(f"Power Type: {power_type}")

        # Test CoT synergy
        if len(test_heroes) > 1:
            synergy = balancer.calculate_synergy_cot(hero, test_heroes[0])
            print(f"Synergy with {test_heroes[0].name}: {synergy:.0%}")

        print("-" * 70)

    # Test combined balance detection
    print("\n🎯 BALANCE ANALYSIS (All Techniques):")
    print("=" * 70)

    report = balancer.detect_imbalance_combined(test_heroes[0], test_heroes)

    print(f"Hero: {report.hero.name}")
    print(f"Analysis Method: {report.analysis_method}")
    print(f"Power Rating: {report.power_rating:.1f}/10")

    if report.balance_issues:
        print("Balance Issues:")
        for issue in report.balance_issues:
            print(f"  ⚠️ {issue}")

    if report.suggested_changes:
        print("Suggested Changes:")
        for change in report.suggested_changes:
            print(f"  ✓ {change}")


if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print("⚠️ Please set OPENAI_API_KEY")
    else:
        test_balancer()
