"""
Assignment 10: AI Dungeon Master
The Ultimate Challenge - Master all prompting techniques to run a D&D game

Your mission: Become the ultimate AI Dungeon Master by seamlessly combining
all prompting techniques to create epic adventures!
"""

import os
import json
import random
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_core.output_parsers import StrOutputParser


class QuestType(Enum):
    RESCUE = "rescue"
    FETCH = "fetch"
    INVESTIGATE = "investigate"
    COMBAT = "combat"
    DIPLOMACY = "diplomacy"
    EXPLORATION = "exploration"


@dataclass
class Character:
    name: str
    class_type: str
    level: int
    hit_points: int
    abilities: List[str]
    inventory: List[str]
    personality: str


@dataclass
class NPC:
    name: str
    role: str
    personality: str
    motivation: str
    dialogue_style: str
    secrets: List[str]


@dataclass
class Quest:
    title: str
    description: str
    objectives: List[str]
    rewards: List[str]
    difficulty: int
    quest_type: str


@dataclass
class CombatState:
    participants: List[Character]
    turn_order: List[str]
    environment: str
    special_conditions: List[str]


@dataclass
class WorldState:
    location: str
    time_of_day: str
    weather: str
    active_quests: List[Quest]
    npcs_present: List[NPC]
    recent_events: List[str]
    player_reputation: Dict[str, int]


class DungeonMasterAI:
    """
    AI Dungeon Master using all prompting techniques seamlessly.
    The ultimate test of prompting mastery!
    """

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(model=model_name, temperature=0.7)
        self.story_generator = None  # Zero-shot
        self.npc_manager = None  # Few-shot
        self.combat_resolver = None  # CoT
        self.world_tracker = None  # Combined
        self.world_state = WorldState(
            location="Tavern",
            time_of_day="Evening",
            weather="Clear",
            active_quests=[],
            npcs_present=[],
            recent_events=[],
            player_reputation={},
        )
        self._setup_chains()

    def _setup_chains(self):
        """
        TODO #1: Set up all chains for different DM tasks.

        Create:
        1. Zero-shot story generator for creative scenarios
        2. Few-shot NPC manager with personality examples
        3. CoT combat resolver for rule calculations
        4. Combined world tracker for state management
        """

        # ---------- Zero-shot for creative story generation ----------
        story_template = PromptTemplate.from_template(
            """You are a vivid, immersive Dungeons & Dragons Dungeon Master.

Your job:
- Describe the setting with sensory detail (sights, sounds, atmosphere).
- React to the player's action in a way that feels logical and fun.
- Present clear options or hooks, but do NOT force a single path.
- Keep it concise: 3–6 sentences.
- Write in second person: "you" (the party) perspective.

Context:
{context}

Player Action:
{action}

Now narrate the next moment in the story."""
        )
        self.story_generator = story_template | self.llm | StrOutputParser()

        # ---------- Few-shot for NPC personalities ----------
        npc_examples = [
            {
                "npc_type": "Gruff Innkeeper",
                "dialogue": "'Ale's two copper, room's a silver. No trouble or you're out.'",
                "personality": "Direct, no-nonsense, secretly kind",
                "quirk": "Always cleaning the same glass",
            },
            {
                "npc_type": "Mysterious Sage",
                "dialogue": "'The answer you seek lies not in what is seen, but what is hidden...'",
                "personality": "Cryptic, wise, slightly mad",
                "quirk": "Speaks in riddles and rhymes",
            },
            {
                "npc_type": "Overenthusiastic Bard",
                "dialogue": "'Friends! Have you heard the tragic tale of the Goblin King? No? Allow me!'",
                "personality": "Cheerful, dramatic, attention-seeking",
                "quirk": "Randomly breaks into song mid-sentence",
            },
        ]

        npc_example_prompt = PromptTemplate.from_template(
            """NPC Type: {npc_type}
Example Dialogue: {dialogue}
Personality: {personality}
Quirk: {quirk}"""
        )

        npc_prefix = """You are generating in-character dialogue for NPCs in a D&D game.

Use:
- The NPC's personality and quirks
- The player's input
- The current context

Speak ONLY as the NPC, in one or two sentences of dialogue.
Do NOT narrate actions, just spoken words.

Here are some example NPC archetypes and how they talk:
"""

        npc_suffix = """
Now the actual scene:

NPC Info:
{name} ({role})
Personality: {personality}
Motivation: {motivation}
Dialogue Style: {dialogue_style}
Secrets: {secrets}
Context: {context}
Player says: "{player_input}"

Reply as this NPC in one or two sentences of dialogue only:"""

        npc_few_shot = FewShotPromptTemplate(
            examples=npc_examples,
            example_prompt=npc_example_prompt,
            prefix=npc_prefix,
            suffix=npc_suffix,
            input_variables=[
                "name",
                "role",
                "personality",
                "motivation",
                "dialogue_style",
                "secrets",
                "context",
                "player_input",
            ],
        )

        self.npc_manager = npc_few_shot | self.llm | StrOutputParser()

        # ---------- CoT for combat calculations ----------
        combat_template = PromptTemplate.from_template(
            """You are a rules-savvy D&D-style combat resolver.

Resolve this combat action step by step using reasonable fantasy RPG logic
(you can assume standard attack roll vs AC and simple damage):

After your internal reasoning, output ONLY JSON with this structure:
{{
  "hit": true,
  "damage": 0,
  "description": "short cinematic description of what happens",
  "special_effects": ["status effects, knockback, advantage, etc"]
}}

Action:
{action}

Attacker Stats:
{stats}

Target:
{target}

Environment:
{environment}

Special Conditions:
{conditions}

Let's calculate the outcome step by step, then output ONLY the JSON:"""
        )

        self.combat_resolver = combat_template | self.llm | StrOutputParser()

        # ---------- Combined approach for world state tracking ----------
        world_template = PromptTemplate.from_template(
            """You are tracking the evolving world state of a D&D campaign.

Your goals:
- Apply logical consequences to player actions.
- Update location, time of day, and weather if appropriate.
- Append new recent events that summarize what happened.
- Adjust player reputation with factions or NPC groups if relevant.
- Keep continuity with the current state.

Return ONLY valid JSON in this structure:
{{
  "location": "string (where the party is now)",
  "time_of_day": "string (morning/afternoon/evening/night, etc.)",
  "weather": "string (brief description)",
  "recent_events": ["short event 1", "short event 2"],
  "player_reputation_changes": {{
    "faction_or_npc": 0
  }}
}}

Current State (JSON):
{current_state}

Player Actions (most recent first):
{actions}

Time Passed: {time}

Think through the consequences, then output ONLY the JSON:"""
        )

        self.world_tracker = world_template | self.llm | StrOutputParser()

    # ----------------- Core methods -----------------

    def generate_quest(self, quest_type: QuestType, party_level: int) -> Quest:
        """
        TODO #2: Generate a quest using zero-shot creativity.

        Create unique, engaging quests without examples.
        """

        quest_prompt = PromptTemplate.from_template(
            """You are an imaginative D&D quest designer.

Create a {quest_type} quest for a party of level {level}.
Keep it appropriate difficulty and flavor.

Return ONLY valid JSON:
{{
  "title": "short quest title",
  "description": "2-4 sentence quest summary",
  "objectives": ["objective 1", "objective 2"],
  "rewards": ["gold", "magic item", "favor", "information"],
  "difficulty": {level}
}}"""
        )

        chain = quest_prompt | self.llm | StrOutputParser()

        raw = chain.invoke(
            {"quest_type": quest_type.value, "level": party_level}
        )

        try:
            data = json.loads(raw)
        except Exception:
            # try to salvage JSON
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    data = json.loads(raw[start : end + 1])
                except Exception:
                    data = {}
            else:
                data = {}

        title = data.get("title", "").strip() or f"{quest_type.value.title()} Quest"
        description = data.get("description", "").strip()
        objectives = data.get("objectives", []) or []
        rewards = data.get("rewards", []) or []
        difficulty = data.get("difficulty", party_level)

        try:
            difficulty = int(difficulty)
        except Exception:
            difficulty = party_level

        return Quest(
            title=title,
            description=description,
            objectives=[str(o).strip() for o in objectives if str(o).strip()],
            rewards=[str(r).strip() for r in rewards if str(r).strip()],
            difficulty=difficulty,
            quest_type=quest_type.value,
        )

    def roleplay_npc(self, npc: NPC, player_input: str, context: Dict[str, any]) -> str:
        """
        TODO #3: Roleplay NPC using few-shot personality examples.

        Match personality patterns from examples.
        """

        context_text = f"Location: {context.get('location', 'unknown')}"
        if "situation" in context:
            context_text += f" | Situation: {context['situation']}"

        response = self.npc_manager.invoke(
            {
                "name": npc.name,
                "role": npc.role,
                "personality": npc.personality,
                "motivation": npc.motivation,
                "dialogue_style": npc.dialogue_style,
                "secrets": ", ".join(npc.secrets),
                "context": context_text,
                "player_input": player_input,
            }
        )

        return response.strip().strip('"')

    def resolve_combat(
        self,
        action: str,
        attacker: Character,
        target: Character,
        combat_state: CombatState,
    ) -> Dict[str, any]:
        """
        TODO #4: Resolve combat using CoT for rule calculations.

        Step-by-step D&D combat resolution.
        """

        stats = (
            f"Name: {attacker.name}, Class: {attacker.class_type}, "
            f"Level: {attacker.level}, HP: {attacker.hit_points}, "
            f"Abilities: {', '.join(attacker.abilities)}, "
            f"Inventory: {', '.join(attacker.inventory)}"
        )

        target_text = (
            f"Name: {target.name}, Class/Type: {target.class_type}, "
            f"Level: {target.level}, HP: {target.hit_points}, "
            f"Personality: {target.personality}"
        )

        conditions = ", ".join(combat_state.special_conditions) or "None"

        raw = self.combat_resolver.invoke(
            {
                "action": action,
                "stats": stats,
                "target": target_text,
                "environment": combat_state.environment,
                "conditions": conditions,
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

        hit = bool(data.get("hit", False))
        damage = data.get("damage", 0)
        description = data.get("description", "") or ""
        special_effects = data.get("special_effects", []) or []

        try:
            damage = int(damage)
        except Exception:
            damage = 0

        special_effects = [
            str(e).strip() for e in special_effects if str(e).strip()
        ]

        return {
            "hit": hit,
            "damage": max(0, damage),
            "description": description.strip(),
            "special_effects": special_effects,
        }

    def narrate_scene(
        self, action: str, world_state: WorldState, characters: List[Character]
    ) -> str:
        """
        TODO #5: Narrate scene using zero-shot creativity.

        Generate atmospheric, engaging descriptions.
        """

        party_desc = ", ".join(
            f"{c.name} the level {c.level} {c.class_type}"
            for c in characters
        )

        ctx = (
            f"Location: {world_state.location}, Time: {world_state.time_of_day}, "
            f"Weather: {world_state.weather}. Party: {party_desc}. "
        )
        if world_state.recent_events:
            ctx += f"Recent events: {', '.join(world_state.recent_events[-3:])}."

        narration = self.story_generator.invoke(
            {"context": ctx, "action": action}
        )

        return narration.strip()

    def update_world(self, actions: List[str], time_passed: str) -> WorldState:
        """
        TODO #6: Update world state using ALL techniques.

        Orchestrate all methods for comprehensive world management.
        """

        current_state_json = json.dumps(
            asdict(self.world_state), ensure_ascii=False
        )
        actions_text = "\n".join(actions)

        raw = self.world_tracker.invoke(
            {
                "current_state": current_state_json,
                "actions": actions_text,
                "time": time_passed,
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

        loc = data.get("location")
        tod = data.get("time_of_day")
        weather = data.get("weather")
        recents = data.get("recent_events", []) or []
        rep_changes = data.get("player_reputation_changes", {}) or {}

        if isinstance(loc, str) and loc.strip():
            self.world_state.location = loc.strip()
        if isinstance(tod, str) and tod.strip():
            self.world_state.time_of_day = tod.strip()
        if isinstance(weather, str) and weather.strip():
            self.world_state.weather = weather.strip()

        for ev in recents:
            ev_str = str(ev).strip()
            if ev_str:
                self.world_state.recent_events.append(ev_str)

        for faction, delta in rep_changes.items():
            try:
                delta_val = int(delta)
            except Exception:
                delta_val = 0
            if faction not in self.world_state.player_reputation:
                self.world_state.player_reputation[faction] = 0
            self.world_state.player_reputation[faction] += delta_val

        return self.world_state

    def run_session(
        self, player_actions: List[str], party: List[Character]
    ) -> Dict[str, any]:
        """
        TODO #7: Run a complete game session using all techniques.

        The ultimate test - seamlessly combine everything!
        """

        session_log = {
            "narration": [],
            "npc_interactions": [],
            "combat_results": [],
            "quest_updates": [],
            "world_changes": [],
        }

        # Make sure some NPCs exist in the world for interactions
        if not self.world_state.npcs_present:
            self.world_state.npcs_present = [
                NPC(
                    name="Gareth",
                    role="Tavern Keeper",
                    personality="Gruff but kind",
                    motivation="Keep tavern safe",
                    dialogue_style="Direct and practical",
                    secrets=["Former adventurer", "Knows local rumors"],
                )
            ]

        for action in player_actions:
            # 1. Narration
            narration = self.narrate_scene(action, self.world_state, party)
            session_log["narration"].append(narration)

            # 2. Possible NPC interaction
            if any(word in action.lower() for word in ["talk", "ask", "speak", "question"]):
                npc = self.world_state.npcs_present[0]
                npc_reply = self.roleplay_npc(
                    npc, action, {"location": self.world_state.location, "situation": "player initiated conversation"}
                )
                session_log["npc_interactions"].append(
                    {"npc": npc.name, "player_action": action, "reply": npc_reply}
                )

            # 3. Simple combat hook if action is aggressive
            if any(word in action.lower() for word in ["attack", "strike", "fight", "ambush"]):
                attacker = party[0]
                enemy = Character(
                    name="Bandit",
                    class_type="Rogue",
                    level=attacker.level,
                    hit_points=12,
                    abilities=["Sneak Attack"],
                    inventory=["Rusty Sword"],
                    personality="Desperate and hostile",
                )
                combat_state = CombatState(
                    participants=[attacker, enemy],
                    turn_order=[attacker.name, enemy.name],
                    environment=self.world_state.location,
                    special_conditions=[],
                )
                result = self.resolve_combat(
                    action, attacker, enemy, combat_state
                )
                session_log["combat_results"].append(result)

            # 4. Update world after each action
            old_location = self.world_state.location
            old_time = self.world_state.time_of_day
            self.update_world([action], "a few minutes")

            change_summary = f"Location: {old_location} -> {self.world_state.location}, Time: {old_time} -> {self.world_state.time_of_day}"
            session_log["world_changes"].append(change_summary)

        return session_log


def test_dungeon_master():
    """Test the AI Dungeon Master with a mini adventure."""

    dm = DungeonMasterAI()

    # Create test party
    test_party = [
        Character(
            name="Aldric",
            class_type="Fighter",
            level=3,
            hit_points=28,
            abilities=["Second Wind", "Action Surge"],
            inventory=["Longsword", "Shield", "Healing Potion"],
            personality="Brave but reckless",
        ),
        Character(
            name="Lyra",
            class_type="Wizard",
            level=3,
            hit_points=18,
            abilities=["Fireball", "Shield", "Detect Magic"],
            inventory=["Spellbook", "Crystal Orb", "Scrolls"],
            personality="Cautious and analytical",
        ),
    ]

    # Create test NPCs
    test_npcs = [
        NPC(
            name="Gareth",
            role="Tavern Keeper",
            personality="Gruff but kind",
            motivation="Keep tavern safe",
            dialogue_style="Direct and practical",
            secrets=["Former adventurer", "Has a treasure map"],
        ),
        NPC(
            name="Lady Morwyn",
            role="Noble Patron",
            personality="Aristocratic and mysterious",
            motivation="Find ancient artifact",
            dialogue_style="Formal and cryptic",
            secrets=["Is actually a dragon", "Knows about the prophecy"],
        ),
    ]

    # put NPCs into world state for later
    dm.world_state.npcs_present = test_npcs

    print("🎲 AI DUNGEON MASTER 🎲")
    print("=" * 70)
    print("Welcome to the Realm of Aethermoor!")
    print("-" * 70)

    # Test quest generation (Zero-shot)
    print("\n📜 QUEST GENERATION (Zero-shot):")
    quest = dm.generate_quest(QuestType.RESCUE, party_level=3)
    print(f"Quest: {quest.title}")
    print(f"Description: {quest.description}")
    print(
        f"Objectives: {', '.join(quest.objectives[:2]) if quest.objectives else 'None'}"
    )

    # Test NPC roleplay (Few-shot)
    print("\n🗣️ NPC INTERACTION (Few-shot):")
    player_input = "We're looking for adventure and gold!"
    for npc in test_npcs[:1]:
        response = dm.roleplay_npc(npc, player_input, {"location": "tavern"})
        print(f'{npc.name}: "{response}"')

    # Test combat resolution (CoT)
    print("\n⚔️ COMBAT RESOLUTION (Chain of Thought):")
    combat = CombatState(
        participants=test_party,
        turn_order=[p.name for p in test_party],
        environment="Dark forest clearing",
        special_conditions=["Fog - disadvantage on ranged attacks"],
    )

    combat_result = dm.resolve_combat(
        "Aldric attacks the goblin with his longsword",
        test_party[0],
        Character("Goblin", "Monster", 1, 7, ["Sneak"], ["Dagger"], "Cowardly"),
        combat,
    )

    print(f"Action: Aldric attacks")
    print(f"Result: {'Hit!' if combat_result.get('hit') else 'Miss!'}")
    print(f"Damage: {combat_result.get('damage', 0)}")

    # Test scene narration (Zero-shot)
    print("\n🎭 SCENE NARRATION (Zero-shot):")
    narration = dm.narrate_scene(
        "The party enters the ancient ruins", dm.world_state, test_party
    )
    print(
        f"DM: {narration[:200]}..." if narration else "DM: [Scene description pending]"
    )

    # Test world state update (All techniques)
    print("\n🌍 WORLD STATE UPDATE (All Techniques):")
    player_actions = [
        "Defeated the goblin raiders",
        "Rescued the merchant",
        "Found mysterious artifact",
    ]

    updated_state = dm.update_world(player_actions, "2 hours")
    print(f"Location: {updated_state.location}")
    print(f"Time: {updated_state.time_of_day}")
    print(f"Recent Events: {len(updated_state.recent_events)} recorded")

    # Run mini session
    print("\n🎮 MINI SESSION (All Techniques Combined):")
    print("=" * 70)

    session_actions = [
        "We investigate the strange noises from the cellar",
        "I cast Detect Magic on the mysterious door",
        "We try to talk to the creature in the shadows",
        "I attack the nearest bandit",
    ]

    session = dm.run_session(session_actions, test_party)

    if session.get("narration"):
        print("Session Highlights:")
        for event in session["narration"][:3]:
            print(f"  • {event}")

    print("\n🏆 Adventure Continues...")
    print("The AI Dungeon Master awaits your next move!")


if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print("⚠️ Please set OPENAI_API_KEY environment variable")
        print("   export OPENAI_API_KEY='your-api-key-here'")
    else:
        test_dungeon_master()
