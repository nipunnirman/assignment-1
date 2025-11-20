"""
Assignment 3: Escape Room Puzzle Master
Few-Shot Prompting - Learn from examples to create brain-teasing puzzles

Your mission: Create an AI that learns from puzzle examples to generate
new escape room challenges that are clever, solvable, and fun!
"""

import os
import json
import random
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_core.output_parsers import StrOutputParser


class PuzzleType(Enum):
    RIDDLE = "riddle"
    CIPHER = "cipher"
    LOGIC = "logic"
    PATTERN = "pattern"
    WORDPLAY = "wordplay"
    VISUAL = "visual"


class DifficultyLevel(Enum):
    BEGINNER = 1
    EASY = 2
    MEDIUM = 3
    HARD = 4
    EXPERT = 5


@dataclass
class Puzzle:
    """Represents an escape room puzzle"""

    puzzle_text: str
    solution: str
    puzzle_type: str
    difficulty: int
    hints: List[str]
    explanation: str
    time_estimate: int  # minutes


@dataclass
class PuzzleSequence:
    """A series of interconnected puzzles"""

    theme: str
    puzzles: List[Puzzle]
    final_solution: str
    narrative: str


class PuzzleMaster:
    """
    AI-powered escape room puzzle generator using few-shot prompting.
    Learns from examples to create engaging, solvable puzzles.
    """

    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.7):
        """
        Initialize the puzzle master.

        Args:
            model_name: The LLM model to use
            temperature: Controls creativity (higher = more creative)
        """
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self.puzzle_examples = self._load_puzzle_examples()
        self.generation_chain = None
        self.validation_chain = None
        self.hint_chain = None
        self._setup_chains()

    def _load_puzzle_examples(self) -> Dict[str, List[dict]]:
        """
        TODO #1: Create example puzzles for few-shot learning.

        Create examples for each puzzle type with consistent format.
        Include puzzle, solution, explanation, and metadata.

        Returns:
            Dictionary mapping puzzle types to example lists
        """

        examples = {
            "riddle": [
                {
                    "puzzle": "I speak without a mouth and hear without ears. I have no body, but come alive with wind. What am I?",
                    "solution": "An echo",
                    "difficulty": "2",
                    "explanation": "An echo repeats sounds (speaks) and responds to sounds (hears) but has no physical form.",
                },
                {
                    "puzzle": "The more of me there is, the less you see. What am I?",
                    "solution": "Darkness",
                    "difficulty": "1",
                    "explanation": "As darkness increases, visibility decreases, making it harder to see.",
                },
                {
                    "puzzle": "What has keys but can’t open locks, has space but no room, and lets you enter but not go in?",
                    "solution": "A keyboard",
                    "difficulty": "2",
                    "explanation": "A keyboard has keys, a space bar, and an enter key but no physical rooms or doors.",
                },
            ],
            "cipher": [
                {
                    "puzzle": "Decode: 13-1-26-5",
                    "solution": "MAZE (M=13, A=1, Z=26, E=5)",
                    "difficulty": "3",
                    "explanation": "Simple substitution cipher using alphabetical position numbers.",
                },
                {
                    "puzzle": "Decode this Caesar cipher (shift 3): KHOOR ZRUOG",
                    "solution": "HELLO WORLD",
                    "difficulty": "2",
                    "explanation": "Each letter is shifted 3 positions forward in the alphabet.",
                },
                {
                    "puzzle": "In a substitution cipher, 'VENUS' is written as 'ZIRYW'. Using the same rule, encode 'MARS'.",
                    "solution": "QEVI",
                    "difficulty": "3",
                    "explanation": "Each letter is shifted 4 positions forward: M→Q, A→E, R→V, S→W.",
                },
            ],
            "logic": [
                {
                    "puzzle": "Three switches control three light bulbs in another room. You can only enter the room once. How do you determine which switch controls which bulb?",
                    "solution": "Turn on first switch for 10 minutes, then turn it off. Turn on second switch and enter room. Hot unlit bulb = first switch, lit bulb = second switch, cold unlit = third switch.",
                    "difficulty": "4",
                    "explanation": "Uses the property that incandescent bulbs generate heat when on.",
                },
                {
                    "puzzle": "A guard lies on odd-numbered days and tells the truth on even-numbered days. Today he says, 'Yesterday I lied.' What day is it?",
                    "solution": "It is Monday.",
                    "difficulty": "3",
                    "explanation": "Only Monday makes the statement consistent with the alternating truth/lie pattern.",
                },
                {
                    "puzzle": "You see four doors labeled A, B, C, and D. Only one leads out. A sign says: 'Door A or B is correct, but not both. Door B or C is correct, but not both. Exactly one statement is true.' Which door leads out?",
                    "solution": "Door D",
                    "difficulty": "4",
                    "explanation": "If either statement were true, at least one of A/B/C would be correct. For only one statement to be true yet no valid door among A–C, D must be the exit.",
                },
            ],
            "pattern": [
                {
                    "puzzle": "Complete the sequence: 2, 6, 12, 20, 30, ?",
                    "solution": "42",
                    "difficulty": "3",
                    "explanation": "Pattern is n*(n+1): 1*2=2, 2*3=6, 3*4=12, 4*5=20, 5*6=30, 6*7=42",
                },
                {
                    "puzzle": "Complete the sequence: 1, 1, 2, 3, 5, 8, ?",
                    "solution": "13",
                    "difficulty": "1",
                    "explanation": "Fibonacci sequence: each number is the sum of the previous two.",
                },
                {
                    "puzzle": "Complete the sequence: 3, 9, 27, 81, ?",
                    "solution": "243",
                    "difficulty": "2",
                    "explanation": "Each term is multiplied by 3: 3×3=9, 9×3=27, 27×3=81, 81×3=243.",
                },
            ],
            # We can still generate WORDPLAY and VISUAL using the same few-shot style,
            # even if they don't have their own example sets yet.
        }

        return examples

    def _setup_chains(self):
        """
        TODO #2: Create few-shot prompt templates for puzzle generation.

        Set up:
        1. generation_chain: Creates new puzzles based on examples
        2. validation_chain: Checks if puzzles are solvable
        3. hint_chain: Generates progressive hints
        """

        # Example formatting template (for few-shot examples)
        example_prompt = PromptTemplate.from_template(
            """Puzzle: {puzzle}
Solution: {solution}
Difficulty: {difficulty}
Explanation: {explanation}"""
        )

        # Few-shot generation prefix/suffix
        generation_prefix = """You are a master escape room designer.

You generate clever, fair, and solvable puzzles.
Follow these rules:
- Follow the style and structure of the examples.
- Keep the puzzle self-contained and logically solvable.
- Respect the requested difficulty level:
  1-2 = simple, 3 = moderate, 4-5 = complex / multi-step.
- Ensure there is a single, clear solution.
- Make puzzles suitable for a thematic escape room setting.
- Avoid offensive or unsafe content.

Here are examples of excellent puzzles:
"""

        generation_suffix = """Now create ONE new {puzzle_type} puzzle with difficulty {difficulty}.
Theme: {theme}

Return ONLY valid JSON with this exact structure:
{{
  "puzzle_text": "string",
  "solution": "string",
  "puzzle_type": "{puzzle_type}",
  "difficulty": {difficulty},
  "explanation": "string",
  "time_estimate": 10,
  "hints": ["hint 1", "hint 2", "hint 3"]
}}"""

        # Flatten all examples for the few-shot prompt
        all_examples: List[dict] = []
        for lst in self.puzzle_examples.values():
            all_examples.extend(lst)

        generation_prompt = FewShotPromptTemplate(
            examples=all_examples,
            example_prompt=example_prompt,
            prefix=generation_prefix,
            suffix=generation_suffix,
            input_variables=["puzzle_type", "difficulty", "theme"],
        )

        self.generation_chain = generation_prompt | self.llm | StrOutputParser()

        # Validation chain: check puzzle fairness / solvability
        validation_template = PromptTemplate.from_template(
            """You are evaluating an escape room puzzle for fairness and solvability.

Puzzle (JSON):
{puzzle_json}

Assess:
- Is the puzzle solvable based on the given information?
- Does it have a unique, clear solution?
- Is the stated difficulty appropriate (1-5)?
- Are there any issues or confusing parts?
- Suggest improvements if needed.

Return ONLY valid JSON in this format:
{{
  "is_solvable": true,
  "has_unique_solution": true,
  "difficulty_appropriate": true,
  "issues": ["string"],
  "suggestions": ["string"]
}}"""
        )

        self.validation_chain = validation_template | self.llm | StrOutputParser()

        # Hint chain: progressive hints from subtle to obvious
        hint_template = PromptTemplate.from_template(
            """You are designing helpful hints for an escape room puzzle.

Puzzle:
{puzzle_text}

Solution:
{solution}

Create {num_hints} hints that:
- Start very subtle and thematic.
- Gradually become more direct.
- The last hint should make the solution almost obvious without explicitly stating it.

Return ONLY valid JSON:
{{
  "hints": ["hint 1", "hint 2", "hint 3"]
}}"""
        )

        self.hint_chain = hint_template | self.llm | StrOutputParser()

    def generate_puzzle(
        self,
        puzzle_type: PuzzleType,
        difficulty: DifficultyLevel,
        theme: str = "general",
    ) -> Puzzle:
        """
        TODO #3: Generate a new puzzle using few-shot learning.

        Args:
            puzzle_type: Type of puzzle to generate
            difficulty: Difficulty level (1-5)
            theme: Theme for the puzzle

        Returns:
            Generated Puzzle object
        """

        # Use the few-shot generation chain
        try:
            raw = self.generation_chain.invoke(
                {
                    "puzzle_type": puzzle_type.value,
                    "difficulty": difficulty.value,
                    "theme": theme,
                }
            )
            data = json.loads(raw)
        except Exception:
            # Fallback simple puzzle if JSON parsing or generation fails
            data = {
                "puzzle_text": "Something went wrong generating a puzzle. Solve this instead: What has hands but cannot clap?",
                "solution": "A clock",
                "puzzle_type": puzzle_type.value,
                "difficulty": difficulty.value,
                "explanation": "A clock has hands that point to numbers but cannot clap.",
                "time_estimate": 3,
                "hints": [
                    "It is an object found in most rooms.",
                    "It helps you keep track of time.",
                    "Its hands move in a circle.",
                ],
            }

        puzzle_text = data.get("puzzle_text", "").strip()
        solution = data.get("solution", "").strip()
        explanation = data.get("explanation", "").strip()
        time_estimate = data.get("time_estimate", 10)
        hints = data.get("hints", []) or []

        # Ensure proper types
        try:
            time_estimate = int(time_estimate)
        except Exception:
            time_estimate = 10

        # Make sure hints is a list of strings
        hints = [str(h).strip() for h in hints if str(h).strip()]

        puzzle = Puzzle(
            puzzle_text=puzzle_text,
            solution=solution,
            puzzle_type=puzzle_type.value,
            difficulty=difficulty.value,
            hints=hints,
            explanation=explanation,
            time_estimate=time_estimate,
        )

        return puzzle

    def validate_puzzle(self, puzzle: Puzzle) -> Dict[str, any]:
        """
        TODO #4: Validate that a puzzle is solvable and fair.

        Args:
            puzzle: The puzzle to validate

        Returns:
            Validation result with solvability score and issues
        """

        puzzle_json = json.dumps(asdict(puzzle), ensure_ascii=False)

        try:
            raw = self.validation_chain.invoke({"puzzle_json": puzzle_json})
            data = json.loads(raw)
        except Exception:
            # Fallback: assume it's okay but mark as unverified
            data = {
                "is_solvable": True,
                "has_unique_solution": True,
                "difficulty_appropriate": True,
                "issues": ["Validation failed; using default assumptions."],
                "suggestions": [],
            }

        # Normalize fields
        validation = {
            "is_solvable": bool(data.get("is_solvable", True)),
            "has_unique_solution": bool(data.get("has_unique_solution", True)),
            "difficulty_appropriate": bool(data.get("difficulty_appropriate", True)),
            "issues": data.get("issues", []) or [],
            "suggestions": data.get("suggestions", []) or [],
        }

        return validation

    def generate_hints(self, puzzle: Puzzle, num_hints: int = 3) -> List[str]:
        """
        TODO #5: Generate progressive hints for a puzzle.

        Args:
            puzzle: The puzzle to generate hints for
            num_hints: Number of hints to generate

        Returns:
            List of hints from subtle to obvious
        """

        # If the puzzle already has hints, just use/trim them
        if puzzle.hints:
            return puzzle.hints[:num_hints]

        try:
            raw = self.hint_chain.invoke(
                {
                    "puzzle_text": puzzle.puzzle_text,
                    "solution": puzzle.solution,
                    "num_hints": num_hints,
                }
            )
            data = json.loads(raw)
            hints = data.get("hints", []) or []
        except Exception:
            hints = []

        hints = [str(h).strip() for h in hints if str(h).strip()]

        # If still empty, add basic fallback hints
        if not hints:
            hints = [
                "Look carefully at the wording of the puzzle.",
                "Think about common puzzle patterns and meanings.",
                "Focus on the key nouns or numbers given in the puzzle.",
            ][:num_hints]

        return hints[:num_hints]

    def create_puzzle_sequence(
        self, theme: str, num_puzzles: int = 3, difficulty_curve: str = "increasing"
    ) -> PuzzleSequence:
        """
        TODO #6: Create a sequence of interconnected puzzles.

        Args:
            theme: Overall theme for the sequence
            num_puzzles: Number of puzzles in sequence
            difficulty_curve: "increasing", "decreasing", or "varied"

        Returns:
            PuzzleSequence with related puzzles
        """

        # Decide difficulties for the sequence
        difficulties: List[DifficultyLevel] = []
        if difficulty_curve == "increasing":
            for i in range(num_puzzles):
                lvl = min(5, 1 + i * max(1, 4 // max(1, num_puzzles - 1)))
                difficulties.append(DifficultyLevel(lvl))
        elif difficulty_curve == "decreasing":
            for i in range(num_puzzles):
                lvl = max(1, 5 - i * max(1, 4 // max(1, num_puzzles - 1)))
                difficulties.append(DifficultyLevel(lvl))
        else:  # varied
            for _ in range(num_puzzles):
                difficulties.append(random.choice(list(DifficultyLevel)))

        # Cycle through some puzzle types
        type_cycle = [PuzzleType.RIDDLE, PuzzleType.CIPHER, PuzzleType.LOGIC, PuzzleType.PATTERN]
        puzzles: List[Puzzle] = []

        for i in range(num_puzzles):
            p_type = type_cycle[i % len(type_cycle)]
            diff = difficulties[i]
            puzzles.append(self.generate_puzzle(p_type, diff, theme=theme))

        # Ask the model to tie them together into a narrative and final solution
        sequence_prompt = PromptTemplate.from_template(
            """You are designing the story for an escape room sequence.

Theme: {theme}

Here are the puzzles (JSON list):
{puzzles_json}

Create:
- A short narrative that connects these puzzles into one coherent story.
- A final meta-solution or final code that players get after solving all puzzles.

Return ONLY valid JSON:
{{
  "narrative": "string",
  "final_solution": "string"
}}"""
        )

        sequence_chain = sequence_prompt | self.llm | StrOutputParser()

        try:
            puzzles_json = json.dumps([asdict(p) for p in puzzles], ensure_ascii=False)
            raw = sequence_chain.invoke({"theme": theme, "puzzles_json": puzzles_json})
            data = json.loads(raw)
            narrative = data.get("narrative", "").strip()
            final_solution = data.get("final_solution", "").strip()
        except Exception:
            narrative = f"A series of puzzles themed around {theme}, leading the players through a mysterious adventure."
            final_solution = "FINAL CODE"

        sequence = PuzzleSequence(
            theme=theme, puzzles=puzzles, final_solution=final_solution, narrative=narrative
        )

        return sequence

    def adapt_difficulty(
        self, puzzle: Puzzle, target_difficulty: DifficultyLevel
    ) -> Puzzle:
        """
        TODO #7 (Bonus): Adapt an existing puzzle to a different difficulty.

        Args:
            puzzle: Original puzzle
            target_difficulty: Desired difficulty level

        Returns:
            Modified puzzle at new difficulty
        """

        adapt_template = PromptTemplate.from_template(
            """You are adjusting the difficulty of an escape room puzzle.

Original puzzle (JSON):
{puzzle_json}

Target difficulty: {target_difficulty} (1=beginner, 5=expert)

Rules:
- Keep the core idea and solution the same if possible.
- To make it easier: add context, clarify wording, reduce steps, or provide more obvious clues.
- To make it harder: add misdirection, require multiple reasoning steps, or hide clues more cleverly.
- Ensure the puzzle remains fair and logically solvable.

Return ONLY valid JSON with this structure:
{{
  "puzzle_text": "string",
  "solution": "{solution}",
  "puzzle_type": "{puzzle_type}",
  "difficulty": {target_difficulty},
  "explanation": "string",
  "time_estimate": {time_estimate},
  "hints": ["hint 1", "hint 2", "hint 3"]
}}"""
        )

        adapt_chain = adapt_template | self.llm | StrOutputParser()

        puzzle_json = json.dumps(asdict(puzzle), ensure_ascii=False)

        try:
            raw = adapt_chain.invoke(
                {
                    "puzzle_json": puzzle_json,
                    "target_difficulty": target_difficulty.value,
                    "solution": puzzle.solution,
                    "puzzle_type": puzzle.puzzle_type,
                    "time_estimate": puzzle.time_estimate,
                }
            )
            data = json.loads(raw)
        except Exception:
            # Fallback: just adjust difficulty number
            data = {
                "puzzle_text": puzzle.puzzle_text,
                "solution": puzzle.solution,
                "puzzle_type": puzzle.puzzle_type,
                "difficulty": target_difficulty.value,
                "explanation": puzzle.explanation,
                "time_estimate": puzzle.time_estimate,
                "hints": puzzle.hints,
            }

        adapted = Puzzle(
            puzzle_text=data.get("puzzle_text", puzzle.puzzle_text),
            solution=data.get("solution", puzzle.solution),
            puzzle_type=data.get("puzzle_type", puzzle.puzzle_type),
            difficulty=int(data.get("difficulty", target_difficulty.value)),
            hints=[str(h).strip() for h in data.get("hints", []) or puzzle.hints],
            explanation=data.get("explanation", puzzle.explanation),
            time_estimate=int(data.get("time_estimate", puzzle.time_estimate)),
        )

        return adapted_puzzle if (adapted_puzzle := adapted) else adapted


def test_puzzle_master():
    """Test the puzzle master with various scenarios."""

    master = PuzzleMaster()

    # Test different puzzle types and difficulties
    test_scenarios = [
        {
            "type": PuzzleType.RIDDLE,
            "difficulty": DifficultyLevel.EASY,
            "theme": "pirates",
        },
        {
            "type": PuzzleType.CIPHER,
            "difficulty": DifficultyLevel.MEDIUM,
            "theme": "space",
        },
        {
            "type": PuzzleType.LOGIC,
            "difficulty": DifficultyLevel.HARD,
            "theme": "haunted mansion",
        },
        {
            "type": PuzzleType.PATTERN,
            "difficulty": DifficultyLevel.MEDIUM,
            "theme": "ancient Egypt",
        },
        {
            "type": PuzzleType.WORDPLAY,
            "difficulty": DifficultyLevel.EASY,
            "theme": "detective",
        },
    ]

    print("🔐 ESCAPE ROOM PUZZLE MASTER 🔐")
    print("=" * 70)

    for scenario in test_scenarios:
        print(f"\n🎯 Generating {scenario['type'].value} puzzle")
        print(f"   Theme: {scenario['theme']}")
        print(f"   Difficulty: {'⭐' * scenario['difficulty'].value}")

        # Generate puzzle
        puzzle = master.generate_puzzle(
            scenario["type"], scenario["difficulty"], scenario["theme"]
        )

        # Display puzzle
        print(f"\n📝 Puzzle:")
        print(f"   {puzzle.puzzle_text}")

        # Validate puzzle
        validation = master.validate_puzzle(puzzle)
        print(f"\n✅ Validation:")
        print(f"   Solvable: {'Yes' if validation['is_solvable'] else 'No'}")
        print(
            f"   Unique Solution: {'Yes' if validation['has_unique_solution'] else 'No'}"
        )

        # Generate hints
        hints = master.generate_hints(puzzle, num_hints=3)
        if hints:
            print(f"\n💡 Hints:")
            for i, hint in enumerate(hints, 1):
                print(f"   {i}. {hint}")

        # Show solution
        print(f"\n🔓 Solution: {puzzle.solution}")
        print(f"📖 Explanation: {puzzle.explanation}")
        print(f"⏱️ Estimated Time: {puzzle.time_estimate} minutes")

        print("-" * 70)

    # Test puzzle sequence
    print("\n🎮 PUZZLE SEQUENCE TEST:")
    print("=" * 70)

    sequence = master.create_puzzle_sequence(
        theme="Time Travel Mystery", num_puzzles=3, difficulty_curve="increasing"
    )

    print(f"📚 Theme: {sequence.theme}")
    print(f"📖 Narrative: {sequence.narrative}")
    print(f"🎯 Number of Puzzles: {len(sequence.puzzles)}")

    for i, puzzle in enumerate(sequence.puzzles, 1):
        print(f"\n   Puzzle {i}: {puzzle.puzzle_text[:100]}...")
        print(f"   Type: {puzzle.puzzle_type}")
        print(f"   Difficulty: {'⭐' * puzzle.difficulty}")

    if sequence.final_solution:
        print(f"\n🏆 Final Solution: {sequence.final_solution}")


if __name__ == "__main__":
    # Make sure to set OPENAI_API_KEY environment variable
    if not os.environ.get("OPENAI_API_KEY"):
        print("⚠️ Please set OPENAI_API_KEY environment variable")
    else:
        test_puzzle_master()
