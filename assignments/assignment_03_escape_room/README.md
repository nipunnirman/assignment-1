# Assignment 3: Escape Room Puzzle Master 🔐
## Few-Shot Prompting with Pattern Recognition

### 🎯 Learning Objectives
- Master few-shot prompting with example-based learning
- Use FewShotPromptTemplate effectively
- Teach LLMs to recognize and apply patterns
- Build dynamic example selection strategies

### 📚 Scenario
You've been hired by "Enigma Escapes" to create an AI puzzle master that designs and validates escape room puzzles. The AI needs to learn from existing successful puzzles to create new ones that are challenging but solvable. By showing it examples of good puzzles and their solutions, it can generate similar brain-teasers with the right difficulty balance.

### 🔧 Your Task
Build a `PuzzleMaster` class that:
1. Learns puzzle patterns from provided examples
2. Generates new puzzles in similar styles
3. Validates puzzle solutions with step-by-step logic
4. Adjusts difficulty based on player profile
5. Creates interconnected puzzle sequences

### 📝 Requirements
- Use few-shot prompting with multiple examples
- Implement dynamic example selection
- Create various puzzle types (logic, cipher, pattern, riddle)
- Ensure generated puzzles have unique solutions
- Provide hints without giving away answers

### 🎮 Challenges
1. **Basic**: Generate simple puzzles using 2-3 examples
2. **Intermediate**: Create difficulty-adaptive puzzles with 5+ examples
3. **Advanced**: Build interconnected puzzle chains with callbacks
4. **Bonus**: Implement a puzzle validator that checks solvability

### 💡 Hints
- Quality of examples matters more than quantity
- Structure your examples consistently
- Use different example sets for different puzzle types
- Consider edge cases in your validation examples

### 🏆 Success Criteria
- Generated puzzles follow the pattern of examples
- All puzzles have logical, unique solutions
- Difficulty matches the requested level
- Hints are helpful but not obvious
- Puzzle themes are coherent and engaging

### 📊 Example Input/Output
**Input Examples**:
```
Puzzle: "I have cities, but no houses. Roads, but no cars. What am I?"
Solution: "A map"
Type: "Riddle"
Difficulty: 2

Puzzle: "Decode: 8-5-12-16"
Solution: "HELP (H=8, E=5, L=12, P=16 in alphabet)"
Type: "Cipher"
Difficulty: 3
```

**Generated Output**:
```json
{
  "puzzle": "I have branches, but no fruit, trunk, or leaves. What am I?",
  "type": "Riddle",
  "difficulty": 2,
  "hints": [
    "Think about non-living things",
    "Often found in buildings",
    "Related to money"
  ],
  "solution": "A bank",
  "solution_explanation": "Banks have branches (locations) but no natural tree parts"
}
```

### 🚀 Getting Started
1. Set up your OpenAI API key
2. Install packages: `pip install langchain-openai langchain-core`
3. Complete the TODO sections in `puzzle_master.py`
4. Test with provided puzzle examples
5. Create your own escape room scenario!

### 🔍 What You'll Learn
- Few-shot prompt engineering best practices
- Dynamic example selection strategies
- Pattern teaching and recognition with LLMs
- Balancing creativity with constraint satisfaction
