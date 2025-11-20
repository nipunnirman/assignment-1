# Assignment 4: Time Travel Paradox Resolver ⏰
## Chain of Thought Reasoning for Complex Logic

### 🎯 Learning Objectives
- Master Chain of Thought (CoT) prompting techniques
- Implement step-by-step reasoning for complex problems
- Compare zero-shot CoT vs few-shot CoT approaches
- Build Auto-CoT for automatic reasoning generation

### 📚 Scenario
The Temporal Research Institute has discovered time travel but faces a crisis: every mission creates potential paradoxes. You've been tasked with building an AI system that analyzes time travel scenarios, identifies paradoxes, and proposes solutions using step-by-step logical reasoning. The AI must think through causality chains, timeline branches, and butterfly effects.

### 🔧 Your Task
Build a `ParadoxResolver` class that:
1. Analyzes time travel scenarios for logical paradoxes
2. Uses step-by-step reasoning to trace causality chains
3. Calculates timeline stability scores
4. Proposes paradox resolution strategies
5. Predicts butterfly effects and unintended consequences

### 📝 Requirements
- Implement both zero-shot CoT ("Let's think step by step")
- Create few-shot CoT with reasoning examples
- Build Auto-CoT that generates its own reasoning examples
- Handle multiple types of paradoxes (grandfather, bootstrap, etc.)
- Provide clear reasoning chains for all conclusions

### 🎮 Challenges
1. **Basic**: Implement zero-shot CoT for simple paradox detection
2. **Intermediate**: Use few-shot CoT with complex reasoning chains
3. **Advanced**: Build Auto-CoT that generates reasoning examples
4. **Bonus**: Create a timeline visualizer showing causality flows

### 💡 Hints
- Break complex problems into smaller reasoning steps
- Make intermediate conclusions explicit
- Use "Let's think step by step" for zero-shot CoT
- Quality of reasoning examples matters in few-shot CoT
- Auto-CoT can generate diverse reasoning patterns

### 🏆 Success Criteria
- Reasoning chains are logical and complete
- Paradoxes are correctly identified and classified
- Solutions follow from the reasoning steps
- Timeline stability calculations are consistent
- Butterfly effects are traced accurately

### 📊 Example Input/Output
**Input Scenario**:
```
"A scientist travels back 30 years and accidentally prevents their parents from meeting at a coffee shop. They have 24 hours before the timeline solidifies."
```

**Chain of Thought Output**:
```json
{
  "paradox_type": "Grandfather Paradox Variant",
  "reasoning_steps": [
    "Step 1: Identify the causal chain - parents meeting led to scientist's birth",
    "Step 2: Analyze the intervention - preventing meeting breaks the causal chain",
    "Step 3: Determine paradox - if parents don't meet, scientist isn't born",
    "Step 4: But if scientist isn't born, they can't travel back to prevent meeting",
    "Step 5: This creates a logical contradiction - classic grandfather paradox"
  ],
  "timeline_stability": 0.15,
  "resolution_strategies": [
    "Arrange alternative meeting within 24 hours",
    "Create conditions for delayed meeting",
    "Accept timeline branch (multiverse theory)"
  ],
  "butterfly_effects": [
    "Different meeting location may alter relationship dynamics",
    "Delayed meeting could affect birth timing",
    "Knowledge of future may influence parent decisions"
  ]
}
```

### 🚀 Getting Started
1. Set up your OpenAI API key
2. Install packages: `pip install langchain-openai langchain-core`
3. Complete the TODO sections in `paradox_resolver.py`
4. Test with provided time travel scenarios
5. Create your own paradoxical situations!

### 🔍 What You'll Learn
- Zero-shot CoT prompting ("Let's think step by step")
- Few-shot CoT with reasoning examples
- Auto-CoT for automatic reasoning generation
- Complex multi-step logical reasoning
- Handling causality and temporal logic
