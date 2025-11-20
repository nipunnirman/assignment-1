# Assignment 1: Dream Journal Analyzer 🌙
## Zero-Shot Prompting Fundamentals

### 🎯 Learning Objectives
- Master zero-shot prompting with clear instructions
- Parse unstructured text into structured outputs
- Handle ambiguous and creative content
- Create JSON-structured responses without examples

### 📚 Scenario
You've been hired by a sleep research lab to build an AI system that analyzes dream journals. Scientists want to understand patterns in dreams without manually reading thousands of entries. Your system needs to extract themes, emotions, symbols, and potential meanings from dream descriptions using only clear instructions - no training data available!

### 🔧 Your Task
Build a `DreamAnalyzer` class that:
1. Extracts key dream symbols and their potential meanings
2. Identifies emotional tones throughout the dream
3. Detects recurring patterns or themes
4. Generates a "lucidity score" (how aware the dreamer was)
5. Suggests potential psychological insights

### 📝 Requirements
- Use ONLY zero-shot prompting (no examples in prompts)
- Return structured JSON outputs
- Handle various dream description styles
- Create clear, specific instructions for the LLM

### 🎮 Challenges
1. **Basic**: Extract symbols and emotions from dreams
2. **Intermediate**: Calculate lucidity scores and detect patterns
3. **Advanced**: Generate psychological insights and dream interpretations
4. **Bonus**: Create a dream similarity checker to find related dreams

### 💡 Hints
- Be very specific about output formats in your prompts
- Use enums and dataclasses for structured data
- Consider edge cases like nightmares, lucid dreams, and fragmented memories
- Temperature settings matter for creative vs. analytical tasks

### 🏆 Success Criteria
- Your analyzer correctly identifies 80%+ of major dream symbols
- Emotional analysis matches human interpretation
- Output is consistently structured JSON
- System handles both detailed and vague dream descriptions

### 📊 Example Input/Output
**Input**: "I was flying over a purple ocean, but then realized I was actually in my childhood bedroom. A talking cat told me I was late for an exam I hadn't studied for."

**Expected Output Structure**:
```json
{
  "symbols": [
    {"symbol": "flying", "meaning": "freedom or escape"},
    {"symbol": "purple ocean", "meaning": "mystery or emotional depth"},
    {"symbol": "childhood bedroom", "meaning": "past or comfort"},
    {"symbol": "talking cat", "meaning": "intuition or independence"},
    {"symbol": "exam", "meaning": "self-evaluation or anxiety"}
  ],
  "emotions": ["wonder", "confusion", "anxiety"],
  "themes": ["transformation", "regression", "performance anxiety"],
  "lucidity_score": 6.5,
  "insights": "Dream suggests conflict between desire for freedom and past obligations..."
}
```

### 🚀 Getting Started
1. Set up your OpenAI API key as environment variable
2. Install required packages: `pip install langchain-openai langchain-core`
3. Complete the TODO sections in `dream_analyzer.py`
4. Test with the provided dream samples
5. Try your own dream descriptions!

### 🔍 What You'll Learn
- How to craft effective zero-shot prompts
- Importance of clear instructions and constraints
- Structured output parsing techniques
- Handling subjective and creative content with LLMs
