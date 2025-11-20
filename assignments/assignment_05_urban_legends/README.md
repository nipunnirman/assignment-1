# Assignment 5: Urban Legend Fact Checker 🕵️
## Combining Zero-shot and Few-shot Prompting

### 🎯 Learning Objectives
- Combine zero-shot and few-shot prompting techniques
- Choose the right approach for different analysis tasks
- Build adaptive prompting strategies
- Create feedback loops between prompting methods

### 📚 Scenario
The "Mythbusters Institute" needs an AI system to analyze urban legends, conspiracy theories, and viral claims. Your system must identify claims, check logical consistency, detect common myth patterns, and provide fact-based analysis. Some tasks need examples (few-shot) while others need clear instructions (zero-shot).

### 🔧 Your Task
Build an `UrbanLegendChecker` class that:
1. Extracts claims using zero-shot prompting
2. Classifies myth types using few-shot examples
3. Detects logical fallacies with mixed approaches
4. Generates debunking explanations
5. Calculates believability scores

### 📝 Requirements
- Use zero-shot for open-ended analysis tasks
- Use few-shot for pattern matching and classification
- Dynamically switch between methods based on task
- Combine outputs from both approaches
- Handle various types of myths and claims

### 🎮 Challenges
1. **Basic**: Use zero-shot for claim extraction, few-shot for classification
2. **Intermediate**: Combine methods for comprehensive analysis
3. **Advanced**: Build adaptive system that chooses optimal method
4. **Bonus**: Create myth evolution tracker showing how stories change

### 💡 Hints
- Zero-shot excels at novel analysis tasks
- Few-shot is better for pattern recognition
- Combine methods for robust results
- Use confidence scores to weight outputs
- Consider task complexity when choosing method

### 🏆 Success Criteria
- Claims are accurately extracted and analyzed
- Myth patterns are correctly identified
- Logical fallacies are detected consistently
- Explanations are clear and fact-based
- System adapts to different myth types

### 🚀 Getting Started
1. Set up your OpenAI API key
2. Install packages: `pip install langchain-openai langchain-core`
3. Complete the TODO sections in `legend_checker.py`
4. Test with provided urban legends
5. Try viral social media claims!

### 🔍 What You'll Learn
- When to use zero-shot vs few-shot
- Combining multiple prompting strategies
- Building adaptive AI systems
- Handling misinformation analysis
