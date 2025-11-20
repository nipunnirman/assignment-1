# LangChain LLM Bootcamp Assignments 🚀

Welcome to the assignments section! These 10 progressive exercises will help you master LLM interaction using LangChain, covering zero-shot, few-shot, and chain-of-thought prompting techniques.

## 📚 Assignment Overview

### Foundation (Single Concepts)
1. **Dream Journal Analyzer** - Zero-shot Prompting Fundamentals
   - Extract symbols and emotions from dreams without examples
   - Master structured output generation
   
2. **Food Safety Inspector** - Zero-shot with Structured Outputs  
   - Analyze restaurant reviews for health violations
   - Build classification systems with clear instructions

3. **Escape Room Puzzle Master** - Few-shot Pattern Recognition
   - Learn from puzzle examples to generate new ones
   - Dynamic example selection strategies

4. **Time Travel Paradox Resolver** - Chain of Thought Reasoning
   - Step-by-step logical analysis of temporal scenarios
   - Compare zero-shot vs few-shot CoT approaches

### Intermediate (Combined Concepts)
5. **Urban Legend Fact Checker** - Zero-shot + Few-shot
   - Combine techniques for myth analysis
   - Adaptive prompting strategies

6. **Alien Language Translator** - Few-shot + Chain of Thought
   - Pattern recognition with logical deduction
   - Decode messages using examples and reasoning

7. **Conspiracy Theory Debunker** - Zero-shot + Chain of Thought
   - Critical analysis with step-by-step reasoning
   - Handle unique claims without examples

### Advanced (All Concepts)
8. **Superhero Power Balancer** - Complete Integration
   - Game balance using all three techniques
   - Choose optimal method for each subtask

9. **Mystery Dinner Party Solver** - Complex Problem Solving
   - Solve murder mysteries with combined approaches
   - Profile, analyze, deduce using all techniques

10. **AI Dungeon Master** - The Ultimate Challenge
    - Run complete D&D sessions
    - Seamlessly switch between all techniques
    - Creative storytelling meets rule-based logic

## 🎯 Learning Path

### Week 1: Foundation
- Complete Assignments 1-4
- Focus on mastering each individual technique
- Understand when each approach works best

### Week 2: Integration
- Complete Assignments 5-7
- Learn to combine techniques effectively
- Build adaptive systems

### Week 3: Mastery
- Complete Assignments 8-10
- Apply all techniques seamlessly
- Create complex, production-ready systems

## 🛠️ Setup Instructions

1. **Install Dependencies**:
```bash
pip install langchain-openai langchain-core
```

2. **Set OpenAI API Key**:
```bash
export OPENAI_API_KEY='your-api-key-here'
```

3. **Navigate to Assignment**:
```bash
cd assignment_01_dream_analyzer/
```

4. **Read the README**:
Each assignment has detailed instructions and learning objectives.

5. **Complete TODOs**:
Follow the guided TODOs in each Python file.

6. **Test Your Solution**:
```bash
python dream_analyzer.py  # Or respective file name
```

## 💡 Tips for Success

### For Beginners
- Start with Assignment 1 and work sequentially
- Read all comments and TODO instructions carefully
- Test with provided examples before creating your own
- Don't worry about perfect solutions - focus on learning

### Understanding the Concepts
- **Zero-shot**: Give clear instructions without examples
- **Few-shot**: Provide examples for pattern matching
- **Chain of Thought**: Break down reasoning step-by-step

### Common Pitfalls to Avoid
- Don't provide examples when using zero-shot
- Ensure examples are high-quality for few-shot
- Make reasoning steps explicit for CoT
- Test edge cases and ambiguous inputs

## 🏆 Completion Criteria

For each assignment, ensure:
- [ ] All TODO sections are completed
- [ ] Test cases run successfully
- [ ] Output matches expected format
- [ ] Edge cases are handled
- [ ] Code is well-commented

## 🔧 Troubleshooting

### Import Errors
Make sure you've installed all dependencies:
```bash
pip install langchain-openai langchain-core
```

### API Key Issues
Verify your OpenAI API key is set:
```python
import os
print(os.environ.get("OPENAI_API_KEY"))  # Should show your key
```

### Rate Limiting
If you hit rate limits, add delays between API calls:
```python
import time
time.sleep(1)  # Wait 1 second between calls
```

## 📊 Assessment Rubric

Each assignment is evaluated on:
- **Functionality** (40%): Does it work as specified?
- **Code Quality** (20%): Clean, readable, well-structured?
- **Prompt Engineering** (30%): Effective use of techniques?
- **Edge Cases** (10%): Handles unusual inputs?

## 🎓 After Completion

Once you've completed all 10 assignments, you'll be able to:
- Build production-ready LLM applications
- Choose the optimal prompting strategy for any task
- Combine techniques for complex problems
- Debug and optimize LLM interactions
- Create engaging AI-powered experiences

## 🤝 Getting Help

- Review the README in each assignment folder
- Check the test functions for usage examples
- Experiment with different prompt variations
- Remember: There's often more than one correct solution!

Happy coding! 🚀 May your prompts be precise and your outputs structured!
