## Assignment 2: AI Food Safety Inspector

Scenario: Reviews and social posts often hint at health code issues. Your AI assistant
should detect likely violations and recommend inspection priority — purely from
clear instructions (zero-shot), producing structured JSON that you map to Python types.

### Concepts practiced
- Zero-shot prompting for structured outputs
- `PromptTemplate.from_template`
- Chaining with `prompt | llm` and parsing JSON to dataclasses

### Your task
Complete the TODOs in `food_inspector.py` to:
- Craft a precise analysis prompt that extracts violations with evidence quotes
- Produce JSON and map it into `Violation` objects
- Compute an overall risk score and priority
- Aggregate across multiple reviews

### Run
```
python food_inspector.py
```

Set your API key first: `export OPENAI_API_KEY=...`

Keep temperature low for consistent schema adherence.


