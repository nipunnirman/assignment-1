## Assignment 3: Transit Delay Explainer

When trains or buses report cryptic status messages, riders want a clear, friendly
explanation and what to do next. You'll build a small assistant that translates
an operations bulletin into rider guidance.

### Concepts practiced
- Prompt design with `ChatPromptTemplate`
- LCEL chaining: `prompt | llm | StrOutputParser()`
- Parameter tuning: `temperature`, `top_p` (compare styles)

### Your task
Complete the TODOs in `transit_delay_explainer.py`:
- Build a role-aware prompt with variables: `{status_text}`, `{line_name}`
- Create two model configurations (calm vs creative) and compare outputs
- Return a final advisory string limited to 2 bullet points

### Run
```
python transit_delay_explainer.py
```

Set `OPENAI_API_KEY` and keep temperature low for the "calm" model. Observe how
raising `temperature` and/or `top_p` changes tone in the "creative" model.


