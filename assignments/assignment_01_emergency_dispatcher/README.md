## Assignment 1: Emergency Call Triage Assistant

Engage with a realistic dispatch scenario. You will design a tiny LLM-powered assistant that reads a caller transcript and produces:

- an urgency label (EMERGENCY, NON_EMERGENCY, UNKNOWN)
- a one-sentence summary
- a recommended next action

### Concepts practiced
- Chat model basics with `ChatOpenAI`
- `ChatPromptTemplate.from_messages`
- LCEL chaining with `prompt | llm | StrOutputParser()`

### Your task
Fill in the TODOs inside `emergency_dispatcher.py` to:
- Build a role-aware prompt (system + user) that instructs the assistant to be concise and safe
- Create a small chain that returns a compact, single-line, pipe-delimited string: `URGENCY | SUMMARY | ACTION`
- Parse and map that string into a `DispatchResult` dataclass

### Acceptance criteria
- The script prints a structured triage result for each example transcript
- Urgency is one of: `EMERGENCY`, `NON_EMERGENCY`, `UNKNOWN`
- Summary is ≤ 20 words
- Action begins with a verb (e.g., "Dispatch ambulance", "Provide reassurance")

### Run
```
python emergency_dispatcher.py
```

Set your API key first: `export OPENAI_API_KEY=...`

Tip: Start with low temperature for consistent outputs.


