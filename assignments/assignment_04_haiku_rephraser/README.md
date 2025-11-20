## Assignment 4: Haiku Rephraser (Streaming)

Turn a short poem into a themed haiku while streaming tokens to the console.

### Concepts practiced
- Streaming with a custom callback handler
- Using a separate non-streaming chain for final tidy-up

### Your task
Implement the TODOs in `haiku_rephraser.py`:
- Build a streaming `ChatOpenAI` with a callback that prints tokens
- Prompt: Reformulate input into a haiku about a chosen theme
- After streaming, run a non-streaming clean-up pass to enforce syllable count hint

### Run
```
python haiku_rephraser.py
```

Try different themes like "rain", "dawn", "city".


