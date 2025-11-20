## Assignment 6: Reply Macro Composer (Runtime Configs)

Support agents often need clean, consistent macros. Build a tiny assistant that
turns a raw customer message + short context into a ready-to-send reply macro.

### Concepts practiced
- Runtime configuration via `.bind` and/or `.with_config`
- Chat prompts and LCEL composition
- Optional `.batch()` for bulk macro creation

### Your task
- See `macro_composer.py` for the class and method signatures. Implement the bodies
  based on the docstrings. Keep code concise and robust.

### Run
```
python macro_composer.py
```

Set `OPENAI_API_KEY` beforehand.


