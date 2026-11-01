# PPTIS++

The application always loads the Master Plan models. It does not switch to a smaller model when one of them fails.

| Component | Model |
| --- | --- |
| Contextual slang | Frozen `xlm-roberta-base` and `backend/models/slang_literal_head.pt`, threshold 0.58 |
| Toxicity | `unitary/toxic-bert`, six scores |
| Voice | Faster-Whisper `base.en` |
| Rewrite | Groq `llama-3.1-8b-instant` |
| Database | PostgreSQL |

Startup loads the local models, runs one inference for each, and exits if a model or the trained slang head cannot load. A missing Groq key or a different `GROQ_MODEL` also stops startup. If Groq fails during a rewrite, the API returns an error for that component and does not invent a suggestion or allow the message to send on that failed analysis.

Setup: [backend/SETUP.md](backend/SETUP.md), [frontend/SETUP.md](frontend/SETUP.md), [DEPLOYMENT.md](DEPLOYMENT.md).

```bash
cd backend
PYTHONPATH=. python -m unittest discover -s tests -p 'test_*.py' -q
PYTHONPATH=. python checks/groq_live_check.py
```

The unit suite runs the real slang, toxicity, and Whisper models. End-to-end rewrite calls use a mocked Groq response. `checks/groq_live_check.py` is the live API call.
