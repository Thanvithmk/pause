# Pause backend setup

Python 3.11+ is recommended. Run commands from `pptis-plus-plus/backend` unless noted.

## Installation

```bash
cd backend
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install flask flask-cors python-dotenv werkzeug sqlalchemy psycopg2-binary
pip install numpy scikit-learn torch transformers groq
```

`requirements.txt` is a placeholder; install the packages above until it is filled.

## Database setup

PostgreSQL is required. The application refuses to start when the database URI is not `postgresql://`.

1. Create a database named `pptis`.
2. Set `DATABASE_URL=postgresql://user:pass@host:5432/pptis`.
3. Apply schema and slang:

```bash
cd database
python migrate.py --postgres
# or
python init_db.py
```

`schema.sql` is the source of truth for Postgres types and tables. Extra tables created by the app (`rewrite_context`, `delayed_send_locks`, `intervention_effectiveness`) are added via SQLAlchemy `create_all` on first use.

## Environment variables

Copy `backend/.env.example` to `backend/.env` and edit values.

| Variable | Purpose | Default |
|---|---|---|
| `FLASK_ENV` | `development` / `production` | `production` |
| `SECRET_KEY` | Flask sessions | `change-me` |
| `DATABASE_URL` | PostgreSQL URI | required |
| `CORS_ORIGINS` | Allowed frontend origins | `http://localhost:3000` |
| `GROQ_API_KEY` | Groq rewrite key | required |
| `GROQ_MODEL` | Groq model | `openai/gpt-oss-120b` only |
| `LOGISTIC_B`, `LOGISTIC_W_E`, `LOGISTIC_W_P` | Trained `P_esc` weights | `0`, `1`, `1` |

Fixed (not env-tuned): `PACL_LAMBDA = 0.5`, `FINAL_RISK_ALPHA = 0.5`, decision thresholds `0.30 / 0.55 / 0.80`.

## Running the Flask server

```bash
cd backend
export FLASK_ENV=development
python app.py
```

Listens on `0.0.0.0:5000` by default (`HOST`, `PORT`). Startup loads `xlm-roberta-base`, `backend/models/slang_literal_head.pt` (threshold 0.58), and `unitary/toxic-bert`. If one of those cannot load, the process exits. Groq is checked for a configured key and the model name `openai/gpt-oss-120b`. A failed Groq call returns an error and does not invent a rewrite.

## API endpoints

Prefix `/api` is registered in `app.py`. There is no login. Calls are stored on the local `pause` profile.

| Method | Path |
|---|---|
| POST | `/api/analyze` |
| GET | `/api/user/profile` |
| GET | `/api/dashboard?days=30` |
| GET | `/api/conversation/<id>` |

See [API_DOCUMENTATION.md](../API_DOCUMENTATION.md) for the full reference.

## Example request

```bash
curl -s http://localhost:5000/api/analyze \
  -H 'Content-Type: application/json' \
  -d '{"text":"Bro you are cooked","input_type":"text"}'
```

Response includes `toxicity`, `pacl`, `escalation_probability`, `final_risk`, `decision`, and optional `rewrite` / `delayed_send`.

## Tests

The default suite uses the real slang and toxicity checkpoints. Groq completions inside the end-to-end tests are mocked. The live Groq call is separate:

```bash
cd backend
PYTHONPATH=. python -m unittest discover -s tests -p 'test_*.py' -v
PYTHONPATH=. python checks/groq_live_check.py
```
