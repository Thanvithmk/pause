# PPTIS++ backend setup

Python 3.11+ is recommended. Run commands from `pptis-plus-plus/backend` unless noted.

## Installation

```bash
cd backend
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install flask flask-cors python-dotenv PyJWT werkzeug sqlalchemy psycopg2-binary
pip install numpy scikit-learn torch transformers faster-whisper groq
```

Optional voice/GPU packages: `faster-whisper` needs FFmpeg on the host.

`requirements.txt` is a placeholder; install the packages above until it is filled.

## Database setup

**SQLite (default)**

```bash
cd ../database
python migrate.py
```

This creates `database/pptis.db` from SQLAlchemy models, loads `slang_dictionary.csv`, seeds a demo user, and runs tests.

**PostgreSQL**

1. Create a database named `pptis`.
2. Set `USE_POSTGRES=true` or `DATABASE_URL=postgresql://user:pass@host:5432/pptis`.
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
| `JWT_SECRET_KEY` | JWT signing | `change-me-jwt` |
| `DATABASE_URL` | Full DB URI | SQLite file |
| `USE_POSTGRES` | Build URI from `POSTGRES_*` | `false` |
| `CORS_ORIGINS` | Allowed frontend origins | `http://localhost:3000` |
| `WHISPER_MODEL` | Faster-Whisper model | `base.en` |
| `GROQ_API_KEY` | Rewrite LLM | unset (templates) |
| `LOGISTIC_B`, `LOGISTIC_W_E`, `LOGISTIC_W_P` | Trained `P_esc` weights | `0`, `1`, `1` |

Fixed (not env-tuned): `PACL_LAMBDA = 0.5`, `FINAL_RISK_ALPHA = 0.5`, decision thresholds `0.30 / 0.55 / 0.80`.

## Running the Flask server

```bash
cd backend
export FLASK_ENV=development
python app.py
```

Listens on `0.0.0.0:5000` by default (`HOST`, `PORT`). First analyze request may download Whisper/BERT/XLM-R models.

## API endpoints

Prefix `/api` is registered in `app.py`. Legacy auth also exists at `/auth/*`.

| Method | Path | Auth |
|---|---|---|
| POST | `/api/auth/register` | no |
| POST | `/api/auth/login` | no |
| POST | `/auth/logout` | bearer |
| POST | `/auth/refresh` | refresh token |
| POST | `/api/analyze` | bearer |
| POST | `/api/analyze-voice` | bearer |
| GET | `/api/user/profile` | bearer |
| GET | `/api/dashboard?days=30` | bearer |
| GET | `/api/conversation/<id>` | bearer (owner) |

See [API_DOCUMENTATION.md](../API_DOCUMENTATION.md) for the full reference.

## Example requests

Register:

```bash
curl -s http://localhost:5000/api/auth/register \
  -H 'Content-Type: application/json' \
  -d '{"username":"alexsafe","email":"alex@example.com","password":"TestPass1!"}'
```

Response (201):

```json
{
  "user_id": 1,
  "token": "<jwt>",
  "user": { "username": "alexsafe", "email": "alex@example.com" }
}
```

Login:

```bash
curl -s http://localhost:5000/api/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"email":"alex@example.com","password":"TestPass1!"}'
```

Analyze text:

```bash
curl -s http://localhost:5000/api/analyze \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"text":"Bro you are cooked","input_type":"text"}'
```

Response includes `toxicity`, `pacl`, `escalation_probability`, `final_risk`, `decision`, and optional `rewrite` / `delayed_send`.

Analyze voice:

```bash
curl -s http://localhost:5000/api/analyze-voice \
  -H "Authorization: Bearer $TOKEN" \
  -F audio=@sample.wav
```

Password rules: 8+ characters, one uppercase letter, one digit, one special character. Usernames are alphanumeric, 3–20 characters.

## Tests

```bash
cd backend
PYTHONPATH=. python -m unittest discover -s tests -p 'test_*.py' -v
```
