# PPTIS++ deployment

Use [backend/SETUP.md](backend/SETUP.md) and [frontend/SETUP.md](frontend/SETUP.md) for local install. This document covers production.

## Docker (optional)

Example layout (not committed as files; copy if you want containers):

**`backend/Dockerfile`**

```dockerfile
FROM python:3.11-slim
RUN apt-get update && apt-get install -y ffmpeg build-essential && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY backend /app
RUN pip install --no-cache-dir flask flask-cors python-dotenv PyJWT werkzeug sqlalchemy \
    psycopg2-binary numpy scikit-learn torch transformers faster-whisper groq gunicorn
ENV FLASK_ENV=production
EXPOSE 5000
CMD ["gunicorn", "-b", "0.0.0.0:5000", "-w", "2", "app:app"]
```

**`frontend/Dockerfile`**

```dockerfile
FROM node:20-alpine AS build
WORKDIR /ui
COPY frontend /ui
RUN npm ci && npm run build
FROM nginx:alpine
COPY --from=build /ui/dist /usr/share/nginx/html
```

**Compose sketch**

- `db`: Postgres 16, volume for data
- `api`: backend image, `DATABASE_URL`, secrets from env file
- `web`: frontend image or static bucket
- Do not run Flask’s builtin server as the public process; use gunicorn/uvicorn workers

Whisper/BERT weights are large. Prefer a persistent volume for Hugging Face cache (`HF_HOME`) so models are not re-downloaded on every deploy.

## Database migration

PostgreSQL is the only supported database.

```bash
export DATABASE_URL=postgresql://pptis:SECRET@db:5432/pptis
python database/migrate.py --postgres --skip-tests
```

Postgres uses `database/schema.sql` plus `database/slang_dictionary.csv`. Application tables added later (`rewrite_context`, `delayed_send_locks`, `intervention_effectiveness`) are created on first process start via SQLAlchemy.

Backup `slang_dictionary` and `behavior_profiles` before schema changes. There is no Alembic history yet; treat `schema.sql` as the contract and add additive migrations by hand.

## Secret management

Never commit `.env`. Inject at runtime:

- `SECRET_KEY`, `JWT_SECRET_KEY` — long random values (32+ bytes)
- `DATABASE_URL` / `POSTGRES_PASSWORD`
- `GROQ_API_KEY`
- `SMTP_*` if password reset email is enabled

Use a secrets manager (AWS Secrets Manager, GCP Secret Manager, Doppler, or systemd `EnvironmentFile` with `0600` perms). Rotate JWT secrets only with a planned logout of all sessions (tokens last 7 days).

## SSL/TLS

Terminate TLS at a reverse proxy:

- nginx or Caddy in front of gunicorn (`127.0.0.1:5000`) and the static UI
- Let’s Encrypt via Caddy or certbot
- Redirect HTTP → HTTPS
- Set `CORS_ORIGINS` to the HTTPS frontend origin only
- Send HSTS once certificates are stable

WebSockets are not required. Voice uploads are multipart; raise `client_max_body_size` (e.g. 10–25 MB).

## Monitoring and logging

Flask request/response logging is enabled in `app.py` (method, path, status, duration). Module loggers cover PACL, CEL, decisions, rewrites, and delayed-send.

Ship stdout to:

- journald / CloudWatch / Datadog
- retain 401/429 spikes (login lockout after 5 failures / 15 minutes)

Health: `POST /api/auth/login` is not a health check. Add a cheap `/health` later or scrape process liveness. Watch GPU/CPU when Whisper and Toxic-BERT load.

Trained logistic params persist in `backend/escalation_logistic.json`. Back that file up with the database.

## Scaling

- **API:** gunicorn with 2–4 sync workers per CPU for I/O routes. Model inference is heavy; keep a small worker count or a dedicated inference process.
- **Models:** load Whisper, Toxic-BERT, and XLM-R once per process. Avoid forking after CUDA init.
- **DB:** Postgres with connection pooling (PgBouncer) if worker count grows. The application does not start on SQLite.
- **Jobs:** delayed-send expiry is computed on read; no background worker is required.
- **Frontend:** static files scale independently of the API.
- **Fixed constants:** do not tune `λ` or `α` per instance; they must stay identical everywhere. Only `b`, `w_E`, `w_P` change after RIL retraining.

Horizontal scale: sticky sessions are unnecessary (JWT is stateless). Token blacklist is in-memory; logout will not replicate across processes unless you replace it with Redis.
