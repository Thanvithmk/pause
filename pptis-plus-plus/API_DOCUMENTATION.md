# PPTIS++ API reference

Base URL (local): `http://localhost:5000`

JSON unless noted. Authenticated routes send:

```
Authorization: Bearer <access_token>
```

Access tokens are JWTs (HS256), type `access`, valid **7 days**. Refresh tokens use type `refresh`. Login lockout: **5** failed attempts, then **15 minutes** (in-memory, per IP + identifier).

---

## Authentication

### POST `/api/auth/register`

Also available as `POST /auth/register`.

**Request**

```json
{
  "username": "alexsafe",
  "email": "alex@example.com",
  "password": "TestPass1!"
}
```

Username: alphanumeric, 3–20 chars. Password: 8+ chars, uppercase, digit, special character.

**201**

```json
{
  "user_id": 1,
  "token": "<jwt>",
  "user": {
    "user_id": 1,
    "username": "alexsafe",
    "email": "alex@example.com",
    "created_at": "2026-11-01T08:00:00+00:00"
  }
}
```

**400** validation, **409** duplicate username/email.

### POST `/api/auth/login`

Also `POST /auth/login`.

**Request** — `email`, `username`, or `identifier` plus `password`.

```json
{
  "email": "alex@example.com",
  "password": "TestPass1!"
}
```

**200** — `user_id`, `token`, `refresh_token`, `user`.

**401** invalid credentials, **429** too many failures.

### POST `/auth/logout`

Blacklists the current JWT `jti`. **401** if missing/expired.

### POST `/auth/refresh`

Body optional: `{ "refresh_token": "..." }` or Bearer refresh token. Returns a new access + refresh pair. Old refresh is revoked.

### POST `/auth/password-reset`

`{ "email": "alex@example.com" }`. Always **200** with a generic message. In `FLASK_ENV=development` the response may include `reset_token`.

### POST `/auth/password-reset/confirm`

`{ "token": "<reset jwt>", "password": "NewPass1!" }` → `{ "message": "Password updated" }`.

---

## Analysis

### POST `/api/analyze`

**Auth required.** Validates conversation ownership when `conversation_id` is sent.

**Request**

```json
{
  "text": "Bro you're cooked",
  "input_type": "text",
  "conversation_id": 3
}
```

`conversation_id` is optional; a new conversation is created for the user.

**200** (fields always present; rewrite/delay only when decided)

```json
{
  "message_id": 12,
  "conversation_id": 3,
  "original_text": "Bro you're cooked",
  "processed_text": "Bro you're cooked",
  "enriched_text": "Bro you're cooked [cooked = defeated/finished/in serious trouble]",
  "input_type": "text",
  "toxicity": {
    "toxic": 0.21,
    "threat": 0.04,
    "insult": 0.11,
    "severe_toxic": 0.02,
    "obscene": 0.03,
    "identity_hate": 0.01
  },
  "pacl": 0.24,
  "delta_vector_6d": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  "escalation_feature": 0.0,
  "escalation_probability": 0.56,
  "final_risk": 0.40,
  "decision": "WARN",
  "confidence": 0.8,
  "color_for_ui": "#eab308",
  "description": "Moderate risk, show warning",
  "action": {},
  "rewrite": null,
  "delayed_send": null,
  "allow_send_original": false,
  "allow_send_anyway": false
}
```

Pipeline: preprocess → slang disambiguation → toxicity `X_t` → PACL (`λ = 0.5` fixed) → CEL `E = W·ΔX` → `P_esc` → `R = α·PACL + (1-α)·P_esc` (`α = 0.5` fixed) → decision.

| `R` | Decision |
|---|---|
| `[0.00, 0.30)` | ALLOW |
| `[0.30, 0.55)` | WARN |
| `[0.55, 0.80)` | REWRITE |
| `[0.80, 1.00]` | DELAYED-SEND |

REWRITE adds `rewrite.suggested_text` (no send-original). DELAYED-SEND adds `delayed_send.lock_id` and `countdown_seconds` (default 60).

**400** missing text, **401** no/invalid token, **403** conversation not owned, **404** conversation missing, **500** model/storage failure.

### POST `/api/analyze-voice`

**Auth required.** Multipart form:

- `audio` (file, required)
- `conversation_id` (optional)

Transcribes with Faster-Whisper (`base.en`), then the same pipeline as `/api/analyze`. Extra fields: `transcription`, `transcription_confidence`.

**400** if `audio` missing, transcription fails, or no speech.

---

## User and dashboard

### GET `/api/user/profile`

**200**

```json
{
  "user": { "user_id": 1, "username": "alexsafe", "email": "alex@example.com" },
  "profile": {
    "user_id": 1,
    "avg_toxic": 0.12,
    "avg_threat": 0.04,
    "avg_insult": 0.05,
    "avg_severe_toxic": 0.01,
    "avg_obscene": 0.02,
    "avg_identity_hate": 0.01,
    "message_count": 18
  },
  "baseline": { "B_T": 0.12, "B_H": 0.04, "B_I": 0.05, "B_S": 0.01, "B_O": 0.02, "B_IH": 0.01, "vector": [] },
  "intervention_statistics": {
    "messages_count": 10,
    "avg_toxicity": 0.14,
    "escalations_count": 2,
    "interventions_count": 2
  },
  "preferences": { "preferred": "REWRITE", "scores": {} },
  "insights": ["Your average toxicity decreased 18% this month"]
}
```

### GET `/api/dashboard?days=30`

`days` defaults to 30.

**200** includes `messages_analyzed`, per-dimension averages, `escalation_count`, `warning_count`, `rewrite_count`, `delayed_send_count`, `toxicity_trends`, `message_outcomes`, `decision_breakdown`, `escalation_history`, `recent_activity`, `insights`.

### GET `/api/conversation/<conversation_id>`

Owner only.

**200** `{ "conversation": {...}, "messages": [ { "message", "toxicity", "decision", "interventions" } ] }`

**403** other user’s thread, **404** unknown id.

---

## Error format

```json
{
  "error": "api_error",
  "message": "text is required"
}
```

Auth failures use `"error": "auth_error"`. Flask also maps:

| Code | When |
|---|---|
| 400 | Bad JSON, validation, failed transcription |
| 401 | Missing/expired/revoked JWT, bad password |
| 403 | Conversation not owned |
| 404 | Missing conversation/resource |
| 405 | Wrong HTTP method |
| 429 | Login rate limit |
| 500 | Unhandled / model / database |

---

## Rate limiting

Implemented for **failed logins only** (5 attempts, 15 minute lock). Analyze routes are not rate-limited in application code; cap them at the reverse proxy if needed.

---

## Notes

- Tokens: put `Authorization: Bearer` on every analyze/profile/dashboard/conversation call.
- First-message CEL uses `E = 0` and a zero `ΔX`.
- `λ` and `α` are not client-configurable.
- There is no API to send a blocked message without re-analysis.
