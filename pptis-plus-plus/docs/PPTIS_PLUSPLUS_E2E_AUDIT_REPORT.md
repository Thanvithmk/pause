# PPTIS++ end-to-end application audit

Executed: 1 November 2026. Backend tests: **77 passed** (`PPTIS_LIGHT_MODE=true`). Database under test: SQLite in-memory (`TestingConfig`). Production may use PostgreSQL when `USE_POSTGRES=true`.

R formulas were not changed. Alpha = 0.5, lambda = 0.5.

## Architecture (as implemented)

React (Vite) `frontend/src` → Axios `http://localhost:5000` → Flask `create_app` (`/api`, `/auth`) → SQLAlchemy (`users`, `conversations`, `messages`, `toxicity_results`, `behavior_profiles`, `risk_results`, `intervention_logs`, `feedback`, `slang_dictionary`, `unknown_slang`, delayed-send locks).

Pipeline on `/api/analyze`: preprocess → slang interpret → toxicity (6-D) + safety overlay → previous toxicity → PACL from Tb vs BT → CEL from ΔX → Pesc → R → Decision Engine (+ documented escalate-only guardrail) → rewrite or delayed-send lock → intervention log. `/api/send` is the only send path and re-checks the stored decision.

## Component status

| Component | Status | Problems | Severity |
|---|---|---|---|
| Authentication | Working after fix | Logout now hits `/api/auth/logout` and blacklists the JWT. Frontend previously only cleared localStorage. `/auth/*` still duplicates `/api/auth/*`. Password reset exists without UI. | Medium (duplicate routes / no reset UI) |
| Conversations | Working after fix | No conversation switcher or create endpoint before this audit. Ownership checks were already present. | Was High |
| Preprocessing | Working | Emoji/contractions preserved. Not independently load-tested on huge payloads. | Low |
| Slang intelligence | Working for detection/enrichment | Dictionary match does not set the decision. No admin review UI for `unknown_slang`. | Medium |
| Toxicity model | Working in light mode | HF timeout previously returned a zero vector (silent ALLOW). Now raises 503. Light mode uses lexicon + safety patterns, not Toxic-BERT. | Was Critical |
| Behavioral profile | Working after fix | Profile was updated on every analyze, including unsent drafts. Now updates on successful send only. | Was High |
| PACL/CEL | Matches spec | `PACL = clip(Tb + λ(Tb−BT), 0, 1)`; `E = W·ΔX`; 6-D vectors validated. New users BT = 0. | None |
| Escalation | Matches spec | `Pesc = σ(b + wE·E + wP·PACL)` with config b, wE, wP. Learning must not change α/λ (it does not). | None |
| Decision Engine | Working | Thresholds R&lt;0.30 ALLOW, &lt;0.55 WARN, &lt;0.80 REWRITE, else DELAYED-SEND. Guardrail may escalate from 6-D threat/identity; it does not alter R. | Low (guardrail is extra, documented) |
| Rewrite | Working | Last 5 prior messages loaded chronologically. Duplicate rewrite call from `apply_decision` removed. Groq optional; templates used when LLM absent. | Medium (no live Groq in this run) |
| Delayed-send | Working | Lock blocks all sends for the user/conversation. Status endpoint now checks ownership. No Send Anyway. | Was High (status leak) |
| PostgreSQL | Schema present | Runtime in this audit is SQLite unless `.env` sets Postgres. Optional columns added via `ensure_optional_columns`. | Medium (env-dependent) |
| Frontend | Working after fix | Conversation switcher, logout revoke, voice transcript object bug, dashboard KPIs. Live browser click-through not repeated this run. | Medium |
| Dashboard | Working after fix | Now reads DB counts, average risk, slang stats, profile. SQLite `date()` incompatibility fixed. Counts include analyzed drafts, not only sent. | Low |
| Security | Working after fix | Unauthenticated 401, cross-user 403, send bypass blocked, edit invalidates analysis. Analysis failure no longer becomes ALLOW. | Was Critical |
| Voice | Partial | Upload + pipeline wired. Whisper is not loaded in light mode (honest failure, not fake transcript). | High remaining |
| Feedback / RIL | Partial | WARN `send_anyway` now writes `feedback`. No dashboard control to rate rewrite quality. | Medium |
| Admin slang | Incomplete | `reload_dictionary` / unknown-slang table exist; no admin API or UI. | Low |

## Fixes applied

1. Model inference failure raises `ToxicityAnalysisError` → HTTP 503 (not ALLOW).
2. Delayed-send status requires message ownership.
3. Logout revokes JWT; frontend calls `/api/auth/logout`.
4. `POST /api/conversations` + chat switcher / new chat.
5. Behavior profile updates on send, not analyze.
6. WARN send writes RIL `feedback`.
7. Dashboard: average risk, sent count, slang stats, profile, SQLite-safe day grouping.
8. Voice UI uses `original_text`, not the transcription object.
9. DecisionEngine no longer runs a second rewrite; routes own rewrite generation.
10. Whisper skipped in light mode instead of downloading weights on first `/analyze-voice`.

## Tests actually executed

`python -m unittest discover -s tests` → **77 OK** in 1.7s, including:

- `tests.test_e2e_application` (register → conversation → ALLOW send → history; WARN/REWRITE/DELAYED-SEND policy; slang ALLOW; cross-user 403; dashboard from DB; logout revoke; voice missing-file; invalid send)
- `tests.test_send_policy`
- `tests.test_safety_patterns` (50-sheet + threat regressions)
- `tests.test_risk_calculation` (PACL/CEL/Pesc/R)
- `tests.test_modules`, `tests.test_integration`, `tests.test_preprocessing`

Not executed this run: real microphone → Whisper (light mode), PostgreSQL server round-trip, Groq rewrite, browser UI click-through.

## Remaining incomplete

- Enable `PPTIS_LIGHT_MODE=false` and a Whisper model for real voice transcription.
- Point `.env` at PostgreSQL (`USE_POSTGRES=true`) for production storage.
- Admin slang review UI.
- User-facing rewrite feedback beyond WARN send-anyway.
- Restart Flask to load these API changes in a running process.

## Reproduce remaining voice limitation

```bash
cd pptis-plus-plus/backend
PPTIS_LIGHT_MODE=true python app.py
# POST /api/analyze-voice with an audio file → 400
# "Speech-to-text is unavailable until Whisper is enabled"
```
