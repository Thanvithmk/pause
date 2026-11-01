Historical record only. Light mode has been removed. Do not start the current application with a light-mode setting. See README.md.

# PPTIS++ classification QA audit

Executed: 1 November 2026. Light-mode lexicon + safety-pattern pipeline (`PPTIS_LIGHT_MODE=true`). Dataset: 265 new cases in `backend/tests/audit_dataset.py` (categories A–Q). Results workbook: `docs/PPTIS_PLUSPLUS_QA_AUDIT_RESULTS.xlsx`.

Labels were not retuned to inflate accuracy. Individual test sentences were not hardcoded into the classifier.

## 1. Totals

| Metric | All 265 | Supported language (263) |
| --- | ---: | ---: |
| Passed | 263 | 263 |
| Failed | 2 | 0 |
| Accuracy | 99.2% | 100% |
| False positives (over-protection) | 0 | 0 |
| False negatives (under-protection) | 2 | 0 |
| FP rate | 0.000 | 0.000 |
| FN rate | 0.0075 | 0.000 |

The two failures are Kannada-script and Hindi-script threats. They are **unsupported language** cases, not English classifier errors.

## 2. Accuracy by decision

| Decision | N | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| ALLOW | 116 | 100% | 98.3% | 100% | 99.1% |
| WARN | 35 | 100% | 100% | 100% | 100% |
| REWRITE | 56 | 100% | 100% | 100% | 100% |
| DELAYED-SEND | 58 | 96.6% | 100% | 96.6% | 98.2% |

ALLOW precision is below 100% only because the two unsupported-script threats were predicted ALLOW.

## 3. Confusion matrix (gold × predicted)

|  | ALLOW | WARN | REWRITE | DELAYED-SEND |
| --- | ---: | ---: | ---: | ---: |
| ALLOW | 116 | 0 | 0 | 0 |
| WARN | 0 | 35 | 0 | 0 |
| REWRITE | 0 | 0 | 56 | 0 |
| DELAYED-SEND | 2 | 0 | 0 | 56 |

## 4. False positives

None on this suite. Harmless greetings, civil disagreement, quoted reports, playful death idioms, and casual profanity stayed ALLOW or WARN as labeled.

## 5. False negatives

| ID | Language | Message | Expected | Actual |
| --- | --- | --- | --- | --- |
| A196 | kn-script | ನಾನು ನಿನ್ನನ್ನು ಹೊಡೆಯುತ್ತೇನೆ | DELAYED-SEND | ALLOW |
| A197 | hi-script | मैं तुम्हें मारूँगा | DELAYED-SEND | ALLOW |

Latin-script Kannada/Hindi threats (`maar dunga`, `hodiyutte`, `ghar ke samne`) pass. Native scripts have no translation layer in light mode.

## 6. Gen-Z slang (E_slang)

30 / 30. Ambiguous terms (`cooked`, `fire`, `dead`, `mid`, `L`/`W`, `it's giving`) were scored from conversational use, not from dictionary presence.

## 7. Contextual disambiguation (J_context)

15 / 15. Prior messages are passed separately into the safety layer (not concatenated as the current utterance). Examples: “You're dead.” after a schedule threat vs after a joke; “Nice work, genius.” after a bugfix vs after deleting a database; “Don't make me come over there.” in a family vs bat/money thread.

Live `/api/analyze` now loads the last four stored messages as that context.

## 8. Threat detection

- Category D (explicit threats): 20 / 20.
- All DELAYED-SEND cases except native-script: 56 / 56.
- 50-sheet benchmark T001–T050: pass.
- Regression: “You're dead. I know where you'll be after class.” → DELAYED-SEND; “Bro, you're dead 💀” → ALLOW.

## 9. Rewrite quality

Template fallback (no Groq in this run) evaluated on all 56 REWRITE gold cases:

- Safer than original (risk re-check): 56 / 56
- Re-classified ALLOW or WARN: 56 / 56
- No leftover lexicon insults: 56 / 56

Personal attacks are replaced with a generic disagreement line when the remainder still targets *you/your*. Candidates that still score REWRITE/DELAYED-SEND are discarded. Sending the original REWRITE/DELAYED-SEND text remains blocked in `send_gateway`.

## 10. Before vs after

Same 265 labels, same formulas (λ = 0.5, α = 0.5, 6-D vector, R unchanged).

| | Before first safety rewrite | After this audit |
| --- | ---: | ---: |
| Accuracy | 45.7% (121 / 265) | 99.2% (263 / 265) |
| FP rate | 0.075 | 0.000 |
| FN rate | 0.468 | 0.008 |

An intermediate checkpoint after the first pattern pass was 92.5% (245 / 265). Remaining English FNs were quoted-apostrophe false “quotes”, missing context wiring, game-harm vs person-harm, identity coercion, and command-form “Come outside. Now.”

## 11. Files modified

- `backend/modules/safety_patterns.py` — sentence-max harm, quotes, negation, playful idioms, identity/threat/insult, context-aware priors. Documented separately from R.
- `backend/modules/text_normalize.py` — leetspeak, emoji weapons, intra-word dots; does not store mutated text.
- `backend/modules/toxicity_model.py` — word-boundary lexicon; `analyze(..., context=)`; merge with safety overlay.
- `backend/modules/decision_engine.py` — escalate-only guardrail on threat / identity / severe (does not change R).
- `backend/modules/rewrite_engine.py` — safer template fallback + toxicity re-check.
- `backend/api/routes.py` — prior-message context into toxicity analysis.
- `backend/tests/audit_dataset.py`, `audit_runner.py`, `test_safety_patterns.py`, `test_modules.py` — new suite and regressions.
- `database/slang_dictionary.csv` — extra Gen-Z / regional senses (disambiguation, not auto-toxicity).

R remains `α·PACL + (1-α)·Pesc` with PACL `clip(Tb + λ(Tb − BT), 0, 1)` and `Pesc = σ(b + wE·E + wP·PACL)`.

## 12. Remaining limitations

1. Native Kannada/Hindi (and other non-Latin) scripts are unsupported in light mode. Latin transliteration is covered for a small threat/insult set only.
2. Light mode does not run Toxic-BERT; production with `PPTIS_LIGHT_MODE=false` can still disagree with this heuristic overlay.
3. Conversation context is the last four stored messages, not a retrieved semantic window.
4. Template rewrites preserve criticism of *work* only in a generic way; Groq/local LLM rewrites were not scored in this run.
5. Flask must be restarted to load the new safety/context code for the live UI.

## Frontend / backend policy (code + unit tests)

Verified in `tests.test_send_policy` (68 backend tests total, all passing):

- ALLOW send, WARN requires `send_anyway`, REWRITE blocks original, DELAYED-SEND lock cannot be bypassed via `/api/send`.
- Edited text vs `analyzed_text` → 409 re-analyze.
- Duplicate send → 409 already sent.

UI (static review): `AnalysisResults` shows decision + `risk_score`; `formatClock` is HH:MM 24h; DELAYED-SEND locks the Send button.

## Error source map (not only the toxicity model)

| Failure class (pre-fix) | Module |
| --- | --- |
| Substring “hate” in “whatever” | lexicon (`toxicity_model`) |
| Apostrophes treated as quotes (`You're` / `you'll`) | `QUOTE` regex |
| Context mixed into current text | `audit_runner` / missing API context |
| Ranking/GPA “hurt” as physical threat | threat vs `GAME_HARM` |
| PACL using only Tb | documented Decision Engine guardrail |
| Harmless criticism as WARN | over-broad frustration patterns (tightened) |
