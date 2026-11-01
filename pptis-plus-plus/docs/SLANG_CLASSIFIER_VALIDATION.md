# Slang classifier validation

The running application no longer loads a word-and-character fallback. Any fallback numbers below are the measurement that led to removing it. They are not a current runtime path.

Measured on 1 Nov 2026. The head was not retrained and the 0.58 threshold was not refit.

## Setup

- Encoder: frozen `xlm-roberta-base`
- Head: `backend/models/slang_literal_head.pt`
- Primary threshold: 0.58
- Fallback: word-and-character logistic regression, threshold 0.70
- Audit file: `backend/data/slang_independent_audit.csv` (168 rows, no sentence shared with the training file)
- Metrics file: `backend/models/slang_independent_audit.json`
- Tests: `backend/tests/test_slang_primary_path.py`

136 rows are labelled slang or literal and were scored by calling the classifier on the named target in the current sentence. 32 rows are unknown or misspelled targets and were scored as a detection task.

Uncertain predictions are a third API usage. They are not counted as literal. In precision and recall below, an uncertain prediction is a miss for the true class.

## Primary encoder

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| Literal | 0.778 | 0.824 | 0.800 | 51 |
| Slang | 0.956 | 0.765 | 0.850 | 85 |

Macro F1 0.825. Accuracy 0.787. Accuracy on confident predictions only 0.877. Uncertain rate 0.103.

Confusion matrix, rows actual, columns predicted literal / slang / uncertain:

| Actual | Literal | Slang | Uncertain |
| --- | ---: | ---: | ---: |
| Literal | 42 | 3 | 6 |
| Slang | 12 | 65 | 8 |

## Fallback

Macro F1 0.441. Accuracy 0.353. Uncertain rate 0.456. Slang recall 0.129.

| Actual | Literal | Slang | Uncertain |
| --- | ---: | ---: | ---: |
| Literal | 37 | 1 | 13 |
| Slang | 25 | 11 | 49 |

## Category macro F1

| Category | N | Primary | Fallback |
| --- | ---: | ---: | ---: |
| Contrast | 36 | 0.853 | 0.479 |
| Ambiguous | 16 | 0.866 | 0.333 |
| Sarcasm and humour | 16 | 0.768 | 0.405 |
| Two senses in one sentence | 16 | 0.304 | 0.318 |
| Kannada-English | 16 | 0.980 | 0.635 |
| Hindi-English | 16 | 0.900 | 0.438 |
| Sentence shape | 20 | 0.844 | 0.316 |

## Unknown slang

`glorped` is not a dictionary candidate. `_unknown_tokens` logs it as `UNKNOWN_SLANG` with no probability and no meaning. The enriched text stays the original sentence, so toxicity analysis does not receive a substituted wording.

29 of 32 intended unknown or misspelled targets were logged. `saltee` and `buss` are entries in the system word list. `suss` is stemmed to `sus`, which is also in that list.

## Tests

95 backend tests passed, including 11 that load the encoder and the trained head. The frontend package has no automated test script. `npm run build` completed successfully.
