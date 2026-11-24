# PPTIS++ test performance summary

Live run against the local application on 14 November 2026. Decisions were not mocked. Application code was not changed. Skipped tests are not counted as passes.

The source file was a PDF of the Test Cases sheet, 122 rows. S014 is not on that sheet. The Slang & Unknown Diagnostics worksheet was not in the attachment. The 50 diagnostic rows below were executed for this investigation and scored with an explicit word-recognition rule. They are not the missing worksheet's expected column.

## Test Cases sheet

| Measure | Count |
| --- | ---: |
| Total test cases in the sheet | 122 |
| Successfully executed | 122 |
| Passed | 60 |
| Failed | 62 |
| Not executed | 0 |
| Pass rate | 49.2% |

A003 was executed as the text the PDF still contained, "You are stu". The invisible-character tail was not in the text layer. That row failed. A separate probe of the full word, with and without a zero-width character, also returned DELAYED-SEND.

| Failure type | Count |
| --- | ---: |
| Over-escalation (action stricter than expected) | 27 |
| False negative (harmful message allowed) | 22 |
| False positive (harmless or protective message intercepted) | 12 |
| Under-escalation (I008 warned instead of rewritten) | 1 |

## By test area

| Area | Cases | Passed | Failed | Pass rate |
| --- | ---: | ---: | ---: | ---: |
| Evasion / obfuscation | 8 | 2 | 6 | 25.0% |
| Sarcasm and irony | 4 | 2 | 2 | 50.0% |
| Emoji interpretation | 8 | 5 | 3 | 62.5% |
| Gen Z slang | 15 | 8 | 7 | 53.3% |
| Identity-based hostility | 10 | 4 | 6 | 40.0% |
| Intervention behavior | 10 | 4 | 6 | 40.0% |
| English language coverage | 10 | 5 | 5 | 50.0% |
| Quotation / counterspeech | 8 | 3 | 5 | 37.5% |
| Threats and intimidation | 8 | 6 | 2 | 75.0% |
| Everyday neutral | 14 | 11 | 3 | 78.6% |
| Direct personal abuse | 12 | 1 | 11 | 8.3% |
| Context window (1–5) | 15 | 9 | 6 | 60.0% |

Intervention checks inside those totals:

- Rewrite was required on I001, I002, and I008. None of the three returned a rewritten message. I001 and I002 were DELAYED-SEND, so the rewriter was not called. I008 was lowered to WARN by a laughter emoji.
- DELAYED-SEND, when that was the actual decision, held the message. Send was rejected and the stored status stayed `delayed` on I005, I006, R001, R002, R004, and R007.
- Context turns were returned for every context case. X012's five-message window excluded the older threat. Six context decisions were still wrong because the score did not use those turns.

## Word recognition and slang diagnostics

50 live sentences. The rule was: an ordinary English word must not be Unknown, a known slang term must be labeled, a misspelling must not be dumped into UNKNOWN_SLANG, and an invented token must be Unknown.

| Measure | Count |
| --- | ---: |
| Executed | 50 |
| Passed the word-recognition rule | 43 |
| Failed the word-recognition rule | 7 |
| Not executed | 0 |
| Pass rate | 86.0% |

The 7 failures are D09 laptop, D27 slayed, D31 iykyk, D34 kys, and the three misspelling sentences D41–D43.

The printer sentence was not one of those 7. This, stupid, printer, broken, and again were not labeled Unknown. The sentence decision was still DELAYED-SEND. Word recognition and the moderation decision are different results.

Invented tokens were distinguished correctly: glorped, zarnak, flibbered, and qxzworth were Unknown, and the ordinary words around them were not.

## Where the row-level results are

`docs/PPTIS_English_Test_Results.xlsx`

- Sheet `Test Cases`: all 122 rows with the actual decision, both risk numbers, six toxicity scores, word labels, rewrite text, delay check, context returned, PASS/FAIL, observation, root cause, and the fix to rerun.
- Sheet `Slang and Unknown Diagnostics`: the 50 investigation sentences.

The narrative of causes and the recommended code changes are in `docs/PPTIS_Failure_Investigation_Report.md`.
