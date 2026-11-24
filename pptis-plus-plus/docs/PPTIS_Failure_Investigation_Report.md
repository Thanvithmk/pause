# PPTIS++ failure investigation

Baseline test of the running application on 14 November 2026. No application code, model weights, or thresholds were changed.

The attached file was a PDF export of the Test Cases sheet only (`PPTIS_English_Test_Cases_and_Slang_Diagnostics-Test_Cases.pdf`). It contains 122 rows, A001 through X015. S014 is not in that sheet. The Slang & Unknown Diagnostics worksheet was not in the file, so it could not be transcribed. A separate 50-sentence diagnostic battery was executed live against the same API to answer the Unknown-label questions. Those 50 sentences are marked as an investigation battery, not as the missing worksheet.

Every Test Cases row was submitted to `POST /api/analyze` on the local server at port 5000. Health reported `unitary/toxic-bert` and the local slang head as actual inference. Each case used a new conversation. Context cases submitted the earlier turns first, in order, and the target message last. Nothing was sent to an external service except the rewrite model already configured for this app. Five local rows that the app itself classified as ALLOW were briefly stored as sent while the delay check ran, then set back to draft.

## What is working

- The API analyzes text, returns six toxicity scores, a final risk, and one of ALLOW, WARN, REWRITE, or DELAYED-SEND.
- Prior messages are stored and returned. X012 kept only the latest five turns and left the older threat out of `context_messages`.
- When the decision really is DELAYED-SEND, the message stays `delayed` and `POST /api/send` is rejected while the lock is active. Checked on I005, I006, R001, R002, R004, and R007.
- Many harmless lines are ALLOW: greetings, disagreement, academic critique, several counterspeech lines, and the Tamil-English line S015.
- Several explicit threats are DELAYED-SEND and held: R001, R002, R004, R007, I005, I006.
- In-vocabulary slang such as fire, mid, ngl, cooked, delulu, rizz, sus, and bussin is recognized with a meaning. That is separate from whether the sentence decision is right.

## Result of the 122 test cases

60 passed and 62 failed on the expected intervention. Pass rate 49.2%. No test was skipped.

| Outcome | Count | Meaning |
| --- | ---: | --- |
| Pass | 60 | Actual intervention matched the sheet |
| Over-escalation | 27 | A real insult or harsh line was pushed above WARN, usually to DELAYED-SEND |
| False negative | 22 | A harmful line was ALLOW |
| False positive | 12 | A harmless or protective line was warned, rewritten, or delayed |
| Under-escalation | 1 | I008 should have been rewritten and was only warned |

Direct personal abuse is the weakest area: 1 of 12 passed. Everyday neutral is the strongest: 11 of 14 passed. Threat wording that the safety list already contains is mostly held. Paraphrases, identity exclusion, sarcasm, and quoted reports are not.

## Why "This stupid printer is broken again" is delayed, and why those words are not Unknown

This sentence is test S010. Expected intervention: ALLOW. Actual: DELAYED-SEND. Final risk 0.8655. Toxic 0.986, insult 0.801, obscene 0.766, threat 0.001. The same result was repeated as diagnostics D01 and D50.

The slang array for that sentence was empty. The API did not label This, stupid, printer, broken, or again as Unknown. The current screen only lists `analysis.slang`, so those five words are not shown as Unknown chips on this build.

The Unknown label is produced by `SlangInterpreter._unknown_tokens` in `backend/modules/slang_layer.py`. A token is marked `UNKNOWN_SLANG` only when all of these are true:

- it is alphabetic and at least 4 characters, so "is" is skipped
- it is not already a slang-dictionary candidate
- it is not in `CHAT_ALLOW`
- it is not in the English word file, and a simple suffix stem is not either

The word file is `ENGLISH_WORDS_PATH`, default `/usr/share/dict/words`. On this Mac that file is the old Webster list. It contains stupid, printer, broken, again, and this. It does not contain women (it has woman), meme, laptop, ethnicity, internet, email, or iykyk. The stemmer does not turn woman into women.

That is why the live diagnostics labeled these Unknown, and did not label the printer words Unknown:

- laptop, recieve, tommorow, definately, wierd, Becuase, brocken, agian, iykyk
- glorped, zarnak, flibbered, qxzworth, which should be Unknown
- In the main sheet: Women (H002), ethnicity (H006), IJBOL, meme (G013)

The list you reported (This, stupid, printer, broken, again, and not "is") matches this function exactly if the word file fails to load. With an empty set, every alphabetic token of length 4 or more becomes Unknown, and "is" is skipped. That failure mode is in the code. It did not happen in this run, because the word file loaded and those five words are in it.

Unknown is not a toxicity class and not a meaning. `usage: unknown` only means "not in the slang dictionary and not in the word file." A recognized slang meaning is a different field. The sentence decision is a third field. "stupid" can be a known English word and still drive a bad DELAYED-SEND decision. Both things are true, and they come from different code.

The printer decision is the risk formula, not the word list.

`RiskCalculator.calculate_pacl` uses `PACL = clip(Tb + 0.5 * (Tb - B_T), 0, 1)`. A new user has baseline `B_T = 0`, so `PACL = clip(1.5 * Tb, 0, 1)`. Once toxic reaches about 0.67, PACL clips to 1. `P_esc = sigmoid(0 + 1 * PACL)` is then `sigmoid(1) ≈ 0.731`. With alpha 0.5, final risk is always 0.8655. `get_risk_category` maps anything at or above 0.80 to DELAYED-SEND.

So every strongly toxic insult, including "stupid" aimed at a printer, a self-aimed "I'm such an idiot" (S009), and "You're worthless" (I010), receives the same risk and the same action as a direct threat. The threat score is not what selects the action. `run_analysis_pipeline` in `backend/api/routes.py` builds that risk from `toxicity_detail["model"]` only. The safety layer's merged vector is discarded. `DecisionEngine._apply_safety_label` can raise an action and cannot lower one, except for the separate emoji and contextual-threat lowerings. The bare word "stupid" does make the safety label WARN (`safety_patterns.py` around the frustration/insult word list), and that WARN is ignored because the model decision is already higher.

## Failure groups

### 1. Insults are delayed instead of warned or rewritten

27 cases. Examples: A001, A008, T001–T005, T007, T010, I001, I002, I004, I010, M001, M005, E003, G004, G006, G011, S009, S010.

Evidence: toxic from about 0.67 to 0.99, threat usually near 0, guardrail empty, mathematical decision DELAYED-SEND, R exactly 0.8655 whenever PACL has clipped.

I001 and I002 were supposed to produce a rewrite. The rewrite engine is only called when the decision is REWRITE (`_rewrite_payload`). These two never got that far, so there is no suggested text. I008, "You're so dumb 😂", did reach a high risk and was then lowered to WARN by the laughter-emoji rule, so it also never rewrote.

Impact: ordinary insults and self-talk get the same hold as a threat. Constructive rewrite, which the sheet expects for harsh feedback, does not run.

### 2. Harmful lines are allowed

22 cases. The model score is near zero and no safety pattern matches.

Identity exclusion the regex does not list: H001, H003, H004, H005, H007, H008. `IDENTITY_TOPIC` is only community, religion, caste, or race. `IDENTITY_HATE` is a handful of exact phrases such as "go back to where you came from" and "disabled students are a burden". "People with disabilities are a burden and should stay away" does not match. H001 includes the word religion, but the hostile-pattern branch still did not fire for "should not be allowed in our college", and the model toxic score was 0.002.

Obfuscated exclusion: A006, "g0 b@ck", toxic 0.001. `unfold_for_analysis` turns that into "go back", but only inside the safety layer, and the exact hate phrase still does not match.

Doxxing and exposure: R003 and I009, toxic about 0.002. The threat expression has no pattern for posting a phone number, address, or private information. Because the decision was ALLOW, send was accepted. The message was not held.

Coded or contextual threats: G010 ("cooked when I catch you"), M003 ("you'll find out what happens"), X005 ("See you after class" after "I know when you leave campus" and "Just wait"). Priors for X005 were present in `context_messages`. `attribute_current_scores` in `toxicity_model.py` keeps the current sentence's own score when it is lower than the gain from the window, so a calm follow-up does not inherit the earlier threat. The safety `hostile_prior` list looks for phrases such as "i know where you live" and does not match "I know when you leave campus".

Sarcasm with polite words: C001, C003, M007, T012, X004. Toxic scores about 0.001 to 0.004. The sarcasm regex only matches "nice work, genius" or "good job, genius".

Other misses: G007 ("You're such an NPC") is recognized as slang but the insult regex only has "stop being an npc". E007 emoji-only exclusion after a hostile prior was ALLOW for the same context-attribution reason.

### 3. Harmless or protective lines are intercepted

12 cases.

S010 and S009 are the saturation false positives above. S005, "Please stop messaging me", matches `FRUSTRATION` on "please stop" and is raised to WARN even though toxic is 0.001. N002, counterspeech that says "Please stop targeting them", is raised the same way.

N001 uses single quotes and "Someone sent me", which `QUOTE` does not match. The quoted insult was scored as the user's own speech (toxic 0.832) and delayed. N008, "I received the message 'I'll hurt you tomorrow'", also missed the quote detector. The threat regex then matched the reported threat and the safety label raised the action to DELAYED-SEND. N007 did match `villain says` (`safety.quoted` true, safety label ALLOW), but the pipeline never applies that suppression to the risk score, and the quoted flag also makes `_apply_safety_label` return without changing the model decision. Result: WARN from the raw model.

E001, emoji-only 💀😭, toxic 0.14, risk 0.38, WARN. The zero baseline turns a small score into a warning. G013, "IJBOL this meme is too funny", is the same edge plus Unknown labels for two words that are missing from both the slang file and the Webster list. X002, friendly "No way, that's insane", landed in REWRITE because toxic 0.338 sits in that band after amplification. R006, fictional "I will destroy the kingdom", was REWRITE rather than ALLOW. G012 was REWRITE rather than WARN, and the suggestion was "I'm sorry, but I can't help with that", which is a refusal, not a moderated message.

### 4. Context is loaded and then mostly ignored

This is confirmed, not only inferred. For every X case, `context_messages` contained the submitted turns in order. X012's returned window was the five newest turns and did not include the older threat. X013 returned all five recent turns.

The decisions still failed where the current sentence is mild and the harm is in the earlier turns (X004, X005, X008, X010). The cause is `attribute_current_scores`, plus safety patterns that only inherit a prior when the prior text matches a short expression list.

### 5. Word recognition versus meaning versus toxicity

These three outputs are different, and the tests show them diverging.

- Known Webster words are omitted from the slang list. They are not labeled Unknown in this run.
- Dictionary slang gets `usage` slang or literal and a meaning. "touch grass" was literal. "ate" in "She ate and left no crumbs" was literal. "sick" was slang in "That's sick, I love it" and uncertain in "I feel sick today". "dead" was literal for a beat and slang for "You're dead to me".
- Tokens missing from Webster and from the slang dictionary, of length 4 or more, become UNKNOWN_SLANG. That includes normal modern words (laptop, women, meme), misspellings, and real inventions (glorped).
- "kys" and "slay/slayed" were not labeled at all. "kys" has length 3, so `_unknown_tokens` skips it, and it is not a dictionary candidate. "slayed" does not match the stored term "slay".
- None of that word label chooses ALLOW or DELAYED-SEND. The printer sentence has no Unknown labels and is still delayed.

## Fixes to make, in order

No threshold was changed in this test. The recommendations below are the specific code changes the failures support. They do not require retraining.

### F1. Do not let a high toxic score mean DELAYED-SEND

Change `DecisionEngine.make_decision` in `backend/modules/decision_engine.py`, and stop feeding only the raw toxic dimension into the action in `run_analysis_pipeline`.

DELAYED-SEND should require a threat, a severe targeted attack, identity harm, or an explicit safety label for those classes. An insult with a low threat score should stay WARN or REWRITE. Self-directed and object-directed lines should not enter DELAYED-SEND just because the word stupid or idiot scores near 1.

Why: 27 over-escalations, including both rewrite cases and the printer sentence, share one clipped risk of 0.8655. Lowering the 0.80 constant would also move real threats, because those threats often ride the same toxic score. The split has to be by dimension.

Effect elsewhere: insults that are currently held for 10 seconds would warn or rewrite instead. Explicit threats that match `THREAT` or the threat guardrail stay delayed. Rerun A001, A008, E003, G004, G006, G011, I001, I002, I004, I010, M001, M005, S009, S010, T001–T005, T007, T010, and the three expected rewrite cases.

### F2. Let a safety label lower a non-personal hit, and do not treat a bare insult word as a person attack

In `backend/modules/safety_patterns.py`, the word list that returns WARN for stupid, dumb, or trash fires with no target. In `decision_engine.py`, `_apply_safety_label` cannot lower the model action.

Add a target check. If the sentence has no addressee and the object is a thing or the speaker, cap the action at ALLOW for a complaint ("stupid printer", "this exam") and at WARN for harsh self-talk only if you still want a nudge. Then actually use that cap.

Why: S009 and S010. The safety layer already wanted a lower action and was ignored.

Effect elsewhere: personal "you are stupid" must still warn or rewrite under F1. Rerun S004, S007, S009, S010, D01, and T005.

### F3. Cover the paraphrases the model scores as zero

Extend the patterns in `safety_patterns.py`, still as patterns, not a new model.

- Identity: "should not be allowed", "go back where they came from", "people with disabilities are a burden", "from that caste", "those people ... don't belong".
- Exposure: phone number, home address, private information.
- Coded threat: "find out what happens", "cooked when I catch you", "see you after class" after a prior that names a place or a wait.
- Insult slang already recognized by the dictionary, including "you're such an NPC".
- Remove "please stop" from `FRUSTRATION` when the sentence is a boundary or a rejection of abuse.

Run `unfold_for_analysis` on the text that Toxic-BERT sees as well, or at least let the unfolded safety label outrank a near-zero model score. Today the unfold never reaches the model, and a near-zero model score plus a missed pattern is ALLOW.

Why: the 22 false negatives. H001 toxic was 0.002 with safety label ALLOW.

Effect elsewhere: broader patterns can false-positive. Keep them tied to a group word or a threat verb, and keep the quote suppression from F4 in front of them. Rerun H001–H008, A002–A006, R003, I009, G007, G010, M003, S005, N002.

### F4. Honor quotes

`QUOTE` in `safety_patterns.py` should accept single quotes and reporting verbs such as "sent me", "received the message", and "I want to report". In `run_analysis_pipeline`, when `safety.quoted` is true and the safety label is ALLOW, use the suppressed vector from `toxicity_detail["vector"]` for the risk score. Do not return early from `_apply_safety_label` in a way that leaves the raw model score in charge.

Why: N007 proves the flag is set and then discarded. N001 and N008 prove the flag is not set for the report wording in the sheet.

Effect elsewhere: a quote that the user endorses ("I agree with that") must not be suppressed. X010 is the endorsement case and should stay a warning under F5. Rerun N001–N008 and R006.

### F5. Use the context that is already loaded

`attribute_current_scores` should not always prefer the calmer current sentence when that sentence is a continuation: pronouns ("they"), "just leave", "see you after class", "I agree with that", sarcastic praise after a mistake. The safety prior list needs the phrases the sheet actually uses, not only "i know where you live".

Why: `context_messages` is correct and the decision still ignores it. X005 is the clearest case.

Effect elsewhere: X014 must stay ALLOW. A neutral "Can you review my code?" after insults should not inherit them. The continuation rule has to require a pronoun, an agreement, or a threat-shaped fragment, not any sentence in a toxic thread. Rerun X001–X015, E007, and M010.

### F6. Do not turn every laughter emoji into a warning downgrade

`_apply_playful_emoji` lowers both REWRITE and DELAYED-SEND to WARN whenever a laughter emoji is present. That saved E006 and T008, and it blocked the rewrite on I008.

Lower only when the text has no insult and no threat. A laughter emoji after "you're so stupid" or "you're so dumb" should not change the action.

Rerun E001, E002, E006, T008, I008, G013.

### F7. Unknown should mean "not English and not known slang"

In `slang_layer.py`:

- Treat a token as UNKNOWN_SLANG only when it is not English and not a known chat abbreviation. A missing Webster entry is not enough. women, meme, laptop, and ethnicity are English.
- Stem irregular plurals, or ship a word list that contains them. Do not rely on `/usr/share/dict/words` alone. If that file is missing, fail or log it. An empty set is what labels every word of four or more letters Unknown.
- Match dictionary terms by stem so "slayed" can find "slay".
- Keep length-3 harmful abbreviations on a safety list ("kys") instead of hoping the unknown-token scan will see them. That scan starts at length 4.
- Leave genuine inventions (glorped, qxzworth) as Unknown. Do not add them to the dictionary just to clear a test.

This does not fix the printer decision. F1 and F2 do that. F7 fixes the label.

Rerun D01, D09, D21, D27, D31, D34, D41–D43, D48, D49, H002, H006, G013, and the printer sentence.

### F8. Rewrite the message, do not refuse it

When a decision is REWRITE, `rewrite_engine.py` should return a sendable softer version. G012 came back as "I can't help with that." I001 and I002 never called the rewriter because F1 had already chosen DELAYED-SEND.

Rerun I001, I002, I008, G012 after F1.

## What not to do

- Do not retrain Toxic-BERT for these failures. The misses on identity paraphrases and the hits on "stupid" are both real model behavior, and the decision layer is what turns that behavior into the wrong action.
- Do not lower 0.30 / 0.55 / 0.80 by itself. The clipped PACL value makes those cutoffs describe almost the same messages.
- Do not add printer, stupid, broken, and again to the slang dictionary. They are ordinary words, and on this run they were not Unknown.

## Regression set after any of the fixes

A001, A006, C001, E001, E006, G007, G010, G012, G013, H001, H004, I001, I005, I008, I009, N001, N002, N007, N008, R001, R003, R006, S005, S009, S010, T001, X005, X012, X014, D01, D09, D48.
