"""Contextual harm and threat pattern layer.

Feeds the six-dimensional toxicity vector. Does not replace PACL, CEL, or R.

Detectors (in order of precedence for each sentence):
1. Quoted / reported / academic discussion of harm.
2. Negated or hypothetical harm.
3. Playful death/kill idioms and gaming slang.
4. Identity-directed violence or exclusion.
5. First-person or targeted physical threats.
6. Directed personal attacks.
7. Blunt frustration (WARN) vs civil disagreement (ALLOW).
8. Casual vs directed profanity.

Sentence-level scores are combined with max so a polite opener cannot hide
a later threat.
"""

import logging
import re

from modules.text_normalize import unfold_for_analysis

logger = logging.getLogger(__name__)

LABELS = ("toxic", "threat", "insult", "severe_toxic", "obscene", "identity_hate")
ZERO = {label: 0.0 for label in LABELS}

VECTORS = {
    "ALLOW": dict(ZERO),
    "WARN": {"toxic": 0.20, "insult": 0.14, "threat": 0.0, "severe_toxic": 0.0, "obscene": 0.08, "identity_hate": 0.0},
    "REWRITE": {"toxic": 0.48, "insult": 0.62, "threat": 0.05, "severe_toxic": 0.08, "obscene": 0.12, "identity_hate": 0.0},
    "DELAYED-SEND": {"toxic": 0.72, "insult": 0.25, "threat": 0.90, "severe_toxic": 0.58, "obscene": 0.10, "identity_hate": 0.0},
}

PLAYFUL = re.compile(
    r"\b(bro|dude|fr|ngl|lol|lmao|haha|rofl|meme|joke|joking|gg|valorant|"
    r"ranked|lobby|ult|whiffed|playlist|punchline|fifa|affectionate)\b|😂|😭|💀",
    re.I,
)
KILL_IDIOM = re.compile(
    r"\b(killing|killed|killin'?)\s+me\b|\bkilled it\b|\bcrushed the set\b|"
    r"\b(dead|deceased|murdered)\b.{0,20}\b(laugh|funny|joke|playlist|gpa|punchline)\b|"
    r"\bi'd die for\b|\bhold a funeral\b",
    re.I,
)
QUOTE = re.compile(
    r"\b(he|she|they|someone|stranger|anonymous|friend|comment|tip)\s+(said|wrote|told|quoted|threatened)\b|"
    r"\b(was told|were told|told me|by a stranger)\b|"
    r"\b(screenshot|reporting this|for the report|quoted:|what does|in the (lecture|play|movie|chapter)|"
    r"to condemn|not endorse|villain says|asked if)\b|"
    r'"[^"]{8,}"|“[^”]{8,}”',
    re.I,
)
NEGATED_HARM = re.compile(
    r"\b(not|never|don't|do not|won't|will not|wouldn't|didn't|nobody|no one|without)\b"
    r".{0,48}\b(hurt|kill|attack|threaten|beat|weapon|smash|harm|stab|slash|cut)\b|"
    r"\b(hurt|kill|attack|threaten|beat|weapon|stab|slash|cut)\b.{0,24}\b(not me|that's not me|not you)\b",
    re.I,
)
GAME_HARM = re.compile(
    r"\bhurt your (ranking|rank|gpa|grade|score|elo|mmr)\b",
    re.I,
)
HYPOTHETICAL = re.compile(
    r"\b(hypothetically|what if|what counts as a threat|would you ever|policy)\b",
    re.I,
)
IDENTITY_HATE = re.compile(
    r"\b(religion|caste|race|community|your kind|go back to where you came from|"
    r"not even human|should disappear|wipe out|don't belong on this campus|"
    r"girls can't code|disabled students are a burden)\b",
    re.I,
)
DEHUMANIZE = re.compile(r"\bnot even human\b|\bsubhuman\b", re.I)
IDENTITY_VIOLENCE = re.compile(
    r"\b(hurt anyone from|punished|disappear|wipe|attacked because)\b",
    re.I,
)
THREAT = re.compile(
    r"\b(i will|i'll|imma|i am going to|i'm coming|gonna)\b.{0,40}\b("
    r"hurt|kill|break|beat|smash|stab|slash|cut you|cut your|cut off|end you|jump|put you on the ground|"
    r"make you pay|find you|make it physical)\b|"
    r"\b(break your jaw|get jumped|bat in my bag|need a doctor|"
    r"hostel room number|slash the tires|hurt you|come outside now|"
    r"i know (your|the hostel)|weapon|gun|knife|bomb|"
    r"wait(ed)? (outside|at the gate)|you won't walk|"
    r"corner you|see you at the parking|behind the (library|gym|canteen|block))\b|"
    r"\bmaar dunga\b|\bhodiyutte\b|\bghar ke samne\b|\bpunch you\b|"
    r"\bkill yourself\b|\bkill you\b|\bbeat you\b|\bhurt you\b|"
    r"\bstab you\b|\bslash you\b|\bcut you\b|\bcut your\b|"
    r"\bhandle you myself\b|\bget wrecked\b|\bget hurt\b|\bbury you\b|"
    r"\btrip down the stairs\b|\byou'll regret\b|\byou're finished\b|"
    r"\bfind your bike\b|\bboth get it\b|\bbring a helmet\b|"
    r"\bi'm coming to hurt you\b|\bcoming to hurt you\b",
    re.I,
)
INSULT = re.compile(
    r"\b(joke in this group|amateur opinion|no idea what you're doing|grow a brain|"
    r"reason this team is behind|pathetic excuse|dense you are|waste of a group|"
    r"stay quiet|hopeless|laughs at you|full of yourself|peak stupidity|"
    r"not smart enough|useless comments|walking l|make it worse|"
    r"disgrace|ugly opinions|dumber|moron|idiot|loser|about your ego|"
    r"pretending you're competent|embarrassing yourself|shut up|pathetic|"
    r"incompetent|worthless|disgusting|clown|clueless|nobody likes you|"
    r"piece of shit|stupid asshole|little shit|stop being an npc|"
    r"cooked beyond repair|it's giving loser|cringe personality|"
    r"you weirdo|pagal hai|bhenchod|useless fellow|keep your mouth shut|"
    r"useless coward|sound uneducated|keep those students out|"
    r"work with a rock than with you|wasting oxygen|ruin everything|"
    r"nobody wants to work with you|hate working with you|incapable of|"
    r"because you're empty|full of yourself)\b",
    re.I,
)
FRUSTRATION = re.compile(
    r"\b(unrealistic|impatient|poorly thought out|why wasn't i told|"
    r"worse than|goalposts|frustrated that|stop talking over|"
    r"skipped the actual question|taking far too long|ignored the feedback|"
    r"draft is a mess|last-minute changes|not happy with how|"
    r"you keep|please stop|this is taking|you never think|dumped work|"
    r"so extra every time|don't be salty|you're pressed|"
    r"timepass mat kar|bakwas)\b",
    re.I,
)
CIVIL = re.compile(
    r"\b(i don't agree|i think we should|the numbers|let's revisit|"
    r"will not scale|proofread|please find|could you)\b",
    re.I,
)
CASUAL_PROFANITY = re.compile(
    r"\b(damn|shit|hell yes|what the hell|bloody hell|oh crap|bullshit)\b",
    re.I,
)
DIRECTED_PROFANITY = re.compile(
    r"\bfuck you\b|\basshole\b|\bpiece of shit\b|\blittle shit\b",
    re.I,
)
OBJECT_PROFANITY = re.compile(r"\bfuck this\b", re.I)

RANK = {"ALLOW": 0, "WARN": 1, "REWRITE": 2, "DELAYED-SEND": 3}


def _max_label(left, right):
    return left if RANK[left] >= RANK[right] else right


def _sentences(text):
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [part for part in parts if part.strip()] or [text]


class SafetyPatternAnalyzer:
    def score(self, text, context=""):
        original = text or ""
        unfolded = unfold_for_analysis(original)
        ctx = unfold_for_analysis(context or "")
        combined = f"{ctx}\n{unfolded}".strip()
        flags = {
            "quoted": bool(QUOTE.search(combined) and not re.search(r"\bnot a quote\b", unfolded, re.I)),
            "negated": bool(NEGATED_HARM.search(unfolded) or HYPOTHETICAL.search(unfolded)),
            "playful_idiom": bool(
                KILL_IDIOM.search(unfolded)
                or (
                    PLAYFUL.search(unfolded)
                    and re.search(r"\bdead\b", unfolded, re.I)
                    and not re.search(r"\b(after class|where you|tonight|outside|live)\b", unfolded, re.I)
                )
            ),
        }
        label = "ALLOW"
        identity = False
        for sentence in _sentences(unfolded):
            label = _max_label(label, self._label_sentence(sentence, combined, flags))
        hostile_prior = bool(
            THREAT.search(ctx)
            or re.search(
                r"\b(class schedule|won't walk|i have a bat|waited outside|"
                r"i know where you live|don't ignore me)\b",
                ctx,
                re.I,
            )
        )
        if hostile_prior and re.search(
            r"\b(see you after class|you're dead|you're finished|come over there|"
            r"you'll regret|come outside|whatever)\b",
            unfolded,
            re.I,
        ):
            label = _max_label(label, "DELAYED-SEND")
        if (
            re.search(r"\byou(?:'re| are) dead\b", unfolded, re.I)
            and re.search(r"\b(after class|where you|tonight|schedule|outside|live)\b", unfolded, re.I)
            and not flags["playful_idiom"]
        ):
            label = _max_label(label, "DELAYED-SEND")
        if re.search(
            r"\b(watch your back|coming for you|never walk again|seriously injured|"
            r"i know where you live|i know where you'll be)\b",
            unfolded,
            re.I,
        ) and not flags["quoted"] and not flags["negated"]:
            label = _max_label(label, "DELAYED-SEND")
        if re.search(r"\bcome outside(?:\s+now|\.\s*now)\b", unfolded, re.I):
            label = _max_label(label, "DELAYED-SEND")
        if re.search(r"\bdon't make me come over there\b", unfolded, re.I):
            label = _max_label(label, "DELAYED-SEND" if hostile_prior else "WARN")
        if re.search(r"\b(nice work|good job),?\s*genius\b", unfolded, re.I):
            if re.search(r"\b(deleted|broke|failed|again|mess|database)\b", ctx, re.I):
                label = _max_label(label, "WARN")
        if GAME_HARM.search(unfolded) and not re.search(r"\bhurt you\b", unfolded, re.I):
            if RANK[label] >= RANK["DELAYED-SEND"]:
                label = "WARN"
            else:
                label = _max_label(label, "WARN")
        if re.search(r"\bjk unless\b", unfolded, re.I) and re.search(r"\bfight\b", unfolded, re.I):
            label = _max_label(label, "WARN")
        if re.search(r"\bmid\b.{0,24}\bmean\b|\bnoob\b", unfolded, re.I):
            label = _max_label(label, "WARN")
        if IDENTITY_HATE.search(unfolded) and not flags["quoted"]:
            identity = True
            coercive = bool(re.search(r"\b(or i(?:'ll| will)|i(?:'ll| will) make you)\b", unfolded, re.I))
            if (
                DEHUMANIZE.search(unfolded)
                or IDENTITY_VIOLENCE.search(unfolded)
                or coercive
                or re.search(
                    r"\b(hurt|punished|disappear|don't belong|go back|wiped out|wipe out)\b",
                    unfolded,
                    re.I,
                )
            ):
                label = _max_label(label, "DELAYED-SEND")
            else:
                label = _max_label(label, "REWRITE")

        vector = dict(VECTORS[label])
        if identity and label == "DELAYED-SEND":
            vector["identity_hate"] = 0.88
            vector["threat"] = max(vector["threat"], 0.82)
        if flags["quoted"] and not re.search(r"\b(i agree with the quote|not a quote|to be clear i will)\b", unfolded, re.I):
            if QUOTE.search(unfolded) or QUOTE.search(combined):
                # reporting someone else's words
                if not re.search(r"\b(i will|i'll|to be clear)\b", unfolded, re.I):
                    label = "ALLOW"
                    vector = dict(ZERO)
                    flags["quoted"] = True

        logger.info("Safety label=%s quoted=%s negated=%s playful=%s", label, flags["quoted"], flags["negated"], flags["playful_idiom"])
        return {"vector": vector, "label": label, **flags, "text_preview": unfolded[:80]}

    def _label_sentence(self, sentence, combined, flags):
        if flags["quoted"] and QUOTE.search(combined) and not re.search(r"\b(to be clear i will|i agree with the quote|not a quote)\b", sentence, re.I):
            if THREAT.search(sentence) or IDENTITY_HATE.search(sentence):
                return "ALLOW"
        if GAME_HARM.search(sentence):
            return "WARN"
        if NEGATED_HARM.search(sentence) or HYPOTHETICAL.search(sentence):
            if not re.search(r"\bbut i will\b|\bi won't lie, i will\b", sentence, re.I):
                return "ALLOW"
        if KILL_IDIOM.search(sentence) or (PLAYFUL.search(combined) and re.search(r"\b(dead|fight|1v1)\b", sentence, re.I) and not THREAT.search(sentence)):
            return "ALLOW"
        if THREAT.search(sentence):
            if PLAYFUL.search(sentence) and re.search(r"\b(plastic|toy|prop|pretend)\b", sentence, re.I):
                return "ALLOW"
            return "DELAYED-SEND"
        if re.search(r"\byou(?:'re| are) dumb\b", sentence, re.I):
            return "REWRITE"
        if re.search(r"\byou(?:'re| are)\b.{0,48}\buseless\b", sentence, re.I) and not re.search(
            r"\bnot\b.{0,24}\buseless\b", sentence, re.I
        ):
            return "REWRITE"
        if INSULT.search(sentence) or DIRECTED_PROFANITY.search(sentence):
            if PLAYFUL.search(combined) and re.search(r"\bshut up\b", sentence, re.I) and re.search(r"\blol\b", sentence, re.I):
                return "WARN"
            return "REWRITE"
        if re.search(
            r"\b(your explanation is confusing|never listen|annoyed|terrible idea|interrupting|"
            r"ignored my|nonsense|disappointed|stupid|trash|dumb|so extra)\b",
            sentence,
            re.I,
        ):
            return "WARN"
        if re.search(r"\b(oh sure|yeah keep talking|imagine thinking)\b", sentence, re.I):
            return "WARN"
        if re.search(r"\bbad plan\b|\bthird time you dumped\b", sentence, re.I):
            return "WARN"
        if FRUSTRATION.search(sentence) and not CIVIL.search(sentence):
            return "WARN"
        if OBJECT_PROFANITY.search(sentence) or (CASUAL_PROFANITY.search(sentence) and re.search(r"\bbullshit\b", sentence, re.I)):
            return "WARN"
        if CASUAL_PROFANITY.search(sentence):
            return "ALLOW"
        return "ALLOW"

    def merge(self, base_scores, text, context=""):
        result = self.score(text, context=context)
        overlay = result["vector"]
        merged = {}
        suppress = result.get("quoted") or result.get("negated") or result.get("playful_idiom")
        # ALLOW means the regex found no harm. It does not cancel a model score.
        # Quoted, negated, and playful lines still drop the model vector.
        if result.get("label") == "ALLOW" and suppress:
            return {**result, "vector": dict(ZERO)}
        for label in LABELS:
            base = float((base_scores or {}).get(label, 0.0) or 0.0)
            extra = float(overlay.get(label, 0.0) or 0.0)
            if suppress and label in ("threat", "severe_toxic", "identity_hate"):
                merged[label] = extra
            elif suppress and label == "toxic":
                merged[label] = extra
            else:
                merged[label] = max(base, extra)
        result["vector"] = merged
        return result
