"""Contextual reading of physical-action and figurative threat language.

This component does not score toxicity and does not change Toxic-BERT
probabilities, PACL, or final risk. It only describes whether a threat
reading fits the action, the object, and the target.

effect "clear" means the threat reading does not fit. The decision layer
may lower its own intervention to the returned floor. effect "hold" means
this component does not lower anything.
"""

import re

WORD = re.compile(r"[a-z0-9']+")
AMBIGUOUS = re.compile(
    r"\b(?:deal with you|regret this|finish you|make you pay|pay for this|end you|cut you)\b",
    re.I,
)
KILLED_ME = re.compile(
    r"\b(?:kill(?:ing|ed)?|destroy(?:ing|ed)?)\s+me\b",
    re.I,
)
DEAD_STATE = re.compile(r"\bdead\s+(?:tired|asleep|hungry|beat|sleepy)\b", re.I)
KILL_TASK = re.compile(
    r"\b(?:kill(?:ing|ed)?|destroy(?:ing|ed)?)\s+(?:this|the|my|our|that|a)\s+([a-z]+)",
    re.I,
)

HARM_VERBS = {
    "cut", "cuts", "cutting", "slash", "slashes", "slashing", "slashed",
    "stab", "stabs", "stabbing", "stabbed", "chop", "chops", "chopping",
    "chopped", "slice", "slices", "slicing", "sliced",
}
VIOLENT_VERBS = {
    "hurt", "hurts", "hurting", "stab", "stabs", "stabbing", "stabbed",
    "kill", "kills", "killing", "killed", "attack", "attacks", "attacking",
    "attacked", "slash", "slashes", "slashing", "slashed", "beat", "beats",
    "beating", "murder", "murders", "murdering", "murdered", "shoot", "shoots",
    "shooting", "shot", "punch", "punches", "punching", "punched", "strangle",
    "strangles", "strangling", "strangled",
}
PERSON = {"you", "ya", "u", "him", "her", "them", "yourself", "himself", "herself"}
VIOLENT_OBJECTS = {
    "head", "throat", "neck", "face", "skull", "brain", "eye", "eyes", "heart",
    "chest", "stomach", "gut", "guts", "artery", "vein", "wrist", "tongue",
    "ear", "ears", "finger", "fingers", "hand", "hands", "arm", "arms", "leg",
    "legs", "body", "skin", "knee", "knees",
}
GROOM_VERBS = {
    "cut", "cuts", "cutting", "trim", "trims", "trimming", "trimmed", "clip",
    "clips", "clipping", "clipped", "shave", "shaves", "shaving", "shaved",
    "snip", "snips", "snipping",
}
GROOM_OBJECTS = {
    "nail", "nails", "fingernail", "fingernails", "toenail", "toenails", "hair",
    "beard", "mustache", "moustache", "stubble",
}
CHORE_OBJECTS = {
    "vegetable", "vegetables", "veggie", "veggies", "paper", "fabric", "cloth",
    "rope", "string", "thread", "ribbon",
}
BENIGN_VERBS = {
    "cut", "cuts", "cutting", "slice", "slices", "slicing", "sliced",
    "chop", "chops", "chopping", "chopped", "peel", "peels", "peeling", "peeled",
    "dice", "dices", "dicing", "mince", "minces", "mincing", "core", "cores",
    "carve", "carves", "carving", "quarter", "halve", "halves",
    "open", "opens", "opening", "opened", "spread", "spreads", "spreading",
}
PLACEMENT_VERBS = {
    "insert", "inserts", "inserting", "inserted", "put", "puts", "putting",
    "place", "places", "placing", "placed", "stick", "sticks", "sticking", "stuck",
    "push", "pushes", "pushing", "pushed",
}
TOOLS = {"knife", "knives", "peeler", "scissors", "cleaver", "blade", "blades"}
CONTENT_SKIP = {
    "a", "an", "the", "your", "my", "his", "her", "their", "our", "its",
    "this", "that", "these", "those", "some", "any", "with", "using", "into",
    "to", "for", "and", "then", "gently", "please", "just", "really", "very",
    "on", "in", "of", "from", "up", "it", "one", "me",
}
SERIOUS_THREAT = re.compile(
    r"\b(hurt|kill|beat|stab|shoot|attack|slash|strangle|murder|punch)\b.{0,24}\b(you|him|her|them|u)\b|"
    r"\b(you|him|her|them)\b.{0,16}\b(hurt|killed|stabbed|shot|attacked)\b|"
    r"\b(gun|guns|bomb|bombs|weapon|weapons|break your jaw|slash the tires|"
    r"watch your back|i know where you|coming for you)\b",
    re.I,
)
CARE_VERBS = {
    "clean", "cleans", "cleaning", "cleaned", "wash", "washes", "washing",
    "washed", "apply", "applies", "applying", "applied", "treat", "treats",
    "treating", "treated", "disinfect", "disinfects",
}
CARE_OBJECTS = {
    "nail", "nails", "hair", "wound", "wounds", "medicine", "ointment",
    "bandage", "bandages",
}
MEDICAL_VERBS = {"give", "gives", "giving", "inject", "injects", "injecting", "injected",
                 "administer", "administers", "administering", "administered"}
MEDICAL_OBJECTS = {"injection", "vaccine", "vaccination", "medicine", "ointment"}
FIGURATIVE_OBJECTS = {
    "presentation", "exam", "test", "quiz", "joke", "song", "performance",
    "deadline", "workout", "assignment", "meeting", "interview", "project",
    "playlist", "game", "show", "class", "practice",
}
WINDOW = 5


def assess(text):
    original = "" if text is None else str(text)
    tokens = WORD.findall(original.lower())
    explicit = _explicit(tokens)
    if explicit:
        return _result("explicit_threat", "hold", "DELAYED-SEND", explicit)
    if AMBIGUOUS.search(original):
        return _result("ambiguous", "hold", "ALLOW", {"action": None, "object": None})
    harmless = _harmless(tokens)
    if harmless:
        return _result("harmless_physical", "clear", _residual_floor(original), harmless)
    everyday = _everyday_action(tokens, original)
    if everyday:
        return _result("harmless_physical", "clear", _residual_floor(original), everyday)
    if _figurative(original):
        return _result("figurative", "clear", _residual_floor(original), {"action": None, "object": None})
    return _result("none", "hold", "ALLOW", {"action": None, "object": None})


def _explicit(tokens):
    for index, token in enumerate(tokens):
        if token in HARM_VERBS:
            obj = _object_after(tokens, index, VIOLENT_OBJECTS)
            if obj:
                return {"action": token, "object": obj}
        if token in VIOLENT_VERBS:
            target = _object_after(tokens, index, PERSON)
            if target:
                return {"action": token, "object": target}
    return None


def _harmless(tokens):
    pairs = (
        (GROOM_VERBS, GROOM_OBJECTS | CHORE_OBJECTS),
        (CARE_VERBS, CARE_OBJECTS),
        (MEDICAL_VERBS, MEDICAL_OBJECTS),
    )
    for index, token in enumerate(tokens):
        for verbs, objects in pairs:
            if token not in verbs:
                continue
            obj = _object_after(tokens, index, objects)
            if obj:
                return {"action": token, "object": obj}
    return None


def _figurative(text):
    if KILLED_ME.search(text) or DEAD_STATE.search(text):
        return True
    match = KILL_TASK.search(text)
    return bool(match and match.group(1).lower() in FIGURATIVE_OBJECTS)


def _aimed_at_person_or_body(tokens):
    """A later person or body part is the target of a harm or placement verb.

    "for you" is a recipient, as in "cut an apple for you", so that one token
    is skipped. "your knife" is a tool, because "your" is not a person token.
    """
    watched = HARM_VERBS | VIOLENT_VERBS | PLACEMENT_VERBS
    for index, token in enumerate(tokens):
        if token not in watched:
            continue
        skip_next = False
        for later in tokens[index + 1:]:
            if skip_next:
                skip_next = False
                continue
            if later == "for":
                skip_next = True
                continue
            if later in PERSON or later in VIOLENT_OBJECTS:
                return True
    return False


def _content_after(tokens, index):
    for token in tokens[index + 1:]:
        if token in CONTENT_SKIP or token in TOOLS or token in BENIGN_VERBS or token in PLACEMENT_VERBS:
            continue
        if token in PERSON or token in VIOLENT_OBJECTS:
            return None
        if len(token) < 3:
            continue
        return token
    return None


def _everyday_action(tokens, original):
    """A tool or a cut is ordinary when it is not aimed at a person or a body.

    The object can be any thing. Food, cardboard, a package, and a rope follow
    the same rule. A gun, a bomb, or an attack on a person does not.
    """
    if SERIOUS_THREAT.search(original) or _aimed_at_person_or_body(tokens):
        return None
    violent_only = VIOLENT_VERBS - GROOM_VERBS - BENIGN_VERBS
    if any(token in violent_only for token in tokens):
        return None
    for index, token in enumerate(tokens):
        if token not in BENIGN_VERBS and token not in GROOM_VERBS and token not in PLACEMENT_VERBS:
            continue
        obj = _content_after(tokens, index)
        if obj:
            return {"action": token, "object": obj}
    if any(token in TOOLS for token in tokens):
        tool = next(token for token in tokens if token in TOOLS)
        return {"action": tool, "object": tool}
    return None


def _object_after(tokens, index, allowed):
    for token in tokens[index + 1:index + 1 + WINDOW]:
        if token in allowed:
            return token
    return None


def _residual_floor(text):
    """Lowest intervention that other harm patterns still support."""
    from modules.safety_patterns import DIRECTED_PROFANITY, IDENTITY_HATE, INSULT

    if IDENTITY_HATE.search(text):
        return "DELAYED-SEND"
    if INSULT.search(text) or DIRECTED_PROFANITY.search(text):
        return "REWRITE"
    return "ALLOW"


def _result(category, effect, floor, slots):
    return {
        "category": category,
        "effect": effect,
        "floor": floor,
        "action": slots.get("action"),
        "object": slots.get("object"),
    }
