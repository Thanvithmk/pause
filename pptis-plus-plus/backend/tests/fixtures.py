from datetime import datetime, timedelta, timezone

from sqlalchemy import or_

from modules.database import (
    Conversation,
    Message,
    SlangDictionary,
    ToxicityResult,
    User,
    close_session,
    get_session,
    init_db,
)

TEST_PASSWORD = "TestPass1!"

TEST_USERS = (
    {
        "username": "alexsafe",
        "email": "alex@example.com",
        "password": TEST_PASSWORD,
    },
    {
        "username": "jordanrisk",
        "email": "jordan@example.com",
        "password": TEST_PASSWORD,
    },
)

TEST_SLANG_TERMS = (
    {"slang_term": "cooked", "meaning": "defeated/finished/in serious trouble", "category": "gen-z", "confidence": 0.95},
    {"slang_term": "rizz", "meaning": "charisma or romantic skill", "category": "gen-z", "confidence": 0.9},
)

SAFE_TEXT = "Thanks for waiting. I appreciate your help."
TOXIC_TEXT = "You are stupid and I hate you."
SLANG_TEXT = "Bro you're cooked"

SAFE_VECTOR = [0.05, 0.02, 0.03, 0.01, 0.02, 0.01]
TOXIC_VECTOR = [0.92, 0.40, 0.88, 0.35, 0.55, 0.20]
PREVIOUS_VECTOR = [0.30, 0.20, 0.20, 0.05, 0.10, 0.02]
CURRENT_VECTOR = [0.70, 0.60, 0.50, 0.10, 0.20, 0.05]
DELTA_VECTOR = [0.40, 0.40, 0.30, 0.05, 0.10, 0.03]

TEST_CONVERSATIONS = ({"title": "primary"},)
TEST_MESSAGES = (
    {"original_text": "Hey, can we talk later?", "input_type": "text"},
    {"original_text": SLANG_TEXT, "input_type": "text"},
    {"original_text": TOXIC_TEXT, "input_type": "text"},
)


def init_test_db():
    close_session()
    init_db("sqlite:///:memory:", create_tables=True)
    return get_session()


def seed_slang(session):
    rows = []
    for item in TEST_SLANG_TERMS:
        rows.append(SlangDictionary.create(session, **item))
    return rows


def seed_user(session, index=0):
    spec = TEST_USERS[index]
    existing = (
        session.query(User)
        .filter(or_(User.username == spec["username"], User.email == spec["email"]))
        .one_or_none()
    )
    if existing is not None:
        return existing
    return User.create(
        session,
        username=spec["username"],
        email=spec["email"],
        password=spec["password"],
    )


def seed_conversation(session, user):
    existing = (
        session.query(Conversation)
        .filter(Conversation.user_id == user.user_id)
        .order_by(Conversation.conversation_id.asc())
        .first()
    )
    if existing is not None:
        return existing
    return Conversation.create(session, user_id=user.user_id)


def seed_message(session, conversation, user, spec, minutes_ago=0):
    message = Message(
        conversation_id=conversation.conversation_id,
        user_id=user.user_id,
        original_text=spec["original_text"],
        processed_text=spec["original_text"],
        input_type=spec.get("input_type", "text"),
    )
    if minutes_ago:
        message.timestamp = datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)
    session.add(message)
    session.commit()
    session.refresh(message)
    return message


def seed_toxicity(session, message, vector):
    return ToxicityResult.create(
        session,
        message_id=message.message_id,
        toxic=vector[0],
        threat=vector[1],
        insult=vector[2],
        severe_toxic=vector[3],
        obscene=vector[4],
        identity_hate=vector[5],
    )


def seed_core_dataset(session):
    slang = seed_slang(session)
    user = seed_user(session)
    conversation = seed_conversation(session, user)
    messages = []
    for offset, spec in enumerate(TEST_MESSAGES):
        messages.append(seed_message(session, conversation, user, spec, minutes_ago=len(TEST_MESSAGES) - offset))
    seed_toxicity(session, messages[0], SAFE_VECTOR)
    seed_toxicity(session, messages[1], CURRENT_VECTOR)
    seed_toxicity(session, messages[2], TOXIC_VECTOR)
    return {
        "user": user,
        "conversation": conversation,
        "messages": messages,
        "slang": slang,
    }
