"""One local profile for Pause. There is no login and no token."""

from modules.database import User, close_session, get_session

LOCAL_USERNAME = "pause"
LOCAL_EMAIL = "pause@local"


def get_local_user():
    session = get_session()
    try:
        user = session.query(User).filter(User.username == LOCAL_USERNAME).one_or_none()
        if user is None:
            user = User.create(session, LOCAL_USERNAME, LOCAL_EMAIL, "local-only")
        payload = user.to_dict()
        user_id = user.user_id
        return user_id, payload
    finally:
        close_session()
