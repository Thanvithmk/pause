-- PPTIS++ PostgreSQL schema

CREATE TYPE input_type_enum AS ENUM ('text', 'voice');
CREATE TYPE decision_action_enum AS ENUM ('ALLOW', 'WARN', 'REWRITE', 'DELAYED-SEND');
CREATE TYPE user_response_enum AS ENUM ('accepted', 'rejected', 'edited');
CREATE TYPE outcome_enum AS ENUM ('positive', 'neutral', 'negative');
CREATE TYPE review_status_enum AS ENUM ('pending', 'reviewed', 'approved', 'rejected');

CREATE TABLE users (
    user_id SERIAL PRIMARY KEY,
    username VARCHAR(150) NOT NULL UNIQUE,
    email VARCHAR(255) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE conversations (
    conversation_id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE messages (
    message_id SERIAL PRIMARY KEY,
    conversation_id INTEGER NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
    user_id INTEGER NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    original_text TEXT NOT NULL,
    processed_text TEXT,
    input_type input_type_enum NOT NULL,
    transcription_confidence DOUBLE PRECISION,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE toxicity_results (
    result_id SERIAL PRIMARY KEY,
    message_id INTEGER NOT NULL REFERENCES messages(message_id) ON DELETE CASCADE,
    toxic DOUBLE PRECISION NOT NULL,
    threat DOUBLE PRECISION NOT NULL,
    insult DOUBLE PRECISION NOT NULL,
    severe_toxic DOUBLE PRECISION NOT NULL,
    obscene DOUBLE PRECISION NOT NULL,
    identity_hate DOUBLE PRECISION NOT NULL
);

CREATE TABLE behavior_profiles (
    profile_id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL UNIQUE REFERENCES users(user_id) ON DELETE CASCADE,
    avg_toxic DOUBLE PRECISION NOT NULL DEFAULT 0,
    avg_threat DOUBLE PRECISION NOT NULL DEFAULT 0,
    avg_insult DOUBLE PRECISION NOT NULL DEFAULT 0,
    avg_severe_toxic DOUBLE PRECISION NOT NULL DEFAULT 0,
    avg_obscene DOUBLE PRECISION NOT NULL DEFAULT 0,
    avg_identity_hate DOUBLE PRECISION NOT NULL DEFAULT 0,
    message_count INTEGER NOT NULL DEFAULT 0,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE risk_results (
    risk_id SERIAL PRIMARY KEY,
    message_id INTEGER NOT NULL REFERENCES messages(message_id) ON DELETE CASCADE,
    pacl DOUBLE PRECISION NOT NULL,
    delta_vector_6d JSONB NOT NULL,
    escalation_feature DOUBLE PRECISION NOT NULL,
    escalation_probability DOUBLE PRECISION NOT NULL,
    final_risk DOUBLE PRECISION NOT NULL,
    decision decision_action_enum NOT NULL,
    trigger_words JSONB,
    trigger_category VARCHAR(64),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE intervention_logs (
    id SERIAL PRIMARY KEY,
    message_id INTEGER NOT NULL REFERENCES messages(message_id) ON DELETE CASCADE,
    action decision_action_enum NOT NULL,
    suggested_text TEXT,
    context_ids JSONB,
    user_response user_response_enum,
    delay_seconds INTEGER,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE slang_dictionary (
    slang_id SERIAL PRIMARY KEY,
    slang_term VARCHAR(255) NOT NULL,
    meaning TEXT NOT NULL,
    category VARCHAR(50),
    confidence DOUBLE PRECISION
);

CREATE TABLE unknown_slang (
    id SERIAL PRIMARY KEY,
    term VARCHAR(255) NOT NULL,
    sentence TEXT NOT NULL,
    user_id INTEGER NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    review_status review_status_enum NOT NULL DEFAULT 'pending'
);

CREATE TABLE feedback (
    id SERIAL PRIMARY KEY,
    message_id INTEGER NOT NULL REFERENCES messages(message_id) ON DELETE CASCADE,
    intervention decision_action_enum NOT NULL,
    user_response user_response_enum,
    outcome outcome_enum,
    reward DOUBLE PRECISION,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_conversations_user_id ON conversations(user_id);
CREATE INDEX idx_messages_conversation_id ON messages(conversation_id);
CREATE INDEX idx_messages_user_id ON messages(user_id);
CREATE INDEX idx_toxicity_results_message_id ON toxicity_results(message_id);
CREATE INDEX idx_risk_results_message_id ON risk_results(message_id);
CREATE INDEX idx_intervention_logs_message_id ON intervention_logs(message_id);
CREATE INDEX idx_unknown_slang_user_id ON unknown_slang(user_id);
CREATE INDEX idx_feedback_message_id ON feedback(message_id);
CREATE INDEX idx_slang_dictionary_term ON slang_dictionary(slang_term);
