import axios from "axios";

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || "http://localhost:5000",
  timeout: 60000,
});

export async function analyzeText(text, conversationId) {
  const payload = { text };
  if (conversationId) {
    payload.conversation_id = conversationId;
  }
  const { data } = await api.post("/api/analyze", payload);
  return data;
}

export async function recordIntervention({ messageId, userResponse, outcome }) {
  const { data } = await api.post("/api/intervention-response", {
    message_id: messageId,
    user_response: userResponse,
    outcome,
  });
  return data;
}

export async function sendMessage({ messageId, conversationId, text, userAction = "send" }) {
  const { data } = await api.post("/api/send", {
    message_id: messageId,
    conversation_id: conversationId,
    text,
    user_action: userAction,
  });
  return data;
}

export async function getDelayedSendStatus(messageId) {
  const { data } = await api.get(`/api/delayed-send/${messageId}`);
  return data;
}

export async function listConversations() {
  const { data } = await api.get("/api/conversations");
  return data;
}

export async function createConversation() {
  const { data } = await api.post("/api/conversations");
  return data;
}

export async function getConversation(conversationId) {
  const { data } = await api.get(`/api/conversation/${conversationId}`);
  return data;
}

export async function getDashboard(days = 30) {
  const { data } = await api.get("/api/dashboard", { params: { days } });
  return data;
}

export async function getProfile() {
  const { data } = await api.get("/api/user/profile");
  return data;
}

export default api;
