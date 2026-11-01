import axios from "axios";

const TOKEN_KEY = "pptis_token";

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || "http://localhost:5000",
  timeout: 60000,
});

export function getToken() {
  return localStorage.getItem(TOKEN_KEY);
}

export function setToken(token) {
  if (token) {
    localStorage.setItem(TOKEN_KEY, token);
  } else {
    localStorage.removeItem(TOKEN_KEY);
  }
}

export function clearToken() {
  localStorage.removeItem(TOKEN_KEY);
}

api.interceptors.request.use((config) => {
  const token = getToken();
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      clearToken();
    }
    return Promise.reject(error);
  }
);

export async function analyzeText(text, conversationId) {
  const payload = { text };
  if (conversationId) {
    payload.conversation_id = conversationId;
  }
  const { data } = await api.post("/api/analyze", payload);
  return data;
}

export async function analyzeVoice(audioBlob, filename = "recording.webm", conversationId) {
  const form = new FormData();
  form.append("audio", audioBlob, filename);
  if (conversationId) {
    form.append("conversation_id", conversationId);
  }
  const { data } = await api.post("/api/analyze-voice", form, {
    headers: { "Content-Type": "multipart/form-data" },
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

export async function login(identifier, password) {
  const { data } = await api.post("/api/auth/login", {
    identifier,
    email: identifier,
    username: identifier,
    password,
  });
  if (data.token) {
    setToken(data.token);
  }
  return data;
}

export async function register(username, email, password) {
  const { data } = await api.post("/api/auth/register", {
    username,
    email,
    password,
  });
  if (data.token) {
    setToken(data.token);
  }
  return data;
}

export async function logout() {
  try {
    await api.post("/api/auth/logout");
  } catch {
    /* token is cleared locally even if the server is unreachable */
  }
  clearToken();
}

export default api;
