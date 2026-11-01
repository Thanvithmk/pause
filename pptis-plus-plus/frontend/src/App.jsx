import { useEffect, useState } from "react";
import { Navigate, Route, Routes, useNavigate } from "react-router-dom";
import Chat from "./components/Chat";
import BehaviorDashboard from "./components/BehaviorDashboard";
import MessageHistory from "./components/MessageHistory";
import SettingsPage from "./components/SettingsPage";
import { ShellProvider } from "./components/AppShell";
import { getProfile, getToken, login, logout, register } from "./services/api";

function App() {
  const [user, setUser] = useState(null);
  const [checking, setChecking] = useState(true);
  const [authMode, setAuthMode] = useState("login");
  const [authForm, setAuthForm] = useState({ username: "", email: "", password: "" });
  const [authError, setAuthError] = useState("");
  const [authLoading, setAuthLoading] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    async function restoreSession() {
      if (!getToken()) {
        setChecking(false);
        return;
      }
      try {
        const profile = await getProfile();
        setUser(profile.user || profile);
      } catch {
        logout();
        setUser(null);
      } finally {
        setChecking(false);
      }
    }
    restoreSession();
  }, []);

  async function handleAuth(event) {
    event.preventDefault();
    setAuthError("");
    setAuthLoading(true);
    try {
      const result =
        authMode === "login"
          ? await login(authForm.email || authForm.username, authForm.password)
          : await register(authForm.username, authForm.email, authForm.password);
      setUser(result.user || { username: authForm.username, email: authForm.email });
      navigate("/chat");
    } catch (error) {
      const apiMessage = error.response?.data?.message;
      const unreachable =
        !error.response &&
        (error.code === "ERR_NETWORK" || error.message?.includes("Network Error"));
      setAuthError(
        apiMessage ||
          (unreachable
            ? "Cannot reach the API. Start Flask on port 5000 (cd backend && python app.py)."
            : error.message || "Authentication failed")
      );
    } finally {
      setAuthLoading(false);
    }
  }

  async function handleLogout() {
    await logout();
    setUser(null);
    navigate("/login");
  }

  if (checking) {
    return (
      <div className="grid min-h-screen place-items-center bg-[#e7edf8] text-slate-400">
        Restoring session…
      </div>
    );
  }

  if (!user) {
    return (
      <div className="grid min-h-screen place-items-center bg-[#e7edf8] px-4">
        <form
          onSubmit={handleAuth}
          className="w-full max-w-md space-y-4 rounded-[28px] bg-white p-8 shadow-[0_20px_60px_rgba(76,70,140,0.12)]"
        >
          <div className="flex items-center gap-3">
            <span className="grid h-11 w-11 place-items-center rounded-2xl bg-[#5b4dff] text-white">
              <svg viewBox="0 0 24 24" className="h-6 w-6" fill="none" stroke="currentColor" strokeWidth="1.8">
                <path d="M12 3 5 6v6c0 4.2 2.8 7.2 7 9 4.2-1.8 7-4.8 7-9V6l-7-3Z" strokeLinejoin="round" />
              </svg>
            </span>
            <div>
              <p className="text-lg font-bold text-slate-900">PPTIS++</p>
              <p className="text-xs text-slate-400">Think Before You Send</p>
            </div>
          </div>
          <h1 className="text-2xl font-semibold text-slate-900">Sign in to continue</h1>
          {authMode === "register" && (
            <input
              className="w-full rounded-2xl bg-[#f6f7fb] px-4 py-3 outline-none ring-1 ring-transparent focus:ring-[#5b4dff]"
              placeholder="Username"
              value={authForm.username}
              onChange={(event) => setAuthForm({ ...authForm, username: event.target.value })}
            />
          )}
          <input
            className="w-full rounded-2xl bg-[#f6f7fb] px-4 py-3 outline-none ring-1 ring-transparent focus:ring-[#5b4dff]"
            placeholder="Email or username"
            value={authForm.email}
            onChange={(event) => setAuthForm({ ...authForm, email: event.target.value })}
          />
          <input
            type="password"
            className="w-full rounded-2xl bg-[#f6f7fb] px-4 py-3 outline-none ring-1 ring-transparent focus:ring-[#5b4dff]"
            placeholder="Password"
            value={authForm.password}
            onChange={(event) => setAuthForm({ ...authForm, password: event.target.value })}
          />
          {authMode === "register" && (
            <p className="text-xs text-slate-400">
              Password needs 8+ characters, one uppercase letter, one digit, and one special character.
            </p>
          )}
          {authError && <p className="text-sm text-rose-500">{authError}</p>}
          <button
            type="submit"
            disabled={authLoading}
            className="w-full rounded-full bg-[#5b4dff] py-3 font-semibold text-white disabled:opacity-60"
          >
            {authLoading ? "Working…" : authMode === "login" ? "Log in" : "Register"}
          </button>
          <button
            type="button"
            className="w-full text-sm text-slate-400"
            onClick={() => setAuthMode(authMode === "login" ? "register" : "login")}
          >
            {authMode === "login" ? "Need an account? Register" : "Have an account? Log in"}
          </button>
        </form>
      </div>
    );
  }

  return (
    <ShellProvider user={user} onLogout={handleLogout}>
      <Routes>
        <Route path="/" element={<Navigate to="/chat" replace />} />
        <Route path="/chat" element={<Chat user={user} />} />
        <Route path="/dashboard" element={<BehaviorDashboard />} />
        <Route path="/history" element={<MessageHistory />} />
        <Route path="/settings" element={<SettingsPage user={user} onLogout={handleLogout} />} />
        <Route path="/login" element={<Navigate to="/chat" replace />} />
      </Routes>
    </ShellProvider>
  );
}

export default App;
