import { useEffect, useState } from "react";
import { Navigate, NavLink, Route, Routes, useNavigate } from "react-router-dom";
import Chat from "./components/Chat";
import BehaviorDashboard from "./components/BehaviorDashboard";
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
      <div className="min-h-screen grid place-items-center text-slate-400">
        Restoring session…
      </div>
    );
  }

  if (!user) {
    return (
      <div className="min-h-screen grid place-items-center px-4">
        <form
          onSubmit={handleAuth}
          className="w-full max-w-md rounded-2xl border border-slate-700 bg-panel p-8 space-y-4"
        >
          <p className="font-mono text-xs tracking-[0.2em] text-emerald-400">PPTIS++</p>
          <h1 className="text-2xl font-semibold">Sign in to continue</h1>
          {authMode === "register" && (
            <input
              className="w-full rounded-lg bg-ink px-3 py-2 outline-none ring-1 ring-slate-600"
              placeholder="Username"
              value={authForm.username}
              onChange={(event) => setAuthForm({ ...authForm, username: event.target.value })}
            />
          )}
          <input
            className="w-full rounded-lg bg-ink px-3 py-2 outline-none ring-1 ring-slate-600"
            placeholder="Email or username"
            value={authForm.email}
            onChange={(event) => setAuthForm({ ...authForm, email: event.target.value })}
          />
          <input
            type="password"
            className="w-full rounded-lg bg-ink px-3 py-2 outline-none ring-1 ring-slate-600"
            placeholder="Password"
            value={authForm.password}
            onChange={(event) => setAuthForm({ ...authForm, password: event.target.value })}
          />
          {authMode === "register" && (
            <p className="text-xs text-slate-500">
              Password needs 8+ characters, one uppercase letter, one digit, and one special character.
            </p>
          )}
          {authError && <p className="text-sm text-delay">{authError}</p>}
          <button
            type="submit"
            disabled={authLoading}
            className="w-full rounded-lg bg-emerald-500 py-2 font-medium text-ink disabled:opacity-60"
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
    <div className="min-h-screen">
      <header className="flex items-center justify-between border-b border-slate-800 px-6 py-4">
        <div>
          <p className="font-mono text-xs tracking-[0.25em] text-emerald-400">PPTIS++</p>
          <p className="text-sm text-slate-400">Personalized Predictive Toxicity Intervention</p>
        </div>
        <nav className="flex items-center gap-4 text-sm">
          <NavLink
            to="/chat"
            className={({ isActive }) => (isActive ? "text-emerald-400" : "text-slate-400")}
          >
            Chat
          </NavLink>
          <NavLink
            to="/dashboard"
            className={({ isActive }) => (isActive ? "text-emerald-400" : "text-slate-400")}
          >
            Dashboard
          </NavLink>
          <span className="text-slate-500">{user.username || user.email}</span>
          <button type="button" onClick={handleLogout} className="text-slate-400">
            Log out
          </button>
        </nav>
      </header>
      <main className="mx-auto max-w-6xl px-4 py-6">
        <Routes>
          <Route path="/" element={<Navigate to="/chat" replace />} />
          <Route path="/chat" element={<Chat user={user} />} />
          <Route path="/dashboard" element={<BehaviorDashboard />} />
          <Route path="/login" element={<Navigate to="/chat" replace />} />
        </Routes>
      </main>
    </div>
  );
}

export default App;
