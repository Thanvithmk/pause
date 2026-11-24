import { Navigate, Route, Routes } from "react-router-dom";
import Chat from "./components/Chat";
import BehaviorDashboard from "./components/BehaviorDashboard";
import MessageHistory from "./components/MessageHistory";
import SettingsPage from "./components/SettingsPage";
import { ShellProvider } from "./components/AppShell";

function App() {
  return (
    <ShellProvider>
      <Routes>
        <Route path="/" element={<Navigate to="/chat" replace />} />
        <Route path="/chat" element={<Chat />} />
        <Route path="/dashboard" element={<BehaviorDashboard />} />
        <Route path="/history" element={<MessageHistory />} />
        <Route path="/settings" element={<SettingsPage />} />
        <Route path="/login" element={<Navigate to="/chat" replace />} />
      </Routes>
    </ShellProvider>
  );
}

export default App;
