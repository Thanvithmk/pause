import { NavLink } from "react-router-dom";

const NAV = [
  { to: "/chat", label: "Chat", icon: "chat" },
  { to: "/dashboard", label: "Behavior Dashboard", icon: "chart" },
  { to: "/history", label: "Message History", icon: "clock" },
  { to: "/settings", label: "Settings", icon: "gear" },
];

function Icon({ name }) {
  const common = "h-5 w-5";
  if (name === "chat") {
    return (
      <svg className={common} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
        <path d="M7 17.5 4 20V6.5A2.5 2.5 0 0 1 6.5 4h11A2.5 2.5 0 0 1 20 6.5v8A2.5 2.5 0 0 1 17.5 17H7Z" strokeLinejoin="round" />
      </svg>
    );
  }
  if (name === "chart") {
    return (
      <svg className={common} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
        <path d="M5 19V10M12 19V5M19 19v-7" strokeLinecap="round" />
      </svg>
    );
  }
  if (name === "clock") {
    return (
      <svg className={common} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
        <circle cx="12" cy="12" r="8" />
        <path d="M12 8v4.5l3 2" strokeLinecap="round" />
      </svg>
    );
  }
  return (
    <svg className={common} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
      <circle cx="12" cy="12" r="3" />
      <path d="M12 3.5v2.2M12 18.3v2.2M3.5 12h2.2M18.3 12h2.2M6 6l1.6 1.6M16.4 16.4 18 18M18 6l-1.6 1.6M7.6 16.4 6 18" strokeLinecap="round" />
    </svg>
  );
}

export function ShellProvider({ children }) {
  return (
      <div className="flex h-screen flex-col overflow-hidden bg-[#e7edf8] px-3 py-5 text-slate-800 md:px-8 md:py-8">
        <div className="mx-auto flex min-h-0 w-full max-w-6xl flex-1 flex-col overflow-hidden rounded-[28px] bg-white shadow-[0_20px_60px_rgba(76,70,140,0.12)]">
          <header className="flex shrink-0 items-center justify-between border-b border-slate-100 px-5 py-4">
            <div className="flex items-center gap-3">
              <span className="grid h-11 w-11 place-items-center rounded-2xl bg-[#5b4dff] text-white">
                <svg viewBox="0 0 24 24" className="h-6 w-6" fill="none" stroke="currentColor" strokeWidth="1.8">
                  <path d="M12 3 5 6v6c0 4.2 2.8 7.2 7 9 4.2-1.8 7-4.8 7-9V6l-7-3Z" strokeLinejoin="round" />
                  <path d="m9 12 2 2 4-4" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              </span>
              <div>
                <p className="text-lg font-bold leading-none text-slate-900">Pause</p>
                <p className="mt-1 text-xs text-slate-400">Think Before You Send</p>
              </div>
            </div>
            <span className="grid h-10 w-10 place-items-center rounded-full bg-[#5b4dff] text-sm font-semibold text-white">
              P
            </span>
          </header>
          <div className="flex min-h-0 flex-1">
            <aside className="hidden w-56 shrink-0 overflow-y-auto border-r border-slate-100 p-4 md:block">
              <nav className="space-y-1">
                {NAV.map((item) => (
                  <NavLink
                    key={item.to}
                    to={item.to}
                    className={({ isActive }) =>
                      `flex items-center gap-3 rounded-2xl px-3 py-2.5 text-sm font-medium ${
                        isActive ? "bg-[#f3efff] text-[#5b4dff]" : "text-slate-500 hover:bg-slate-50"
                      }`
                    }
                  >
                    <Icon name={item.icon} />
                    {item.label}
                  </NavLink>
                ))}
              </nav>
            </aside>
            <div className="flex min-h-0 min-w-0 flex-1 flex-col overflow-y-auto">{children}</div>
          </div>
          <nav className="flex shrink-0 border-t border-slate-100 md:hidden">
            {NAV.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                className={({ isActive }) =>
                  `flex flex-1 flex-col items-center gap-1 py-2 text-[10px] ${
                    isActive ? "text-[#5b4dff]" : "text-slate-400"
                  }`
                }
              >
                <Icon name={item.icon} />
                {item.label.split(" ")[0]}
              </NavLink>
            ))}
          </nav>
        </div>
      </div>
  );
}
