function SettingsPage({ user, onLogout }) {
  return (
    <div className="p-5 md:p-8">
      <h2 className="text-xl font-semibold text-slate-900">Settings</h2>
      <p className="mt-1 text-sm text-slate-400">Account for this browser session.</p>
      <div className="mt-6 max-w-md rounded-3xl bg-[#f6f7fb] p-5">
        <p className="text-sm text-slate-500">Signed in as</p>
        <p className="mt-1 text-lg font-semibold text-slate-900">{user?.username || user?.email}</p>
        {user?.email && <p className="text-sm text-slate-400">{user.email}</p>}
        <button
          type="button"
          onClick={onLogout}
          className="mt-5 rounded-full bg-[#5b4dff] px-5 py-2 text-sm font-semibold text-white"
        >
          Log out
        </button>
      </div>
    </div>
  );
}

export default SettingsPage;
