function SettingsPage() {
  return (
    <div className="p-5 md:p-8">
      <h2 className="text-xl font-semibold text-slate-900">Settings</h2>
      <p className="mt-1 text-sm text-slate-400">Pause checks a message before it is sent.</p>
      <div className="mt-6 max-w-md rounded-3xl bg-[#f6f7fb] p-5">
        <p className="text-lg font-semibold text-slate-900">Pause</p>
        <p className="mt-1 text-sm text-slate-500">Think before you send.</p>
      </div>
    </div>
  );
}

export default SettingsPage;
