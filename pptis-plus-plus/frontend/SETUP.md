# Pause frontend setup

The UI is a Vite + React 18 app with Tailwind CSS, React Router, Axios, and Recharts.

## Node.js

Install **Node.js 18+** (20 LTS recommended) and npm. Confirm:

```bash
node -v
npm -v
```

## Dependencies

From `pptis-plus-plus/frontend`:

```bash
npm install
```

This installs `react`, `react-dom`, `react-router-dom`, `axios`, `recharts`, Vite, Tailwind, PostCSS, and Autoprefixer.

## Development server

```bash
npm run dev
```

Vite serves on [http://localhost:3000](http://localhost:3000) (`vite.config.js`). `/api` and `/auth` are proxied to `http://localhost:5000`. Start the Flask backend first.

## Production build

```bash
npm run build
npm run preview
```

`build` writes static files to `frontend/dist`. Serve that directory behind nginx, Caddy, or an object store. `preview` is a local production-like check.

## Environment variables

Vite only exposes variables prefixed with `VITE_`. Create `frontend/.env`:

```bash
VITE_API_URL=http://localhost:5000
```

| Variable | Purpose | Default |
|---|---|---|
| `VITE_API_URL` | Axios `baseURL` | `http://localhost:5000` |

In production set this to the public API origin, for example `https://api.example.com`. CORS on the backend must allow the frontend origin (`CORS_ORIGINS`).

The chat opens directly. There is no login token.

## Component documentation

| File | Role |
|---|---|
| `src/main.jsx` | React root + `BrowserRouter` |
| `src/App.jsx` | Session restore, login/register, Chat / Dashboard nav |
| `src/components/Chat.jsx` | Thread, wires text analysis and results |
| `src/components/TextInput.jsx` | Placeholder “Type your message…”, Send / Analyze → `POST /api/analyze` |
| `src/components/AnalysisResults.jsx` | Six toxicity bars, PACL, `P_esc`, final risk, color-coded decision |
| `src/components/BehaviorDashboard.jsx` | 7/30/90 day charts from `GET /api/dashboard` |
| `src/services/api.js` | Axios instance, token helpers, API wrappers |

Decision colors: ALLOW green, WARN yellow, REWRITE orange, DELAYED-SEND red. The UI does not offer “send original” or “send anyway”.

Routes:

- `/login` — shown when unauthenticated (inline in `App.jsx`)
- `/chat` — analysis console
- `/dashboard` — behavior stats
