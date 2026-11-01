/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["IBM Plex Sans", "ui-sans-serif", "system-ui"],
        mono: ["IBM Plex Mono", "ui-monospace", "monospace"],
      },
      colors: {
        ink: "#12151c",
        panel: "#1b2030",
        allow: "#22c55e",
        warn: "#eab308",
        rewrite: "#f97316",
        delay: "#ef4444",
      },
    },
  },
  plugins: [],
};
