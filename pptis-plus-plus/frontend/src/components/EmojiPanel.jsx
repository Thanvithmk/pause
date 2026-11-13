import { useEffect, useRef, useState } from "react";

const EMOJIS = [
  ["😂", "face with tears of joy"],
  ["🤣", "rolling on the floor laughing"],
  ["😭", "loudly crying face"],
  ["💀", "skull"],
  ["🔥", "fire"],
  ["❤️", "red heart"],
  ["😍", "smiling face with heart-eyes"],
  ["🥰", "smiling face with hearts"],
  ["😘", "face blowing a kiss"],
  ["😊", "smiling face with smiling eyes"],
  ["🙂", "slightly smiling face"],
  ["😐", "neutral face"],
  ["😒", "unamused face"],
  ["🙄", "face with rolling eyes"],
  ["🤔", "thinking face"],
  ["😬", "grimacing face"],
  ["😡", "pouting face"],
  ["🤬", "face with symbols on mouth"],
  ["💔", "broken heart"],
  ["💖", "sparkling heart"],
  ["💯", "hundred points"],
  ["👍", "thumbs up"],
  ["👎", "thumbs down"],
  ["🙏", "folded hands"],
  ["👀", "eyes"],
  ["🤡", "clown face"],
  ["🤦", "person facepalming"],
  ["🤷", "person shrugging"],
  ["💪", "flexed biceps"],
  ["🤝", "handshake"],
  ["👑", "crown"],
  ["🧠", "brain"],
  ["✨", "sparkles"],
  ["🎉", "party popper"],
  ["🎮", "video game"],
  ["🏆", "trophy"],
  ["📈", "chart increasing"],
  ["📉", "chart decreasing"],
  ["✅", "check mark button"],
  ["❌", "cross mark"],
  ["⚠️", "warning"],
  ["👊", "oncoming fist"],
  ["🔪", "kitchen knife"],
  ["🔫", "water pistol"],
  ["💣", "bomb"],
  ["🫠", "melting face"],
  ["🥵", "hot face"],
  ["🥶", "cold face"],
  ["🫡", "saluting face"],
];

function EmojiPanel({ onInsert }) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef(null);

  useEffect(() => {
    if (!open) {
      return undefined;
    }
    function onPointerDown(event) {
      if (!rootRef.current?.contains(event.target)) {
        setOpen(false);
      }
    }
    function onKeyDown(event) {
      if (event.key === "Escape") {
        setOpen(false);
      }
    }
    document.addEventListener("mousedown", onPointerDown);
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("mousedown", onPointerDown);
      document.removeEventListener("keydown", onKeyDown);
    };
  }, [open]);

  return (
    <div ref={rootRef} className="relative">
      <button
        type="button"
        aria-label="Insert emoji"
        aria-expanded={open}
        aria-haspopup="dialog"
        onClick={() => setOpen((current) => !current)}
        className={`grid h-10 w-10 place-items-center rounded-full ${
          open ? "bg-[#efeefe] text-[#5b4dff]" : "text-slate-400 hover:bg-[#f4f5f9] hover:text-slate-600"
        }`}
      >
        <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="1.8">
          <circle cx="12" cy="12" r="8" />
          <path d="M8.5 10h.01M15.5 10h.01M8.5 14.5c.8 1.2 2 1.8 3.5 1.8s2.7-.6 3.5-1.8" strokeLinecap="round" />
        </svg>
      </button>
      {open && (
        <div
          role="dialog"
          aria-label="Emoji"
          className="absolute bottom-12 left-0 z-20 w-[280px] rounded-2xl border border-slate-100 bg-white p-3 shadow-[0_16px_40px_rgba(76,70,140,0.16)]"
        >
          <p className="px-1 pb-2 text-xs font-semibold uppercase tracking-wide text-slate-400">Emoji</p>
          <div className="grid max-h-52 grid-cols-7 gap-1 overflow-y-auto">
            {EMOJIS.map(([emoji, description]) => (
              <button
                key={emoji}
                type="button"
                title={description}
                aria-label={description}
                onClick={() => onInsert(emoji)}
                className="grid h-9 w-9 place-items-center rounded-lg text-xl hover:bg-[#f4f5f9]"
              >
                {emoji}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default EmojiPanel;
