import assert from "node:assert/strict";
import test from "node:test";
import { deliveryConfirmed, visibleSentMessages } from "./messageLifecycle.js";

test("only confirmed sent messages appear in the conversation", () => {
  const visible = visibleSentMessages([
    { role: "user", text: "ok", sendStatus: "sent" },
    { role: "user", text: "watch it", sendStatus: "pending_warning" },
    { role: "user", text: "not yet", sendStatus: "delayed" },
    { role: "user", text: "rewrite", sendStatus: "pending_rewrite" },
    { role: "user", text: "dropped", sendStatus: "cancelled" },
    { role: "user", text: "broken", sendStatus: "failed" },
  ]);
  assert.deepEqual(visible.map((item) => item.text), ["ok"]);
});

test("analysis success is not delivery", () => {
  assert.equal(deliveryConfirmed({ analysis_complete: true, decision: "WARN", send_status: "pending_warning" }), false);
  assert.equal(deliveryConfirmed({ sent: true, send_status: "pending_warning" }), false);
  assert.equal(deliveryConfirmed({ sent: true, send_status: "sent" }), true);
});
