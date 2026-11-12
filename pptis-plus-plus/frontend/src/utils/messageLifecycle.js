export function visibleSentMessages(messages) {
  return messages.filter((message) => message.role === "user" && message.sendStatus === "sent");
}

export function deliveryConfirmed(result) {
  return Boolean(result && result.sent === true && result.send_status === "sent");
}
