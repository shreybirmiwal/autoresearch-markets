"""Use Claude to draft a tweet thread for the calibration chart."""
import base64, anthropic

with open("charts/calibration_all_trades.png", "rb") as f:
    img_b64 = base64.standard_b64encode(f.read()).decode()

client = anthropic.Anthropic()

response = client.messages.create(
    model="claude-opus-4-6",
    max_tokens=1024,
    messages=[{
        "role": "user",
        "content": [
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": img_b64}
            },
            {
                "type": "text",
                "text": (
                    "I ran an analysis on 37 million Polymarket prediction market trades. "
                    "This chart shows: if you bought a contract at price X, what was your average PnL? "
                    "In a perfectly efficient market every bar should be $0. "
                    "Write a single tweet (max 280 chars) about what this chart reveals. "
                    "Use enticing, accessible language — words like 'underdogs', 'favorites', 'overrated'. "
                    "Explain the simple human why: people overpay for underdogs (longshot bias) and underpay for favorites. "
                    "Use one or two real numbers. Hook them with the surprising finding. No hashtags, no emojis."
                )
            }
        ]
    }]
)

print(response.content[0].text)
