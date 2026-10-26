"""Claude Opus 5.5 as a chat partner in the app (needs an Anthropic API key).

The key lives in the Windows Credential Manager (keyring), or in ANTHROPIC_API_KEY;
it is never written to the repo. Assistant turns are kept exactly as the API returned
them (thinking and fallback blocks included) and sent back unchanged, so the history
stays append-only.
"""
import os

import anthropic
import keyring

SERVICE = "HydrarGyrum HyGra"
MODEL = "claude-opus-5-5"
PRICE_IN, PRICE_OUT = 4.00, 20.00          # $ per million tokens (Claude Opus 5.5)
SYSTEM = ("Je bent Claude, de assistent in HyGra, een desktop-app van HydrarGyrum. "
          "Antwoord in de taal van de gebruiker (meestal Nederlands), helder en to the point. "
          "Gebruik Markdown; zet code altijd in een codeblok met de taal erbij.")


def get_key():
    try:
        return os.environ.get("ANTHROPIC_API_KEY") or keyring.get_password(SERVICE, "anthropic")
    except keyring.errors.KeyringError:
        return os.environ.get("ANTHROPIC_API_KEY")


def set_key(key):
    keyring.set_password(SERVICE, "anthropic", key.strip())


def check_key(key):
    """Cheap validity check: list one model."""
    try:
        anthropic.Anthropic(api_key=key.strip(), max_retries=0).models.list(limit=1)
        return None
    except anthropic.AuthenticationError:
        return "Deze sleutel wordt geweigerd."
    except anthropic.APIConnectionError:
        return "Geen verbinding met Anthropic."
    except anthropic.APIStatusError as e:
        return f"Fout {e.status_code}: {e.message}"


def reply(history, on_text, on_status):
    """history: API messages (user strings, assistant content lists as returned before).
    Streams text through on_text; returns the assistant content to store, usage and cost."""
    key = get_key()
    if not key:
        return {"error": "nokey"}
    client = anthropic.Anthropic(api_key=key)
    try:
        with client.beta.messages.stream(
            model=MODEL,
            max_tokens=64000,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",                       # a declined request is retried on Anthropic's fallback model
            thinking={"type": "adaptive", "display": "summarized"},
            output_config={"effort": "medium"},
            system=SYSTEM,
            messages=history,
        ) as stream:
            for event in stream:
                if event.type == "content_block_start" and event.content_block.type == "thinking":
                    on_status("denkt na")
                elif event.type == "content_block_start" and event.content_block.type == "text":
                    on_status("schrijft")
                elif event.type == "content_block_delta" and event.delta.type == "text_delta":
                    on_text(event.delta.text)
            msg = stream.get_final_message()
    except anthropic.AuthenticationError:
        return {"error": "nokey"}
    except anthropic.RateLimitError:
        return {"error": "Even te veel verzoeken; probeer het zo opnieuw."}
    except anthropic.APIStatusError as e:
        return {"error": f"Fout {e.status_code}: {e.message}"}
    except anthropic.APIConnectionError:
        return {"error": "Geen verbinding met Anthropic (internet?)."}

    if msg.stop_reason == "refusal":
        return {"error": "Claude weigerde dit verzoek.", "raw": [b.model_dump(by_alias=True, exclude_none=True) for b in msg.content]}
    u = msg.usage
    tokens_in = (u.input_tokens or 0) + (u.cache_read_input_tokens or 0) + (u.cache_creation_input_tokens or 0)
    cost = ((u.input_tokens or 0) + 0.1 * (u.cache_read_input_tokens or 0) + 1.25 * (u.cache_creation_input_tokens or 0)) \
        * PRICE_IN / 1e6 + (u.output_tokens or 0) * PRICE_OUT / 1e6
    text = "".join(b.text for b in msg.content if b.type == "text")
    return {"text": text, "raw": [b.model_dump(by_alias=True, exclude_none=True) for b in msg.content],
            "tokens_in": tokens_in, "tokens_out": u.output_tokens, "cost": round(cost, 4), "model": msg.model}
