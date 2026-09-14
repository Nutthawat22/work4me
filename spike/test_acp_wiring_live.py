"""
spike/test_acp_wiring_live.py

Throwaway live-wiring check (Phase 3 verification). NOT a pytest test --
run directly with `python spike/test_acp_wiring_live.py`.

Exercises the real end-to-end path:
    specialists.llm_client.call_llm(..., provider="acp")
    -> specialists.providers.PROVIDERS["acp"] (acp_client.acp_call)
    -> real `opencode acp` subprocess (spawned via agent-client-protocol)
    -> real LiteLLM proxy (config["litellm_url"])

No mocks/stubs anywhere in this script. Uses the real
~/.config/Agents/config.json for litellm_url/litellm_key.

Run with the ROS2 PYTHONPATH leak worked around:
    env -u PYTHONPATH spike/venv/bin/python spike/test_acp_wiring_live.py
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from specialists.llm_client import call_llm
from specialists.providers.acp_client import teardown

CONFIG_PATH = os.path.expanduser("~/.config/Agents/config.json")
MODEL = "claude-sonnet-5"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return json.load(f)


def redact(cfg: dict) -> dict:
    out = dict(cfg)
    if "litellm_key" in out:
        out["litellm_key"] = "***REDACTED***"
    return out


def main() -> int:
    config = load_config()
    print(f"Loaded config from {CONFIG_PATH}: {redact(config)}")

    ok = True

    # Call 1: pays handshake cost (subprocess spawn + ACP initialize + new_session)
    messages_1 = [{"role": "user", "content": "Say hello in exactly 3 words."}]
    t0 = time.monotonic()
    result_1 = call_llm(messages_1, MODEL, config, provider="acp", timeout=60)
    t1 = time.monotonic()
    elapsed_1 = t1 - t0

    print(f"\n=== Call 1 (prompt: {messages_1[0]['content']!r}) ===")
    print(f"Elapsed: {elapsed_1:.2f}s")
    print(f"Result: {result_1!r}")

    is_error_1 = result_1.startswith(f"[{MODEL}] Error:")
    is_nonempty_1 = bool(result_1.strip())
    print(f"Is error string: {is_error_1}")
    print(f"Is non-empty: {is_nonempty_1}")
    if is_error_1 or not is_nonempty_1:
        ok = False

    # Call 2: same role_key -> should reuse pooled session, no handshake.
    messages_2 = [{"role": "user", "content": "What is 2+2? Answer with just the number."}]
    t2 = time.monotonic()
    result_2 = call_llm(messages_2, MODEL, config, provider="acp", timeout=60)
    t3 = time.monotonic()
    elapsed_2 = t3 - t2

    print(f"\n=== Call 2 (prompt: {messages_2[0]['content']!r}) ===")
    print(f"Elapsed: {elapsed_2:.2f}s")
    print(f"Result: {result_2!r}")

    is_error_2 = result_2.startswith(f"[{MODEL}] Error:")
    is_nonempty_2 = bool(result_2.strip())
    print(f"Is error string: {is_error_2}")
    print(f"Is non-empty: {is_nonempty_2}")
    if is_error_2 or not is_nonempty_2:
        ok = False

    print(f"\n=== Timing comparison (indirect evidence of session reuse) ===")
    print(f"Call 1 (handshake + prompt): {elapsed_1:.2f}s")
    print(f"Call 2 (prompt only, pooled): {elapsed_2:.2f}s")
    print(f"Call 2 faster than call 1: {elapsed_2 < elapsed_1}")

    teardown()
    print("\nteardown() called.")

    print(f"\n=== VERDICT: {'PASS' if ok else 'FAIL'} ===")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
