"""
specialists/providers/acp_client.py

Agent Client Protocol (ACP) adapter -- the sole LLM provider (see
specialists/providers/__init__.py's PROVIDERS registry; the former
chat_completions/responses HTTP adapters were removed in Phase 1 of
the ACP-native rearchitecture, see
dev-plans/agents/features/2026-09-14-acp-native-master-architecture.md).
Routes LLM calls through a spawned `opencode acp` subprocess speaking
JSON-RPC over stdio, instead of a direct HTTP request to a LiteLLM
proxy. Uses the `agent-client-protocol` PyPI package (imports as `acp`).

Session lifecycle: each config["models"][role] gets its own pooled
`opencode acp` subprocess + ACP session, spawned once and reused across
calls (see AcpSessionPool). Every pooled subprocess is spawned with an
isolated HOME (so it never discovers a developer's real
~/.config/opencode/) and OPENCODE_CONFIG pointed at a minimal generated
opencode.json (tools disabled, deny-all permissions, a custom
openai-compatible provider entry pointing at config["litellm_url"] /
config["litellm_key"]) -- this combination was verified in Phase 1's
follow-up spike (spike/acp_spike_minimal.py) to cut per-call input-token
overhead by ~99% versus an unconfigured spawn, and independently
verified in spike/acp_spike_provider_test.py (Phase 2 Step 0 research)
to actually route completions through our LiteLLM proxy rather than
silently falling back to some other model.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import acp

# Fallback module import name from Phase 1 findings: the PyPI package
# "agent-client-protocol" is imported as "acp", not "agent_client_protocol".

DEFAULT_AGENT_NAME = "minimal"

DEFAULT_DENY_ALL_PERMISSIONS: dict[str, str] = {
    "edit": "deny",
    "bash": "deny",
    "read": "deny",
    "glob": "deny",
    "grep": "deny",
    "list": "deny",
    "task": "deny",
    "webfetch": "deny",
    "websearch": "deny",
    "skill": "deny",
    "external_directory": "deny",
}


def _build_minimal_opencode_config(config: dict[str, Any], model: str) -> dict[str, Any]:
    """
    Build the minimal, per-role opencode.json config dict written to disk
    and passed via OPENCODE_CONFIG when spawning `opencode acp`.

    Uses a custom `@ai-sdk/openai-compatible` provider entry ("litellm")
    pointing baseURL/apiKey at our own LiteLLM proxy
    (config["litellm_url"] / config["litellm_key"]) -- verified in
    spike/acp_spike_provider_test.py to route real completions through
    our proxy (confirmed via a negative control: an intentionally wrong
    apiKey against this exact config shape fails with the LiteLLM
    proxy's own auth error, not a silent fallback to a different model).

    All tools disabled and every permission key set to "deny" on the
    default agent, since specialist/design/master calls only ever need
    text in/text out -- never actual file edits, shell commands, or
    other tool use. This is a pure function: no subprocess spawning, no
    I/O beyond building the dict, so it's independently unit-testable.

    Args:
        config: Loaded pipeline config dict, must contain "litellm_url"
            and "litellm_key".
        model: Model name string to route through the litellm provider
            entry (e.g. config["models"][role]["model"]).

    Returns:
        A JSON-serializable dict matching opencode.json's schema.
    """
    return {
        "$schema": "https://opencode.ai/config.json",
        "provider": {
            "litellm": {
                "name": "LiteLLM Gateway",
                "npm": "@ai-sdk/openai-compatible",
                "options": {
                    "apiKey": config["litellm_key"],
                    "baseURL": config["litellm_url"],
                },
                "models": {model: {"name": model}},
            },
        },
        "model": f"litellm/{model}",
        "agent": {
            DEFAULT_AGENT_NAME: {
                "mode": "primary",
                "description": "Minimal pipeline specialist agent (text in/text out only)",
                "prompt": "Respond directly. No commentary.",
                "tools": {"*": False},
                "permission": dict(DEFAULT_DENY_ALL_PERMISSIONS),
            },
        },
        "default_agent": DEFAULT_AGENT_NAME,
    }


def _build_schema_instruction(response_schema: dict[str, Any]) -> str:
    """
    Best-effort structured-output instruction for ACP.

    ACP has no token-constrained `response_format: json_schema,
    strict: true` mechanism like the removed chat_completions/responses
    HTTP adapters had (see the ACP-native rearchitecture design doc's
    "Accepted Risk: No Hard Schema Enforcement, Anywhere"). This
    function only prepends a clearly-labeled prompt instruction -- no
    enforcement, no guarantee the model actually complies. The
    reliability fallback is acp_call_with_retry's same-session
    retry-with-error-feedback loop (Option A), not this function.
    """
    return (
        "Respond with valid JSON matching this schema: "
        f"{json.dumps(response_schema['schema'])}. "
        "Output ONLY the JSON, no other text."
    )


def _messages_to_prompt_text(messages: list[dict]) -> str:
    """
    Flatten a chat_completions-style messages list into a single prompt
    string for ACP's session/prompt call, which takes a list of
    ContentBlocks rather than a role-tagged message array. System
    messages are prefixed so the model still sees them as instructions
    distinct from user content.
    """
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            parts.append(f"[System instructions]\n{content}")
        else:
            parts.append(content)
    return "\n\n".join(parts)


@dataclass
class _AccumulatingClient:
    """
    Minimal ACP Client implementation. Accumulates agent_message_chunk
    text from session/update notifications; denies every tool-use /
    file-access / terminal / elicitation request, since pooled sessions
    are configured deny-all and pipeline calls never expect actual tool
    use -- only text completion.
    """

    chunks: list[str] = field(default_factory=list)

    async def session_update(self, session_id: str, update, **kwargs: Any) -> None:
        if getattr(update, "sessionUpdate", None) == "agent_message_chunk":
            text = getattr(update.content, "text", None)
            if text:
                self.chunks.append(text)

    async def request_permission(self, session_id, tool_call, options, **kwargs: Any):
        raise acp.RequestError.method_not_found("request_permission not supported (deny-all pipeline session)")

    async def write_text_file(self, session_id, path, content, **kwargs: Any):
        raise acp.RequestError.method_not_found("write_text_file not supported (deny-all pipeline session)")

    async def read_text_file(self, session_id, path, line=None, limit=None, **kwargs: Any):
        raise acp.RequestError.method_not_found("read_text_file not supported (deny-all pipeline session)")

    async def create_terminal(self, session_id, command, args=None, env=None, cwd=None, output_byte_limit=None, **kwargs: Any):
        raise acp.RequestError.method_not_found("create_terminal not supported (deny-all pipeline session)")

    async def terminal_output(self, session_id, terminal_id, **kwargs: Any):
        raise acp.RequestError.method_not_found("terminal_output not supported (deny-all pipeline session)")

    async def release_terminal(self, session_id, terminal_id, **kwargs: Any):
        raise acp.RequestError.method_not_found("release_terminal not supported (deny-all pipeline session)")

    async def wait_for_terminal_exit(self, session_id, terminal_id, **kwargs: Any):
        raise acp.RequestError.method_not_found("wait_for_terminal_exit not supported (deny-all pipeline session)")

    async def kill_terminal(self, session_id, terminal_id, **kwargs: Any):
        raise acp.RequestError.method_not_found("kill_terminal not supported (deny-all pipeline session)")

    async def create_elicitation(self, message, mode, **kwargs: Any):
        raise acp.RequestError.method_not_found("create_elicitation not supported (deny-all pipeline session)")

    async def complete_elicitation(self, elicitation_id, **kwargs: Any):
        return None

    async def ext_method(self, method, params):
        raise acp.RequestError.method_not_found(f"ext_method {method} not supported (deny-all pipeline session)")

    async def ext_notification(self, method, params):
        return None

    def on_connect(self, conn) -> None:
        return None

    def accumulated_text(self) -> str:
        return "".join(self.chunks)


class _ForwardingClient:
    """
    The actual ACP Client bound to the connection at spawn time.
    Forwards session_update to whichever _AccumulatingClient is
    "current" -- necessary because the connection is created once per
    subprocess (at spawn time) but prompt() is called repeatedly across
    the pooled session's lifetime, and each call needs its own fresh
    chunk accumulator. AcpSession.prompt() sets `.current` before each
    call.
    """

    def __init__(self) -> None:
        self.current: Optional[_AccumulatingClient] = None

    async def session_update(self, session_id: str, update, **kwargs: Any) -> None:
        if self.current is not None:
            await self.current.session_update(session_id, update, **kwargs)

    async def request_permission(self, session_id, tool_call, options, **kwargs: Any):
        raise acp.RequestError.method_not_found("request_permission not supported (deny-all pipeline session)")

    async def write_text_file(self, session_id, path, content, **kwargs: Any):
        raise acp.RequestError.method_not_found("write_text_file not supported (deny-all pipeline session)")

    async def read_text_file(self, session_id, path, line=None, limit=None, **kwargs: Any):
        raise acp.RequestError.method_not_found("read_text_file not supported (deny-all pipeline session)")

    async def create_terminal(self, session_id, command, args=None, env=None, cwd=None, output_byte_limit=None, **kwargs: Any):
        raise acp.RequestError.method_not_found("create_terminal not supported (deny-all pipeline session)")

    async def terminal_output(self, session_id, terminal_id, **kwargs: Any):
        raise acp.RequestError.method_not_found("terminal_output not supported (deny-all pipeline session)")

    async def release_terminal(self, session_id, terminal_id, **kwargs: Any):
        raise acp.RequestError.method_not_found("release_terminal not supported (deny-all pipeline session)")

    async def wait_for_terminal_exit(self, session_id, terminal_id, **kwargs: Any):
        raise acp.RequestError.method_not_found("wait_for_terminal_exit not supported (deny-all pipeline session)")

    async def kill_terminal(self, session_id, terminal_id, **kwargs: Any):
        raise acp.RequestError.method_not_found("kill_terminal not supported (deny-all pipeline session)")

    async def create_elicitation(self, message, mode, **kwargs: Any):
        raise acp.RequestError.method_not_found("create_elicitation not supported (deny-all pipeline session)")

    async def complete_elicitation(self, elicitation_id, **kwargs: Any):
        return None

    async def ext_method(self, method, params):
        raise acp.RequestError.method_not_found(f"ext_method {method} not supported (deny-all pipeline session)")

    async def ext_notification(self, method, params):
        return None

    def on_connect(self, conn) -> None:
        return None


@dataclass
class AcpSession:
    """
    A live ACP connection: the spawned subprocess, its ACP session id,
    and the temp dirs backing HOME/OPENCODE_CONFIG/cwd for this session
    (kept alive for the session's lifetime and cleaned up on teardown).
    """

    conn: Any  # acp.ClientSideConnection
    process: Any  # asyncio.subprocess.Process
    session_id: str
    client: _ForwardingClient
    _cm: Any  # the async context manager yielding (conn, process), kept to close it later
    _home_dir: tempfile.TemporaryDirectory
    _config_dir: tempfile.TemporaryDirectory
    _cwd_dir: tempfile.TemporaryDirectory

    def is_alive(self) -> bool:
        returncode = getattr(self.process, "returncode", None)
        return returncode is None

    async def prompt(self, prompt_text: str, timeout: int):
        accumulator = _AccumulatingClient()
        self.client.current = accumulator
        prompt_resp = await asyncio.wait_for(
            self.conn.prompt(
                session_id=self.session_id,
                prompt=[acp.text_block(prompt_text)],
            ),
            timeout=timeout,
        )
        return prompt_resp, accumulator.accumulated_text()

    async def close(self) -> None:
        try:
            await self.conn.close()
        except Exception:
            pass
        try:
            await self._cm.__aexit__(None, None, None)
        except Exception:
            pass
        self._home_dir.cleanup()
        self._config_dir.cleanup()
        self._cwd_dir.cleanup()


class AcpSessionPool:
    """
    Manages pooled `opencode acp` subprocesses + ACP sessions, one per
    role_key, reused across calls within a pipeline run. Owns a single
    background thread running a persistent asyncio event loop -- every
    async ACP operation (spawn, handshake, prompt) is scheduled onto that
    loop via `asyncio.run_coroutine_threadsafe` and waited on
    synchronously, so callers (acp_call) never need to manage an event
    loop themselves.

    A single module-level instance (`_POOL` below) is shared by all
    roles, matching the design doc's "one shared background loop for all
    pooled roles" decision.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, AcpSession] = {}
        self._lock = threading.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is not None:
            return self._loop
        with self._lock:
            if self._loop is not None:
                return self._loop
            ready = threading.Event()

            def _run() -> None:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                self._loop = loop
                ready.set()
                loop.run_forever()

            thread = threading.Thread(target=_run, name="acp-session-pool-loop", daemon=True)
            thread.start()
            ready.wait()
            self._thread = thread
            return self._loop  # type: ignore[return-value]

    def _run_coro(self, coro, timeout: Optional[float] = None):
        """Schedule `coro` on the pool's background loop and block until done."""
        loop = self._ensure_loop()
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        return future.result(timeout=timeout)

    async def _create_session(self, config: dict[str, Any], model: str) -> AcpSession:
        home_dir = tempfile.TemporaryDirectory(prefix="acp-pool-home-")
        config_dir = tempfile.TemporaryDirectory(prefix="acp-pool-config-")
        cwd_dir = tempfile.TemporaryDirectory(prefix="acp-pool-cwd-")

        opencode_config = _build_minimal_opencode_config(config, model)
        config_path = os.path.join(config_dir.name, "opencode.json")
        with open(config_path, "w") as f:
            json.dump(opencode_config, f)

        env = {
            **os.environ,
            "HOME": home_dir.name,
            "OPENCODE_CONFIG": config_path,
        }

        client = _ForwardingClient()

        cm = acp.spawn_agent_process(client, "opencode", "acp", cwd=cwd_dir.name, env=env)
        conn, process = await cm.__aenter__()

        await conn.initialize(
            protocol_version=acp.PROTOCOL_VERSION,
            client_capabilities=acp.schema.ClientCapabilities(
                fs=acp.schema.FileSystemCapabilities(read_text_file=False, write_text_file=False),
                terminal=False,
            ),
        )
        session_resp = await conn.new_session(cwd=cwd_dir.name, mcp_servers=[])

        return AcpSession(
            conn=conn,
            process=process,
            session_id=session_resp.session_id,
            client=client,
            _cm=cm,
            _home_dir=home_dir,
            _config_dir=config_dir,
            _cwd_dir=cwd_dir,
        )

    def get_or_create(self, role_key: str, config: dict[str, Any], model: str) -> AcpSession:
        """
        Return the pooled AcpSession for `role_key`, spawning a new
        `opencode acp` subprocess + session if none exists yet, or if
        the previously pooled subprocess has died. Respawn is attempted
        exactly once on a detected-dead session; if the respawn itself
        fails, the exception propagates to the caller (acp_call), which
        converts it to the standard error-string format.
        """
        with self._lock:
            existing = self._sessions.get(role_key)

        if existing is not None and existing.is_alive():
            return existing

        if existing is not None and not existing.is_alive():
            # Stale entry from a dead subprocess -- drop it before respawning.
            with self._lock:
                self._sessions.pop(role_key, None)

        new_session = self._run_coro(self._create_session(config, model))
        with self._lock:
            self._sessions[role_key] = new_session
        return new_session

    def prompt_sync(self, role_key: str, config: dict[str, Any], model: str, prompt_text: str, timeout: int):
        """
        Get or create the pooled session for `role_key` and run a single
        prompt against it, synchronously. If the pooled session turns
        out to be dead (detected either before or during the call),
        respawn once and retry the prompt; if that second attempt also
        fails, the exception propagates to the caller.
        """
        session = self.get_or_create(role_key, config, model)
        try:
            return self._run_coro(session.prompt(prompt_text, timeout), timeout=timeout + 5)
        except Exception:
            if session.is_alive():
                raise
            # Session died mid-call -- respawn once and retry.
            with self._lock:
                self._sessions.pop(role_key, None)
            respawned = self.get_or_create(role_key, config, model)
            return self._run_coro(respawned.prompt(prompt_text, timeout), timeout=timeout + 5)

    def teardown_all(self) -> None:
        """
        Cleanly shut down every pooled subprocess/session. Safe to call
        at pipeline-run end (wiring this into pipeline/runner.py's
        actual run lifecycle is Phase 3 scope, not done here).
        """
        with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()

        if self._loop is not None:
            for session in sessions:
                try:
                    self._run_coro(session.close(), timeout=10)
                except Exception:
                    pass

            loop = self._loop
            loop.call_soon_threadsafe(loop.stop)
            if self._thread is not None:
                self._thread.join(timeout=5)
            self._loop = None
            self._thread = None


_POOL = AcpSessionPool()


def teardown() -> None:
    """
    Public accessor for pipeline/runner.py (and any other caller outside
    this module) to tear down the module-level session pool without
    reaching into the "private" _POOL directly. Delegates to
    AcpSessionPool.teardown_all() — see its docstring for behavior.
    Safe to call even if no ACP sessions were ever created (e.g. no role
    is configured with provider: "acp" yet).
    """
    _POOL.teardown_all()


def _role_key_for(model: str, config: dict[str, Any], session_scope: Optional[dict] = None) -> str:
    """
    Derive a pooling key distinguishing "agent instances" for the
    session pool.

    If `session_scope` (shape {"project": str, "role": str}) is given,
    the key is f"{project}::{role}" -- this is the real fix for the
    "two roles sharing a model incorrectly share a session" bug (and,
    worse, the "two separate pipeline runs with no distinguishing key
    at all could collide" bug): project+role is already fully specific,
    no model/URL needed.

    If `session_scope` is None (backward-compat path -- callers that
    haven't been updated to pass it yet, e.g. existing tests), falls
    back to the original (model, litellm_url) pooling key. This is
    unique enough to avoid two different models colliding on one pooled
    subprocess while still reusing a session across repeated calls for
    the same model, but does NOT distinguish roles or pipeline runs --
    that's exactly the gap session_scope closes.
    """
    if session_scope is not None:
        return f"{session_scope['project']}::{session_scope['role']}"
    return f"{model}::{config.get('litellm_url', '')}"


def acp_call(
    messages: list[dict],
    model: str,
    config: dict[str, Any],
    timeout: int = 60,
    response_schema: Optional[dict[str, Any]] = None,
    session_scope: Optional[dict] = None,
) -> str:
    """
    Send a prompt through a pooled `opencode acp` subprocess/session,
    routed to config["litellm_url"]/config["litellm_key"] via a
    generated minimal opencode.json (see _build_minimal_opencode_config).

    Args:
        messages: Full list of {"role": ..., "content": ...} messages to
            send. Flattened into a single prompt string for ACP's
            session/prompt call (see _messages_to_prompt_text), since
            ACP takes a list of ContentBlocks rather than a role-tagged
            message array.
        model: Model name string (e.g. config["models"]["design"]["model"]).
        config: Loaded config dict, must contain "litellm_url" and
            "litellm_key".
        timeout: Prompt timeout in seconds. Session spawn/handshake is
            not subject to this timeout (handshake typically completes
            in a few seconds per Phase 1 spike measurements); only the
            session/prompt round-trip is bounded by it.
        response_schema: Optional dict with keys "name" and "schema" (a
            JSON Schema object). ACP has no token-constrained
            structured-output primitive, so this is NOT enforced --
            it's merely prepended to the prompt text as a best-effort
            instruction (see _build_schema_instruction). This function
            has no retry loop of its own -- see acp_call_with_retry for
            the reliability fallback (Option A). None (default) sends
            no schema instruction at all.
        session_scope: Optional dict {"project": str, "role": str}
            identifying which pipeline run ("project", e.g. the run
            dir's basename) and config role ("design"/"specialist"/
            "master"/etc.) this call belongs to -- see _role_key_for.
            None (default) falls back to the pre-existing
            (model, litellm_url) pooling key.

    Returns:
        Accumulated agent_message_chunk text on success, or a formatted
        f"[{model}] Error: ..." string (never raises) on: subprocess
        spawn failure, handshake failure, prompt timeout, or any other
        unrecoverable error -- this error-string-not-exception
        convention is what specialists/base.py depends on via
        raw.startswith(...), and what acp_call_with_retry's failure
        detection relies on.
    """
    prompt_text = _messages_to_prompt_text(messages)
    if response_schema is not None:
        prompt_text = f"{prompt_text}\n\n{_build_schema_instruction(response_schema)}"

    role_key = _role_key_for(model, config, session_scope)

    try:
        prompt_resp, accumulated_text = _POOL.prompt_sync(role_key, config, model, prompt_text, timeout)
    except asyncio.TimeoutError:
        return f"[{model}] Error: ACP prompt timed out after {timeout}s."
    except FileNotFoundError as e:
        return f"[{model}] Error: could not spawn 'opencode acp' subprocess: {e}"
    except Exception as e:
        return f"[{model}] Error: {e}"

    if not accumulated_text.strip():
        return f"[{model}] Error: no agent_message_chunk text received (stopReason={getattr(prompt_resp, 'stopReason', None)!r})"

    return accumulated_text


def _build_retry_feedback_prompt(response_schema: Optional[dict[str, Any]], failure_reason: str) -> str:
    """
    Build the short follow-up prompt sent for a same-session retry
    attempt (see acp_call_with_retry). Deliberately does NOT resend the
    original messages -- the pooled ACP session already has the prior
    (malformed) turn in its own conversation history, since retries
    reuse the same session_scope/role_key. Only the specific failure
    reason is sent here; the schema instruction itself is NOT
    duplicated in this text -- acp_call already re-appends it via
    _build_schema_instruction whenever response_schema is passed (which
    acp_call_with_retry does on every attempt, including retries), so
    embedding it again here would send the schema twice in one prompt.
    Matches the design doc's Option A description ("the original schema
    instruction plus the specific parse/validation error message") --
    acp_call contributes the schema half, this function contributes the
    error half.
    """
    if response_schema is not None:
        return f"Your previous response was invalid: {failure_reason}\n\nCorrect your response accordingly."
    return (
        f"Your previous response was invalid: {failure_reason}\n\n"
        "Correct your response and output ONLY the corrected content, no other text."
    )


def _first_failure_reason(
    result: str,
    model: str,
    parse_and_validate_fn: Optional[Callable[[str], Any]],
) -> Optional[str]:
    """
    Classify a single acp_call result as success or failure for
    acp_call_with_retry's loop.

    Returns:
        None if `result` is a success (not an acp_call error string, and
        -- if parse_and_validate_fn was given -- it did not raise).
        Otherwise a human-readable failure reason string: either the
        acp_call error string itself (stripped of the "[model] Error:"
        prefix is NOT done here -- the raw error is descriptive enough
        as-is), or str(exception) from a failing parse_and_validate_fn.
    """
    if result.startswith(f"[{model}] Error:"):
        return result

    if parse_and_validate_fn is not None:
        try:
            parse_and_validate_fn(result)
        except Exception as e:
            return str(e)

    return None


def acp_call_with_retry(
    messages: list[dict],
    model: str,
    config: dict[str, Any],
    timeout: int = 60,
    response_schema: Optional[dict[str, Any]] = None,
    session_scope: Optional[dict] = None,
    parse_and_validate_fn: Optional[Callable[[str], Any]] = None,
    max_attempts: int = 3,
) -> str:
    """
    acp_call wrapped in Option A's retry-with-error-feedback loop (see
    the ACP-native rearchitecture design doc's "Recommended Approach" >
    Option A). Compensates for ACP having no token-level schema
    enforcement (unlike the now-removed chat_completions/responses
    providers' `response_format: json_schema, strict: true`): on
    failure, re-invokes acp_call in the SAME pooled session (same
    session_scope, so AcpSessionPool reuses the existing subprocess/ACP
    session) with the schema instruction plus the specific failure
    reason appended, up to `max_attempts` total tries. This lets the
    model see its own prior malformed output directly in the session's
    conversation history alongside the concrete error -- the strongest
    available self-correction signal, per the design doc's rationale for
    choosing Option A over fresh-session or parallel-attempt retries.

    Args:
        messages: Full list of {"role": ..., "content": ...} messages for
            the FIRST attempt only. Retry attempts send a short new
            message instead of resending these (see
            _build_retry_feedback_prompt) -- the same-session reuse
            means the prior turn is already in context.
        model: Model name string, forwarded to acp_call unchanged.
        config: Loaded config dict, forwarded to acp_call unchanged.
        timeout: Per-attempt prompt timeout in seconds, forwarded to
            every acp_call invocation (including retries).
        response_schema: Optional dict with keys "name" and "schema" (a
            JSON Schema object). Forwarded to acp_call on EVERY attempt
            (including retries) -- acp_call itself re-appends the schema
            instruction each time via _build_schema_instruction, so the
            schema is always restated alongside the failure reason on a
            retry without this function duplicating it. None (default)
            disables the schema instruction on every attempt.
        session_scope: Optional dict {"project": str, "role": str}
            forwarded unchanged to every acp_call invocation across all
            attempts -- this is what makes retries "same-session":
            identical session_scope means AcpSessionPool resolves the
            same role_key and thus the same pooled subprocess/session on
            every attempt (see _role_key_for). None falls back to
            acp_call's own (model, litellm_url) pooling key, which is
            equally stable across retries.
        parse_and_validate_fn: Optional callable taking the raw response
            string and returning anything (return value is discarded --
            only used to detect success/failure). Should raise on
            invalid input (e.g. a chained
            `lambda raw: DesignAgent._validate_and_build(DesignAgent._parse_json(raw), valid_languages)`)
            with a descriptive exception message, which becomes the
            failure reason appended to the next retry's feedback prompt.
            None (default) means only acp_call's own error-string
            convention is checked -- any non-error-string result is
            treated as success without further validation.
        max_attempts: Total attempts including the first (not additional
            retries on top of it). Must be >= 1.

    Returns:
        The successful raw response string on the first attempt that
        both (a) is not an acp_call error string and (b) passes
        parse_and_validate_fn (if given) -- same contract as acp_call
        itself, so callers still do their own parsing/construction
        afterward exactly as they do today. If every attempt fails, a
        single f"[{model}] Error: acp_call_with_retry failed after
        {max_attempts} attempts, last error: {last_error}" string is
        returned (never raises), preserving acp_call's
        error-string-not-exception convention.

    Raises:
        ValueError: if max_attempts < 1.
    """
    if max_attempts < 1:
        raise ValueError(f"max_attempts must be >= 1, got {max_attempts}")

    current_messages = messages
    last_error: Optional[str] = None

    for attempt in range(1, max_attempts + 1):
        result = acp_call(
            current_messages, model, config, timeout=timeout,
            response_schema=response_schema, session_scope=session_scope,
        )

        failure_reason = _first_failure_reason(result, model, parse_and_validate_fn)
        if failure_reason is None:
            return result

        last_error = failure_reason
        if attempt == max_attempts:
            break

        retry_prompt = _build_retry_feedback_prompt(response_schema, failure_reason)
        current_messages = [{"role": "user", "content": retry_prompt}]

    return (
        f"[{model}] Error: acp_call_with_retry failed after {max_attempts} attempts, "
        f"last error: {last_error}"
    )
