"""
pipeline/acp_server.py

ACP Agent/server entrypoint for the pipeline (Phase 5 of the ACP-native
rearchitecture — see
dev-plans/agents/features/2026-09-14-acp-native-master-architecture.md).

This is the mirror image of spike/acp_spike.py: that spike implements
the ACP CLIENT role (spawning `opencode acp` and driving it). This
module implements the ACP AGENT role — it IS the thing an external
caller spawns and drives via `initialize` -> `new_session` -> `prompt`,
the same way this app's own specialists.providers.acp_client spawns and
drives `opencode acp` today.

v1 scope (confirmed, do not expand without checking the design doc /
flagging first):

  - Prompt contract is FREE-TEXT ONLY. The prompt's text content is
    passed straight through as `user_input` to
    pipeline.runner.run_pipeline(config, user_input=text, label=...) --
    i.e. this mirrors runner.py's REPL/decompose() path, NOT the
    handoff-package (ingest.py) path. Handoff-package support is
    deferred to a later version.
  - Progress streaming is STDOUT-REDIRECT based, not a threaded
    progress-callback system. run_pipeline()/dispatch.py are untouched
    -- they still just print(). This module captures that output via a
    contextvars-based sys.stdout proxy (see _SessionStdoutProxy below)
    and forwards each captured line as a session_update
    agent_message_chunk notification.
  - Cancellation is WEAK/BEST-EFFORT. cancel(session_id) sets a flag on
    the session's state dict; run_pipeline()/dispatch.py do NOT check
    this flag mid-run (that would require threading a cancellation flag
    through dispatch.py's specialist-call loop, out of scope for v1).
    The flag is only consulted after run_pipeline() returns, to decide
    whether prompt() reports "cancelled" instead of "end_turn" -- a
    request already-in-flight when cancel() arrives WILL run to
    completion.
  - Session model is 1 ACP session = 1 pipeline run. No multi-turn
    steering, no session reuse across multiple prompts. A second
    prompt() call against an already-used session_id is rejected.
  - Concurrency: prompt() calls are NOT serialized behind a global lock
    -- multiple sessions can run concurrently, each on its own executor
    thread, each with correctly-isolated captured stdout (see the
    concurrency-safety note below). See also the KNOWN CONCURRENCY
    HAZARD note below regarding specialists.providers.acp_client's
    shared session-pool teardown, which this module does NOT attempt to
    fix.

Concurrency-safety of stdout capture:

  sys.stdout is process-global, so naively swapping it per-request would
  let concurrent runs interleave/corrupt each other's captured output.
  This module solves that with a `contextvars.ContextVar` holding a
  per-session "sink" callback, plus a single stdout proxy object
  (`_SessionStdoutProxy`, installed once at import time) whose
  `write()` checks the contextvar and routes to that session's sink if
  set, or falls through to the real stdout otherwise. `prompt()` drives
  the blocking `run_pipeline()` call via `asyncio.to_thread`, which
  (per Python's documented behavior, and verified empirically for this
  change) automatically copies the calling task's `contextvars.Context`
  into the worker thread -- so the contextvar set inside `prompt()`
  correctly follows execution into run_pipeline()'s worker thread, and
  two concurrent prompt() calls each see only their own session's sink,
  with zero cross-session line leakage. This was chosen over adding a
  new `output_sink` parameter to run_pipeline() because it requires
  ZERO changes to pipeline/runner.py or pipeline/dispatch.py -- both
  files are untouched by this change, so no regression risk was
  introduced to either module or their existing tests.

KNOWN CONCURRENCY HAZARD (flagged, not fixed here): every
run_pipeline() call unconditionally calls
specialists.providers.acp_client.teardown() in its own `finally` block,
which stops AcpSessionPool's single shared background asyncio event
loop/thread entirely -- not just that run's own pooled ACP sessions.
Two concurrent prompt() calls (each internally calling run_pipeline())
will race: whichever run finishes first tears down the shared loop out
from under the other run, which is very likely to break or hang that
still-in-flight run's own specialist calls. This is a pre-existing
property of specialists/providers/acp_client.py's teardown() design,
not something introduced by this module -- fixing it would require
changing acp_client.py's pool lifecycle (e.g. reference-counting active
runs before stopping the shared loop), which is out of scope for this
phase. Concurrent prompt() calls are therefore NOT safe in practice
today, even though this module itself does not add any additional
serialization -- this is an explicit, flagged limitation, not a
silently-accepted one.

Manual invocation for testing:

    venv/bin/python pipeline/acp_server.py

This starts the agent listening on stdio (see acp.run_agent), exactly
as `opencode acp` does for the client side. Use
spike/acp_server_test_client.py to drive it end-to-end without needing
a real external ACP client.
"""

from __future__ import annotations

import asyncio
import contextvars
import os
import re
import sys
import uuid
from typing import Any, Callable, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import acp
from acp.schema import (
    AgentCapabilities,
    Implementation,
    InitializeResponse,
    NewSessionResponse,
    PromptResponse,
    TextContentBlock,
)

from pipeline.master import MasterPlanError
from pipeline.runner import load_config, run_pipeline

AGENT_NAME = "agents-pipeline"
AGENT_VERSION = "0.1.0"

# ── stdout capture plumbing ─────────────────────────────────────────────────

# Holds the current "session sink" callback (str -> None) for whichever
# prompt() call's context is active on the current thread/task. None
# means "no session in progress on this context" -- falls through to
# the real stdout. See module docstring's "Concurrency-safety of stdout
# capture" section for why this is safe across asyncio.to_thread calls.
_SESSION_SINK: contextvars.ContextVar[Optional[Callable[[str], None]]] = contextvars.ContextVar(
    "acp_server_session_sink", default=None
)


class _SessionStdoutProxy:
    """
    Drop-in stdout replacement installed once at import time (module
    level, process-global -- there is exactly one of these per process,
    matching sys.stdout's own process-global nature). Every write()
    checks _SESSION_SINK: if a sink is set on the calling
    context (i.e. we're inside a prompt() call's run_pipeline()
    execution), the write is routed to that session's sink instead of
    the real stdout, so pipeline print() output never reaches the
    server process's own stdout (which ACP itself uses for JSON-RPC
    framing -- writing pipeline progress there directly would corrupt
    the protocol stream). Falls through to the real underlying stdout
    for anything printed outside of an active session context (e.g.
    import-time errors, or this module run standalone without any
    prompt() in flight).
    """

    def __init__(self, real: Any) -> None:
        self._real = real

    def write(self, data: str) -> int:
        sink = _SESSION_SINK.get()
        if sink is not None:
            sink(data)
            return len(data)
        return self._real.write(data)

    def flush(self) -> None:
        self._real.flush()

    def isatty(self) -> bool:
        return False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


def _install_stdout_proxy() -> None:
    """
    Install (or refresh) the module-global stdout proxy. Unwraps any
    previously-installed proxy first, so this is safe to call more than
    once against a process whose sys.stdout has since been replaced out
    from under a stale proxy (e.g. pytest's per-test stdout-capture
    fixture installs a fresh capture object before each test -- calling
    this again in a test fixture must re-point the proxy's `_real` at
    that fresh object, not silently keep wrapping the previous test's
    now-stale one).
    """
    current = sys.stdout
    if isinstance(current, _SessionStdoutProxy):
        current = current._real
    sys.stdout = _SessionStdoutProxy(current)


_install_stdout_proxy()


def _slugify_label(text: str, max_len: int = 60) -> str:
    """Derive a short run-dir-friendly label from free-text prompt
    content, for run_pipeline()'s `label` argument. run_pipeline()
    itself defaults label to the full user_input when work_items is
    given as None and label is omitted -- but the full prompt text can
    be very long, and label also gets used verbatim as the manifest's
    "prompt" field, so a short slug is nicer for a server-driven call
    where the caller may not think about labeling at all."""
    stripped = text.strip()
    if not stripped:
        return "acp-prompt"
    collapsed = re.sub(r"\s+", " ", stripped)
    if len(collapsed) > max_len:
        collapsed = collapsed[:max_len].rsplit(" ", 1)[0] or collapsed[:max_len]
    return collapsed


def _extract_prompt_text(prompt: list[Any]) -> str:
    """
    Extract free text from an ACP prompt's list of content blocks.

    v1 scope: only TextContentBlock entries are consulted (joined with
    a single space); ImageContentBlock/AudioContentBlock/
    ResourceContentBlock/EmbeddedResourceContentBlock entries are
    ignored entirely -- this pipeline has no multimodal or
    file-attachment handling on the prompt-ingestion path, matching the
    "free-text only" v1 scope decision in the module docstring.
    """
    parts = []
    for block in prompt:
        text = getattr(block, "text", None)
        if isinstance(block, TextContentBlock) or (text is not None and getattr(block, "type", None) == "text"):
            if text:
                parts.append(text)
    return " ".join(parts).strip()


class _SessionState:
    """Per-session bookkeeping. One ACP session == one pipeline run (see
    module docstring's "Session model"). `cancelled` is best-effort only
    -- see module docstring's "Cancellation" section."""

    __slots__ = ("session_id", "cwd", "cancelled", "used")

    def __init__(self, session_id: str, cwd: str) -> None:
        self.session_id = session_id
        self.cwd = cwd
        self.cancelled = False
        self.used = False


class PipelineAgent:
    """
    Implements acp.interfaces.Agent: the ACP-server-facing wrapper around
    this app's existing synchronous pipeline (pipeline/runner.py). See
    module docstring for full scope/limitations.
    """

    def __init__(self) -> None:
        # Config is loaded ONCE at agent construction (process start).
        # A running server does not pick up config.json changes without
        # a restart -- this is a deliberate simplicity choice for v1,
        # matching the same load-once-per-process behavior
        # pipeline/runner.py's REPL already has.
        self.config, self.config_path = load_config()
        self._sessions: dict[str, _SessionState] = {}
        self._conn: Optional[Any] = None  # set via on_connect(); acp.Client-shaped

    # ── connection lifecycle ────────────────────────────────────────────

    def on_connect(self, conn: Any) -> None:
        """
        Called synchronously by acp.AgentSideConnection at construction
        time, handing us the connection object we use to send
        session_update notifications back to the client for the
        lifetime of this process.
        """
        self._conn = conn

    # ── required Agent protocol methods ─────────────────────────────────

    async def initialize(
        self,
        protocol_version: int,
        client_capabilities: Any = None,
        client_info: Any = None,
        **kwargs: Any,
    ) -> InitializeResponse:
        return InitializeResponse(
            protocol_version=protocol_version,
            agent_capabilities=AgentCapabilities(),
            agent_info=Implementation(name=AGENT_NAME, version=AGENT_VERSION),
        )

    async def new_session(
        self,
        cwd: str,
        additional_directories: Optional[list[str]] = None,
        mcp_servers: Optional[list[Any]] = None,
        **kwargs: Any,
    ) -> NewSessionResponse:
        session_id = uuid.uuid4().hex
        self._sessions[session_id] = _SessionState(session_id, cwd)
        return NewSessionResponse(session_id=session_id)

    async def prompt(
        self,
        session_id: str,
        prompt: list[Any],
        **kwargs: Any,
    ) -> PromptResponse:
        state = self._sessions.get(session_id)
        if state is None:
            raise acp.RequestError.invalid_params({"session_id": session_id, "reason": "unknown session"})

        if state.used:
            # 1 ACP session = 1 pipeline run (v1 scope) -- reject reuse
            # rather than silently steering/queueing a second run onto
            # the same session id.
            raise acp.RequestError.invalid_params(
                {"session_id": session_id, "reason": "session already used for a prompt (1 session = 1 run in v1)"}
            )
        state.used = True

        text = _extract_prompt_text(prompt)
        if not text:
            await self._send_message(session_id, "⚠️  No text content found in prompt (v1 only reads TextContentBlock).")
            return PromptResponse(stop_reason="refusal")

        label = _slugify_label(text)

        loop = asyncio.get_running_loop()

        def _sink(data: str) -> None:
            # Called from the executor thread that's running
            # run_pipeline(). Schedule the actual (async) send back onto
            # the event loop rather than awaiting here directly -- this
            # function itself must stay synchronous, since it's invoked
            # from sys.stdout.write().
            if not data:
                return
            asyncio.run_coroutine_threadsafe(self._send_message(session_id, data, chunk=True), loop)

        token = _SESSION_SINK.set(_sink)
        try:
            success = await asyncio.to_thread(self._run_pipeline_safely, text, label)
        finally:
            _SESSION_SINK.reset(token)

        if state.cancelled:
            await self._send_message(session_id, "\n🛑 Session was cancelled (best-effort; run may have completed anyway).\n")
            return PromptResponse(stop_reason="cancelled")

        if success is None:
            # Unexpected exception path -- _run_pipeline_safely already
            # sent an error session_update before returning None here.
            return PromptResponse(stop_reason="refusal")

        summary = "✅ Pipeline passed" if success else "❌ Pipeline failed"
        await self._send_message(session_id, f"\n{summary}\n")

        return PromptResponse(stop_reason="end_turn")

    def _run_pipeline_safely(self, user_input: str, label: str) -> Optional[bool]:
        """
        Runs on an executor thread (via asyncio.to_thread). Wraps
        run_pipeline() so unexpected exceptions (anything NOT already
        handled internally by run_pipeline() itself, e.g. MasterPlanError
        which run_pipeline() catches and converts to a `False` return)
        are caught here rather than propagating out of the executor call
        and crashing the ACP connection. Returns True/False as
        run_pipeline() would, or None if an unexpected exception was
        caught (caller distinguishes None from False to know an error
        session_update was already sent).
        """
        try:
            return run_pipeline(self.config, user_input=user_input, label=label)
        except MasterPlanError as e:
            # run_pipeline() already catches this internally and returns
            # False -- this branch is defensive/documentation only (see
            # its docstring), kept in case that contract ever changes.
            print(f"⚠️  MasterAgent failed to produce a valid plan: {e}\n")
            return False
        except Exception as e:  # noqa: BLE001 - deliberately broad: must not crash the ACP connection
            print(f"⚠️  Unexpected error during pipeline run: {e}\n")
            return None

    async def _send_message(self, session_id: str, text: str, chunk: bool = False) -> None:
        """
        Send a session_update agent_message_chunk notification back to
        the connected client, if a connection exists (on_connect() has
        fired) and the target session is still known. Swallows errors --
        a failed notification send should not crash the pipeline run
        itself, and there is nothing more disruptive we could usefully
        do at this point besides logging to the real stdout.
        """
        if self._conn is None:
            return
        try:
            await self._conn.session_update(
                session_id=session_id,
                update=acp.update_agent_message_text(text),
            )
        except Exception:
            pass

    # ── cancellation (best-effort, see module docstring) ────────────────

    async def cancel(self, session_id: str, **kwargs: Any) -> None:
        """
        Sets a best-effort cancellation flag on the session, if it
        exists. Per the module docstring's "Cancellation" section, this
        does NOT preemptively stop an in-flight run_pipeline() call --
        pipeline/dispatch.py has no cancellation-check hook in v1. The
        flag is only consulted by prompt() after run_pipeline() returns,
        to report stop_reason="cancelled" instead of "end_turn"/
        "refusal". Silently no-ops for an unknown session_id (matches
        ACP's CancelNotification being a notification, not a
        request -- there is no response to return an error via).
        """
        state = self._sessions.get(session_id)
        if state is not None:
            state.cancelled = True

    # ── optional Agent protocol methods: minimal/default handling ───────

    async def load_session(
        self,
        cwd: str,
        session_id: str,
        mcp_servers: Optional[list[Any]] = None,
        additional_directories: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> None:
        # Not supported: AgentCapabilities.load_session defaults to
        # False, so a spec-compliant client should never call this --
        # but per the interface's optional-method contract, respond
        # with a clear error if it does.
        raise acp.RequestError.method_not_found("load_session not supported in v1")

    async def list_sessions(
        self, cwd: Optional[str] = None, cursor: Optional[str] = None, **kwargs: Any
    ) -> Any:
        from acp.schema import ListSessionsResponse, SessionInfo

        sessions = [
            SessionInfo(session_id=sid, cwd=state.cwd, title=None)
            for sid, state in self._sessions.items()
            if cwd is None or state.cwd == cwd
        ]
        return ListSessionsResponse(sessions=sessions)

    async def set_session_mode(self, session_id: str, mode_id: str, **kwargs: Any) -> None:
        raise acp.RequestError.method_not_found("set_session_mode not supported in v1")

    async def set_config_option(
        self, config_id: str, session_id: str, value: Any, **kwargs: Any
    ) -> None:
        raise acp.RequestError.method_not_found("set_config_option not supported in v1")

    async def authenticate(self, method_id: str, **kwargs: Any) -> None:
        # AgentCapabilities.auth defaults to {} / auth_methods defaults
        # to [] on InitializeResponse -- no auth methods are advertised,
        # so a spec-compliant client should never call this.
        raise acp.RequestError.method_not_found("authenticate not supported in v1")

    async def fork_session(
        self,
        session_id: str,
        cwd: str,
        additional_directories: Optional[list[str]] = None,
        mcp_servers: Optional[list[Any]] = None,
        **kwargs: Any,
    ) -> Any:
        raise acp.RequestError.method_not_found("fork_session not supported in v1")

    async def resume_session(
        self,
        session_id: str,
        cwd: str,
        additional_directories: Optional[list[str]] = None,
        mcp_servers: Optional[list[Any]] = None,
        **kwargs: Any,
    ) -> Any:
        raise acp.RequestError.method_not_found("resume_session not supported in v1")

    async def close_session(self, session_id: str, **kwargs: Any) -> None:
        self._sessions.pop(session_id, None)
        return None

    async def ext_method(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        raise acp.RequestError.method_not_found(f"ext_method {method} not supported in v1")

    async def ext_notification(self, method: str, params: dict[str, Any]) -> None:
        return None


async def _main() -> None:
    agent = PipelineAgent()
    await acp.run_agent(agent)


if __name__ == "__main__":
    asyncio.run(_main())
