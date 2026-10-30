"""
src/pipeline/acp_server.py

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
    - Human-readable progress remains STDOUT-REDIRECT based:
      run_pipeline()/dispatch.py console output is captured via a
      contextvars-based sys.stdout proxy and streamed as ACP message
      chunks. A runner callback separately updates queryable structured
      progress through the advertised status extension.
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
    - Concurrency: prompt() calls are queued FIFO. One full run at a time
      is allowed per server process, including planning, dispatch,
      testing/retries, progress delivery, and runner teardown.

Stdout capture uses a `contextvars.ContextVar` sink and one
`_SessionStdoutProxy`. Pipeline output is routed to the owning session;
scheduled output deliveries are drained before the next queued run begins.

Manual invocation for testing:

    venv/bin/python src/pipeline/acp_server.py

This starts the agent listening on stdio (see acp.run_agent), exactly
as `opencode acp` does for the client side. Use
spike/acp_server_test_client.py to drive it end-to-end without needing
a real external ACP client.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextvars
import os
import re
import sys
import threading
import uuid
from collections import deque
from pathlib import Path
from typing import Any, Callable, Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC_ROOT = str(_REPO_ROOT / "src")
sys.path.insert(0, _SRC_ROOT)

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
from pipeline.project_paths import PROJECT_ROOT, resolve_project_path
from pipeline.project_context import validate_project_path
from pipeline.runner import load_config, run_pipeline

AGENT_NAME = "agents-pipeline"
AGENT_VERSION = "0.1.0"

# ── stdout capture plumbing ─────────────────────────────────────────────────

# Holds the current "session sink" callback (str -> None) for whichever
# prompt() call's context is active on the current thread/task. None
# means "no session in progress on this context" -- falls through to
# the real stdout. asyncio.to_thread propagates this context into the
# pipeline worker thread.
_SESSION_SINK: contextvars.ContextVar[Optional[Callable[[str], None]]] = contextvars.ContextVar(
    "acp_server_session_sink", default=None
)


class _RunTicket:
    __slots__ = ("session_id", "acquired", "cancelled")

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.acquired = False
        self.cancelled = False


class _RunQueue:
    """FIFO process-local queue; blocking waits happen off the event loop."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._waiting: deque[_RunTicket] = deque()
        self._active: Optional[_RunTicket] = None

    def enqueue(self, session_id: str) -> _RunTicket:
        ticket = _RunTicket(session_id)
        with self._condition:
            self._waiting.append(ticket)
            self._condition.notify_all()
        return ticket

    def position(self, ticket: _RunTicket) -> Optional[int]:
        with self._condition:
            try:
                return list(self._waiting).index(ticket) + 1
            except ValueError:
                return None

    async def acquire(self, ticket: _RunTicket) -> None:
        waiter = asyncio.create_task(asyncio.to_thread(self._wait_and_claim, ticket))
        try:
            claimed = await asyncio.shield(waiter)
        except asyncio.CancelledError:
            self._cancel_waiter(ticket)
            try:
                claimed = await asyncio.shield(waiter)
            except asyncio.CancelledError:
                claimed = False
            if claimed:
                self.release(ticket)
            raise
        if not claimed:
            raise asyncio.CancelledError

    def _wait_and_claim(self, ticket: _RunTicket) -> bool:
        with self._condition:
            while not ticket.cancelled:
                if self._active is None and self._waiting and self._waiting[0] is ticket:
                    self._waiting.popleft()
                    self._active = ticket
                    ticket.acquired = True
                    return True
                self._condition.wait()
            return False

    def _cancel_waiter(self, ticket: _RunTicket) -> None:
        with self._condition:
            ticket.cancelled = True
            try:
                self._waiting.remove(ticket)
            except ValueError:
                pass
            self._condition.notify_all()

    def release(self, ticket: _RunTicket) -> None:
        with self._condition:
            if self._active is ticket:
                self._active = None
                ticket.acquired = False
                self._condition.notify_all()


_RUN_QUEUE = _RunQueue()


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


def _install_stdout_proxy() -> _SessionStdoutProxy:
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
    proxy = _SessionStdoutProxy(current)
    sys.stdout = proxy
    return proxy


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

    __slots__ = (
        "session_id", "cwd", "cancelled", "used", "mode", "project_path",
        "progress", "progress_lock", "queue_ticket"
    )

    def __init__(self, session_id: str, cwd: str) -> None:
        self.session_id = session_id
        self.cwd = cwd
        self.cancelled = False
        self.used = False
        self.mode = "new_project"
        self.project_path: Optional[str] = None
        self.progress: dict[str, Any] = {
            "schema_version": 1,
            "session_id": session_id,
            "run_id": None,
            "status": "created",
              "phase": "created",
              "outcome": None,
              "validation_status": "pending",
              "validation_warnings": 0,
              "validation_limitations": [],
              "external_testing_status": "pending",
            "cancel_requested": False,
            "queue_position": None,
            "tasks_total": 0,
            "tasks_started": 0,
            "tasks_completed": 0,
            "tasks_blocked": 0,
            "current_task": None,
            "attempt_round": 0,
            "max_attempts": None,
              "mode": "new_project",
            "project_path": None,
        }
        self.progress_lock = threading.Lock()
        self.queue_ticket: Optional[_RunTicket] = None


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
            agent_capabilities=AgentCapabilities(
                field_meta={
                    "agents-pipeline": {
                        "runStatus": {
                            "method": "_agents-pipeline/run/status",
                            "params": {"sessionId": "string"},
                        },
                        "sessionConfig": {
                            "method": "_agents-pipeline/session/configure",
                            "params": {
                                "sessionId": "string",
                                "mode": "existing_project",
                                "projectPath": "absolute server-side path",
                            },
                        },
                    }
                }
            ),
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
            self._set_progress(
                state,
                status="failed",
                phase="completed",
                outcome="invalid_prompt",
                validation_status="not_completed",
                external_testing_status="pending",
            )
            await self._send_message(session_id, "⚠️  No text content found in prompt (v1 only reads TextContentBlock).")
            return PromptResponse(stop_reason="refusal")

        label = _slugify_label(text)
        loop = asyncio.get_running_loop()
        ticket = _RUN_QUEUE.enqueue(session_id)
        state.queue_ticket = ticket
        self._set_progress(state, status="queued", phase="queued", outcome=None)
        deliveries: list[concurrent.futures.Future[None]] = []
        deliveries_lock = threading.Lock()

        def _sink(data: str) -> None:
            if not data:
                return
            future = asyncio.run_coroutine_threadsafe(
                self._send_message(session_id, data, chunk=True), loop
            )
            with deliveries_lock:
                deliveries.append(future)

        async def _drain_deliveries() -> None:
            with deliveries_lock:
                pending = list(deliveries)
                deliveries.clear()
            if pending:
                drain_task = asyncio.gather(
                    *(asyncio.wrap_future(future) for future in pending),
                    return_exceptions=True,
                )
                while not drain_task.done():
                    try:
                        await asyncio.shield(drain_task)
                    except asyncio.CancelledError:
                        state.cancelled = True

        async def _wait_for_pipeline(task: asyncio.Task[Optional[bool]]) -> tuple[Optional[bool], bool]:
            interrupted = False
            while True:
                try:
                    return await asyncio.shield(task), interrupted
                except asyncio.CancelledError:
                    interrupted = True
                    state.cancelled = True

        acquired = False
        token: Optional[contextvars.Token[Optional[Callable[[str], None]]]] = None
        installed_proxy: Optional[_SessionStdoutProxy] = None
        pipeline_task: Optional[asyncio.Task[Optional[bool]]] = None
        success: Optional[bool] = None
        interrupted = False
        response: PromptResponse
        try:
            await self._send_message(session_id, "⏳ Pipeline queued.")
            await _RUN_QUEUE.acquire(ticket)
            acquired = True
            self._set_progress(state, status="running", phase="starting")
            installed_proxy = _install_stdout_proxy()
            token = _SESSION_SINK.set(_sink)
            pipeline_task = asyncio.create_task(
                asyncio.to_thread(
                    self._run_pipeline_safely,
                    text,
                    label,
                    lambda update: self._apply_pipeline_progress(state, update),
                    state.project_path,
                )
            )
            success, interrupted = await _wait_for_pipeline(pipeline_task)
            await _drain_deliveries()
            if state.project_path is not None and success is not True:
                await self._send_message(
                    session_id,
                    "⚠️ Existing-project edits are in place; partial changes may remain. "
                    "No rollback was performed. See the run's changes.json for the change summary.\n",
                )
            outcome = self._progress_outcome(state)
            if success is None:
                handoff_failed = state.progress.get("testing_handoff_status") == "failed"
                self._set_progress(
                    state,
                    status="failed",
                    phase="completed",
                    outcome=(
                        "failed"
                        if handoff_failed or state.project_path is not None
                        else "error"
                    ),
                    validation_status="not_completed",
                    external_testing_status="pending",
                )
                if handoff_failed:
                    await self._send_message(
                        session_id,
                        "❌ Pipeline request failed; required testing handoff artifacts are incomplete.\n",
                    )
                response = PromptResponse(stop_reason="refusal")
            elif state.cancelled or interrupted:
                self._set_progress(state, status="cancelled", phase="completed", outcome=outcome or "cancelled")
                await self._send_message(
                    session_id,
                    "\n🛑 Session was cancelled (best-effort; run may have completed anyway).\n",
                )
                response = PromptResponse(stop_reason="cancelled")
            else:
                outcome = outcome or (
                    "validation_passed_external_testing_pending" if success else "failed"
                )
                if outcome in {"passed", "passed_with_test_failures"}:
                    outcome = "validation_passed_external_testing_pending"
                self._set_progress(
                    state,
                    status="completed" if success else "failed",
                    phase="completed",
                    outcome=outcome,
                )
                if outcome == "validation_passed_external_testing_pending":
                    summary = (
                        "✅ Static validation passed; external runtime testing is pending. "
                        "Review the testing handoff for limitations."
                    )
                elif state.progress.get("testing_handoff_status") == "failed":
                    summary = (
                        "❌ Pipeline failed; required testing handoff artifacts are incomplete."
                    )
                else:
                    summary = "❌ Pipeline validation or execution failed"
                await self._send_message(session_id, f"\n{summary}\n")
                response = PromptResponse(stop_reason="end_turn")
        except asyncio.CancelledError:
            interrupted = True
            state.cancelled = True
            self._set_progress(state, status="cancelled", phase="completed", outcome="cancelled")
            response = PromptResponse(stop_reason="cancelled")
        except Exception as error:
            self._set_progress(
                state,
                status="failed",
                phase="completed",
                outcome="error",
                validation_status="not_completed",
                external_testing_status="pending",
            )
            await self._send_message(session_id, f"⚠️  Pipeline request failed: {error}\n")
            response = PromptResponse(stop_reason="refusal")
        finally:
            try:
                if pipeline_task is not None and not pipeline_task.done():
                    success, worker_interrupted = await _wait_for_pipeline(pipeline_task)
                    interrupted = interrupted or worker_interrupted
                if token is not None:
                    _SESSION_SINK.reset(token)
                if installed_proxy is not None and sys.stdout is installed_proxy:
                    sys.stdout = installed_proxy._real
                await _drain_deliveries()
            finally:
                if acquired:
                    _RUN_QUEUE.release(ticket)
                state.queue_ticket = None
        return response

    @staticmethod
    def _set_progress(state: _SessionState, **updates: Any) -> None:
        with state.progress_lock:
            state.progress.update(updates)

    @staticmethod
    def _progress_outcome(state: _SessionState) -> Optional[str]:
        with state.progress_lock:
            outcome = state.progress.get("outcome")
            return outcome if isinstance(outcome, str) else None

    @staticmethod
    def _apply_pipeline_progress(state: _SessionState, update: dict[str, object]) -> None:
        with state.progress_lock:
            state.progress.update(update)
            if "task" in update:
                task = update["task"]
                if isinstance(task, dict) and task.get("status") == "running":
                    state.progress["current_task"] = dict(task)
                else:
                    state.progress["current_task"] = None

    def _status_snapshot(self, state: _SessionState) -> dict[str, Any]:
        with state.progress_lock:
            snapshot = dict(state.progress)
            task = snapshot.get("current_task")
            if isinstance(task, dict):
                snapshot["current_task"] = dict(task)
        ticket = state.queue_ticket
        snapshot["queue_position"] = _RUN_QUEUE.position(ticket) if ticket is not None else None
        return snapshot

    def _run_pipeline_safely(
        self,
        user_input: str,
        label: str,
        progress: Callable[[dict[str, object]], None],
        existing_project_path: Optional[str] = None,
    ) -> Optional[bool]:
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
            if existing_project_path is None:
                return run_pipeline(self.config, user_input=user_input, label=label, progress=progress)
            return run_pipeline(
                self.config,
                user_input=user_input,
                label=label,
                progress=progress,
                existing_project_path=existing_project_path,
            )
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
            self._set_progress(state, cancel_requested=True)

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
        if method == "agents-pipeline/session/configure":
            session_id = params.get("sessionId", params.get("session_id"))
            if not isinstance(session_id, str):
                raise acp.RequestError.invalid_params({"sessionId": "required string"})
            state = self._sessions.get(session_id)
            if state is None:
                raise acp.RequestError.invalid_params({"sessionId": session_id, "reason": "unknown session"})
            if state.used or state.progress["status"] != "created":
                raise acp.RequestError.invalid_params(
                    {"sessionId": session_id, "reason": "session configuration is only allowed before prompt"}
                )
            if state.project_path is not None:
                raise acp.RequestError.invalid_params(
                    {"sessionId": session_id, "reason": "session is already configured"}
                )
            mode = params.get("mode")
            if mode != "existing_project":
                raise acp.RequestError.invalid_params({"mode": "must be 'existing_project'"})
            runs_root = os.path.realpath(
                resolve_project_path(self.config["pipeline"]["runs_dir"])
            )
            try:
                project_path = validate_project_path(
                    params.get("projectPath", params.get("project_path")),
                    runs_root,
                )
            except ValueError as error:
                raise acp.RequestError.invalid_params({"projectPath": str(error)}) from error
            state.mode = mode
            state.project_path = project_path
            self._set_progress(state, mode=mode, project_path=project_path)
            return {"sessionId": session_id, "mode": mode, "projectPath": project_path}

        if method != "agents-pipeline/run/status":
            raise acp.RequestError.method_not_found(f"ext_method {method} not supported")
        session_id = params.get("sessionId", params.get("session_id"))
        if not isinstance(session_id, str):
            raise acp.RequestError.invalid_params({"sessionId": "required string"})
        state = self._sessions.get(session_id)
        if state is None:
            raise acp.RequestError.invalid_params({"sessionId": session_id, "reason": "unknown session"})
        return self._status_snapshot(state)

    async def ext_notification(self, method: str, params: dict[str, Any]) -> None:
        return None


async def _main() -> None:
    agent = PipelineAgent()
    await acp.run_agent(agent)


if __name__ == "__main__":
    asyncio.run(_main())
