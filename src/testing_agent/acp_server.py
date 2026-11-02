"""Manual ACP endpoint for explicit external-testing delegation."""

from __future__ import annotations

import asyncio
import concurrent.futures
import shlex
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event
from typing import Any

REPO_SRC = str(Path(__file__).resolve().parents[1])
if REPO_SRC not in sys.path:
    sys.path.insert(0, REPO_SRC)

import acp
from acp.schema import AgentCapabilities, Implementation, InitializeResponse, NewSessionResponse, PromptResponse

from testing_agent.config import load_settings
from testing_agent.runner import AgentSettings, run_external_tests


AGENT_NAME = "agents-testing-agent"
AGENT_VERSION = "0.1.0"
PROMPT_USAGE = (
    "Use: run /absolute/path/to/testing-handoff.json; for existing-project handoffs add "
    "--project-root /absolute/path/to/project"
)


@dataclass
class _Session:
    session_id: str
    cwd: str
    cancel_event: Event = field(default_factory=Event)
    used: bool = False
    status: str = "created"
    outcome: str | None = None
    phase: str = "created"
    result_path: str | None = None
    error: str | None = None


class TestingAgent:
    def __init__(self, settings: AgentSettings | None = None) -> None:
        self.settings = settings or load_settings()
        self._sessions: dict[str, _Session] = {}
        self._conn: Any = None

    def on_connect(self, conn: Any) -> None:
        self._conn = conn

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
                    "agents-testing-agent": {
                        "runStatus": {
                            "method": "_agents-testing-agent/run/status",
                            "params": {"sessionId": "string"},
                        },
                        "manualDelegation": True,
                        "promptUsage": PROMPT_USAGE,
                    }
                }
            ),
            agent_info=Implementation(name=AGENT_NAME, version=AGENT_VERSION),
        )

    async def new_session(
        self,
        cwd: str,
        additional_directories: list[str] | None = None,
        mcp_servers: list[Any] | None = None,
        **kwargs: Any,
    ) -> NewSessionResponse:
        session_id = uuid.uuid4().hex
        self._sessions[session_id] = _Session(session_id, cwd)
        return NewSessionResponse(session_id=session_id)

    @staticmethod
    def _parse_prompt(text: str) -> tuple[Path, Path | None]:
        try:
            tokens = shlex.split(text)
        except ValueError as error:
            raise ValueError(f"Invalid delegation syntax: {error}") from error
        if len(tokens) not in {2, 4} or tokens[0].casefold() != "run":
            raise ValueError(PROMPT_USAGE)
        handoff = Path(tokens[1])
        if not handoff.is_absolute():
            raise ValueError("handoff path must be absolute")
        project_root: Path | None = None
        if len(tokens) == 4:
            if tokens[2] != "--project-root":
                raise ValueError(PROMPT_USAGE)
            project_root = Path(tokens[3])
            if not project_root.is_absolute():
                raise ValueError("--project-root must be absolute")
        return handoff, project_root

    async def prompt(self, session_id: str, prompt: list[Any], **kwargs: Any) -> PromptResponse:
        state = self._sessions.get(session_id)
        if state is None:
            raise acp.RequestError.invalid_params({"session_id": session_id, "reason": "unknown session"})
        if state.used:
            raise acp.RequestError.invalid_params({"session_id": session_id, "reason": "session already used for a prompt"})
        state.used = True
        text = " ".join(
            block.text for block in prompt
            if isinstance(block, acp.schema.TextContentBlock) and isinstance(block.text, str)
        ).strip()
        try:
            handoff, project_root = self._parse_prompt(text)
            session_cwd = Path(state.cwd)
            if not session_cwd.is_absolute() or handoff.parent.resolve(strict=True) != session_cwd.resolve(strict=True):
                raise ValueError("handoff must be directly inside the ACP session cwd")
        except (OSError, ValueError) as error:
            state.status = "failed"
            state.outcome = "invalid_prompt"
            state.error = str(error)
            await self._send(session_id, f"External testing delegation rejected: {error}\n")
            return PromptResponse(stop_reason="refusal")

        state.status = "running"
        state.phase = "snapshot"
        loop = asyncio.get_running_loop()

        def progress(message: str) -> None:
            lowered = message.casefold()
            phase = (
                "snapshot" if "snapshot" in lowered
                else "setup" if "install" in lowered or "container" in lowered or "network" in lowered
                else "testing" if "test" in lowered or "build" in lowered
                else state.phase
            )
            loop.call_soon_threadsafe(setattr, state, "phase", phase)
            future: concurrent.futures.Future[None] = asyncio.run_coroutine_threadsafe(
                self._send(session_id, message + "\n"), loop
            )
            try:
                future.result(timeout=10)
            except Exception:
                pass

        task = asyncio.create_task(asyncio.to_thread(
            run_external_tests,
            handoff,
            authorized_project_root=project_root,
            settings=self.settings,
            cancel_event=state.cancel_event,
            progress=progress,
        ))
        interrupted = False
        while True:
            try:
                result = await asyncio.shield(task)
                break
            except asyncio.CancelledError:
                interrupted = True
                state.cancel_event.set()
                state.phase = "cancelling"
            except Exception as error:
                state.status = "failed"
                state.phase = "completed"
                state.outcome = "setup_failed"
                state.error = str(error)
                await self._send(session_id, f"External testing could not start: {error}\n")
                return PromptResponse(stop_reason="refusal")
        state.outcome = str(result.get("outcome", "setup_failed"))
        state.status = "cancelled" if state.outcome == "cancelled" or interrupted else "completed"
        state.phase = "completed"
        snapshot = result.get("source_snapshot")
        source_hash = snapshot.get("sha256") if isinstance(snapshot, dict) else "unavailable"
        state.result_path = str(result.get("result_path", "")) or None
        await self._send(
            session_id,
            f"External testing outcome: {state.outcome}. Source SHA-256: {source_hash}. Result: {state.result_path or 'unavailable'}\n",
        )
        return PromptResponse(stop_reason="cancelled" if state.status == "cancelled" else "end_turn")

    async def cancel(self, session_id: str, **kwargs: Any) -> None:
        state = self._sessions.get(session_id)
        if state is not None:
            state.cancel_event.set()
            state.phase = "cancelling"

    async def ext_method(self, method: str, params: dict[str, Any]) -> dict[str, object]:
        if method != "agents-testing-agent/run/status":
            raise acp.RequestError.method_not_found(f"ext_method {method} not supported")
        session_id = params.get("sessionId", params.get("session_id"))
        if not isinstance(session_id, str) or session_id not in self._sessions:
            raise acp.RequestError.invalid_params({"sessionId": session_id, "reason": "unknown session"})
        state = self._sessions[session_id]
        return {
            "schema_version": 1,
            "session_id": session_id,
            "status": state.status,
            "phase": state.phase,
            "outcome": state.outcome,
            "cancel_requested": state.cancel_event.is_set(),
            "result_path": state.result_path,
            "error": state.error,
        }

    async def ext_notification(self, method: str, params: dict[str, Any]) -> None:
        return None

    async def close_session(self, session_id: str, **kwargs: Any) -> None:
        self._sessions.pop(session_id, None)

    async def _send(self, session_id: str, text: str) -> None:
        if self._conn is None:
            return
        try:
            await self._conn.session_update(
                session_id=session_id,
                update=acp.update_agent_message_text(text),
            )
        except Exception:
            pass


async def _main() -> None:
    await acp.run_agent(TestingAgent())


if __name__ == "__main__":
    asyncio.run(_main())
