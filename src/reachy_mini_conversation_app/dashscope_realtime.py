"""DashScope (Qwen3.5-Omni-Flash-Realtime) backend for the Reachy Mini conversation app.

This module provides a drop-in handler that connects the conversation app's
audio pipeline to Alibaba Cloud's DashScope Qwen-Omni-Realtime API via
WebSocket, bypassing the OpenAI SDK entirely.

Architecture
------------
BaseRealtimeHandler  ──uses──►  self.connection  (an OpenAI-compatible interface)

We provide ``DashScopeRealtimeConnection`` which wraps a raw ``websockets``
connection to DashScope and mimics the same attribute/event protocol that
the OpenAI SDK's ``AsyncRealtimeConnection`` exposes, so the base handler
works without modification.
"""

import asyncio
import base64
import json
import logging
import os
from types import SimpleNamespace
from typing import Any, ClassVar, Optional

import websockets
import websockets.asyncio.client

from reachy_mini_conversation_app.config import (
    config,
    get_default_voice_for_backend,
    get_available_voices_for_backend,
)
from reachy_mini_conversation_app.prompts import get_session_voice, get_session_instructions
from reachy_mini_conversation_app.base_realtime import (
    BaseRealtimeHandler,
    to_realtime_tools_config,
)
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies, get_active_tool_specs

from openai.types.realtime import (
    RealtimeAudioConfigParam,
    RealtimeAudioConfigInputParam,
    RealtimeAudioConfigOutputParam,
    RealtimeSessionCreateRequestParam,
)
from openai.types.realtime.realtime_audio_formats_param import AudioPCM
from openai.types.realtime.realtime_audio_input_turn_detection_param import ServerVad

logger = logging.getLogger(__name__)

# ── Backend identity ─────────────────────────────────────────────
DASHSCOPE_BACKEND = "dashscope"

DASHSCOPE_AVAILABLE_VOICES: list[str] = [
    "Tina",
    "Ethan",
    "Cherry",
    "Chelsie",
    "Serena",
    "Dylan",
    "Aiden",
]

DASHSCOPE_DEFAULT_VOICE = "Tina"

# ── Pricing (CNY per million tokens) ─────────────────────────────
AUDIO_IN_PER_1M = 27.0
AUDIO_OUT_PER_1M = 107.0
TEXT_IN_PER_1M = 3.3
TEXT_OUT_PER_1M = 20.0
IMAGE_IN_PER_1M = 3.3


# =====================================================================
# WebSocket adapter – makes DashScope look like OpenAI Realtime
# =====================================================================


class _NamespaceProxy:
    """Proxy that converts method calls into WebSocket JSON messages."""

    def __init__(self, conn: "DashScopeRealtimeConnection"):
        self._conn = conn


class _SessionProxy(_NamespaceProxy):
    async def update(self, *, session: dict[str, Any]) -> None:
        await self._conn._send({"type": "session.update", "session": session})


class _InputAudioBufferProxy(_NamespaceProxy):
    async def append(self, *, audio: str) -> None:
        await self._conn._send({"type": "input_audio_buffer.append", "audio": audio})

    async def commit(self) -> None:
        await self._conn._send({"type": "input_audio_buffer.commit"})

    async def clear(self) -> None:
        await self._conn._send({"type": "input_audio_buffer.clear"})


class _ConversationItemProxy:
    def __init__(self, conn: "DashScopeRealtimeConnection"):
        self._conn = conn

    async def create(self, *, item: dict[str, Any]) -> None:
        await self._conn._send({"type": "conversation.item.create", "item": item})


class _ConversationProxy(_NamespaceProxy):
    def __init__(self, conn: "DashScopeRealtimeConnection"):
        super().__init__(conn)
        self.item = _ConversationItemProxy(conn)


class _ResponseProxy(_NamespaceProxy):
    async def create(self, **kwargs: Any) -> None:
        msg: dict[str, Any] = {"type": "response.create"}
        if kwargs:
            msg["response"] = kwargs
        await self._conn._send(msg)

    async def cancel(self) -> None:
        await self._conn._send({"type": "response.cancel"})


class DashScopeRealtimeConnection:
    """Async context manager that wraps a DashScope WebSocket connection
    and presents the same interface as OpenAI's ``AsyncRealtimeConnection``.

    Usage::

        async with DashScopeRealtimeConnection(url, api_key, model) as conn:
            await conn.session.update(session={...})
            await conn.input_audio_buffer.append(audio=base64_pcm)
            async for event in conn:
                print(event.type)
    """

    def __init__(self, url: str, api_key: str, model: str = "qwen3.5-omni-flash-realtime"):
        self._url = url
        self._api_key = api_key
        self._model = model
        self._ws: Optional[websockets.asyncio.client.ClientConnection] = None
        self._event_queue: asyncio.Queue[Any] = asyncio.Queue()
        self._recv_task: Optional[asyncio.Task[None]] = None

        # Namespace proxies matching OpenAI SDK layout
        self.session = _SessionProxy(self)
        self.input_audio_buffer = _InputAudioBufferProxy(self)
        self.conversation = _ConversationProxy(self)
        self.response = _ResponseProxy(self)

    async def __aenter__(self) -> "DashScopeRealtimeConnection":
        headers = {"Authorization": f"Bearer {self._api_key}"}
        full_url = f"{self._url}?model={self._model}"
        self._ws = await websockets.connect(full_url, additional_headers=headers, ping_interval=20)
        self._recv_task = asyncio.create_task(self._recv_loop(), name="dashscope-recv")
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    async def close(self) -> None:
        if self._recv_task and not self._recv_task.done():
            self._recv_task.cancel()
            try:
                await self._recv_task
            except asyncio.CancelledError:
                pass
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass

    async def _send(self, msg: dict[str, Any]) -> None:
        if self._ws is None:
            raise RuntimeError("WebSocket not connected")
        await self._ws.send(json.dumps(msg))

    async def _recv_loop(self) -> None:
        """Read WebSocket frames and push them into the event queue."""
        assert self._ws is not None
        try:
            async for raw in self._ws:
                try:
                    data = json.loads(raw)
                    etype_dbg = data.get("type", "?")
                    logger.info("DashScope WS <<< %s", etype_dbg)
                except json.JSONDecodeError:
                    logger.warning("Non-JSON frame received: %s", raw[:200])
                    continue

                # Convert dict → SimpleNamespace so the base handler can
                # use attribute access (event.type, event.delta, …).
                event = self._dict_to_namespace(data)
                await self._event_queue.put(event)
        except websockets.ConnectionClosed as e:
            logger.info("DashScope WebSocket closed: %s", e)
        except asyncio.CancelledError:
            return
        except Exception as e:
            logger.error("DashScope recv loop error: %s", e)
        finally:
            # Sentinel so __aiter__ stops
            await self._event_queue.put(None)

    @staticmethod
    def _dict_to_namespace(d: Any) -> Any:
        """Recursively convert a dict to SimpleNamespace for attribute access."""
        if isinstance(d, dict):
            return SimpleNamespace(**{k: DashScopeRealtimeConnection._dict_to_namespace(v) for k, v in d.items()})
        if isinstance(d, list):
            return [DashScopeRealtimeConnection._dict_to_namespace(item) for item in d]
        return d

    def __aiter__(self) -> "DashScopeRealtimeConnection":
        return self

    async def __anext__(self) -> Any:
        event = await self._event_queue.get()
        if event is None:
            raise StopAsyncIteration
        return event


# =====================================================================
# Handler – plugs into BaseRealtimeHandler
# =====================================================================


class DashScopeRealtimeHandler(BaseRealtimeHandler):
    """Realtime handler backed by DashScope Qwen-Omni-Realtime.

    Drop-in replacement for ``OpenaiRealtimeHandler`` / ``HuggingFaceRealtimeHandler``.
    """

    BACKEND_PROVIDER: ClassVar[str] = DASHSCOPE_BACKEND
    SAMPLE_RATE: ClassVar[int] = 24000
    REFRESH_CLIENT_ON_RECONNECT: ClassVar[bool] = False
    AUDIO_INPUT_COST_PER_1M: ClassVar[float] = AUDIO_IN_PER_1M / 7.2   # rough USD
    AUDIO_OUTPUT_COST_PER_1M: ClassVar[float] = AUDIO_OUT_PER_1M / 7.2
    TEXT_INPUT_COST_PER_1M: ClassVar[float] = TEXT_IN_PER_1M / 7.2
    TEXT_OUTPUT_COST_PER_1M: ClassVar[float] = TEXT_OUT_PER_1M / 7.2
    IMAGE_INPUT_COST_PER_1M: ClassVar[float] = IMAGE_IN_PER_1M / 7.2

    DASHSCOPE_WS_URL = "wss://dashscope.aliyuncs.com/api-ws/v1/realtime"

    # ── Provider hooks required by BaseRealtimeHandler ────────────

    def _get_session_instructions(self) -> str:
        return get_session_instructions()

    def _get_session_voice(self, default: str | None = None) -> str:
        return get_session_voice(default)

    def _get_active_tool_specs(self) -> list[dict[str, Any]]:
        return get_active_tool_specs(self.deps)

    def _get_session_config(self, tool_specs: list[dict[str, Any]]) -> RealtimeSessionCreateRequestParam:
        """Build session config in OpenAI format (our adapter translates it)."""
        voice = self.get_current_voice()
        return RealtimeSessionCreateRequestParam(
            type="realtime",
            instructions=self._get_session_instructions(),
            audio=RealtimeAudioConfigParam(
                input=RealtimeAudioConfigInputParam(
                    format=AudioPCM(type="audio/pcm", rate=16000),
                    transcription={"model": "qwen3-asr-flash-realtime", "language": "zh"},
                    turn_detection=None,  # Local VAD handles turn detection
                ),
                output=RealtimeAudioConfigOutputParam(
                    format=AudioPCM(type="audio/pcm", rate=24000),
                    voice=voice,
                ),
            ),
            tools=to_realtime_tools_config(tool_specs),
            tool_choice="auto",
        )

    async def _build_realtime_client(self) -> Any:
        """Return a *dummy* client – we override ``_run_realtime_session``
        so the OpenAI client is never used."""
        return None  # type: ignore[return-value]

    # ── Override the session runner to use our WebSocket adapter ──

    # -- Local VAD: energy-based speech detection --
    _audio_frame_count: int = 0
    _lv_speech_active: bool = False
    _lv_silence_frames: int = 0
    _lv_speech_frames: int = 0
    _lv_total_speech_frames: int = 0
    # At 16kHz with ~320-sample frames = ~50 frames/sec
    _LV_SILENCE_FRAMES: int = 75    # 1.5 seconds of silence = speech ended
    _LV_MIN_SPEECH: int = 30        # 0.6 seconds minimum = real speech
    _LV_ENERGY_FLOOR: float = 0.08  # Initial floor, will adapt
    _lv_rms_log_interval: int = 0
    _lv_noise_floor: float = 0.02   # Adaptive noise floor estimate
    _lv_noise_alpha: float = 0.995  # EMA smoothing for noise floor (slow)
    _robot_speaking: bool = False   # True while robot audio is playing
    _post_speech_silence: int = 0   # Frames of silence after robot stops
    _POST_SPEECH_GRACE: int = 25    # 0.5s grace period after robot stops
    _wake_word_detected: bool = True  # Disabled wake word (always active) "Reachy" detected
    _WAKE_WORD: str = "Reachy"      # Wake word to listen for

    async def receive(self, frame):
        """Local VAD: analyze audio energy, trigger response after sufficient silence."""
        import numpy as np

        if self._audio_frame_count == 0:
            logger.info(
                "Local VAD active: silence_thresh=%.1fs, min_speech=%.1fs, energy_floor=%.0f",
                self._LV_SILENCE_FRAMES / 50.0,
                self._LV_MIN_SPEECH / 50.0,
                self._LV_ENERGY_FLOOR,
            )

        self._audio_frame_count += 1
        self._lv_rms_log_interval += 1

        # Calculate RMS energy from audio frame (use louder channel for stereo)
        _sr, audio_data = frame
        audio_f64 = audio_data.astype(np.float64)
        if audio_f64.ndim == 2:
            # Stereo: use the channel with higher RMS (closer to speaker)
            ch_rms = np.sqrt(np.mean(audio_f64 ** 2, axis=0))
            rms = float(np.max(ch_rms))
        else:
            rms = float(np.sqrt(np.mean(audio_f64 ** 2)))

        # Barge-in suppression: ignore mic input while robot is speaking
        if self._robot_speaking:
            self._post_speech_silence = 0
            return
        if self._post_speech_silence < self._POST_SPEECH_GRACE:
            self._post_speech_silence += 1
            return
        
        # Wake word mode: VAD always runs, but we only trigger response when wake word is active
        # (transcription events will set _wake_word_detected when "Reachy" is detected)

        # Adaptive noise floor: slowly track ambient level (only when not speaking)
        if not self._lv_speech_active and rms < 0.2:
            self._lv_noise_floor = (
                self._lv_noise_alpha * self._lv_noise_floor
                + (1 - self._lv_noise_alpha) * rms
            )
        # Adaptive threshold: 2x noise floor, minimum 0.02
        adaptive_floor = max(self._lv_noise_floor * 2.0, 0.02)
        is_speech = rms > adaptive_floor
        # Log RMS periodically for calibration
        if self._lv_rms_log_interval >= 50:  # every ~1 second
            self._lv_rms_log_interval = 0
            logger.info(
                "RMS=%.4f (adaptive=%.4f, noise=%.4f) | speech=%s | sil=%d/%d | f=%d",
                rms, adaptive_floor, self._lv_noise_floor,
                self._lv_speech_active,
                self._lv_silence_frames, self._LV_SILENCE_FRAMES,
                self._audio_frame_count,
            )

        if is_speech:
            if not self._lv_speech_active:
                self._lv_speech_active = True
                self._lv_total_speech_frames = 0
                self._lv_silence_frames = 0
                logger.info("Local VAD: speech START (rms=%.4f)", rms)
            self._lv_total_speech_frames += 1
            self._lv_speech_frames += 1
            self._lv_silence_frames = 0
        else:
            if self._lv_speech_active:
                self._lv_silence_frames += 1
                if self._lv_silence_frames >= self._LV_SILENCE_FRAMES:
                    # Silence long enough -> speech ended
                    dur = self._lv_total_speech_frames / 50.0
                    self._lv_speech_active = False
                    self._lv_silence_frames = 0
                    self._lv_speech_frames = 0

                    if self._lv_total_speech_frames >= self._LV_MIN_SPEECH:
                        if self._wake_word_detected:
                            logger.info(
                                "Local VAD: speech END (%.1fs) -> triggering response", dur
                            )
                            try:
                                if self.connection:
                                    await self.connection.input_audio_buffer.commit()
                                    await self.connection.response.create()
                            except Exception as exc:
                                logger.warning("Local VAD: trigger failed: %s", exc)
                        else:
                            logger.info(
                                "Local VAD: speech END (%.1fs) -> no wake word, ignoring", dur
                            )
                    else:
                        logger.info(
                            "Local VAD: speech END (%.1fs) -> too short, ignoring", dur
                        )
                    self._lv_total_speech_frames = 0

        # Forward audio to server for transcription
        await super().receive(frame)

    async def _run_realtime_session(self) -> None:
        """Manage a single DashScope realtime session."""
        tool_specs = self._get_active_tool_specs()
        logger.info(
            "Tools to be used in conversation: %s",
            [tool["name"] for tool in tool_specs],
        )

        api_key = os.getenv("DASHSCOPE_API_KEY", "")
        if not api_key:
            raise RuntimeError("DASHSCOPE_API_KEY environment variable is not set")

        model = getattr(config, "MODEL_NAME", "") or "qwen3.5-omni-flash-realtime"

        async with DashScopeRealtimeConnection(
            self.DASHSCOPE_WS_URL, api_key, model
        ) as conn:
            try:
                session_config = self._get_session_config(tool_specs)
                # Convert the OpenAI-style config dict to the DashScope wire format
                session_dict = self._translate_session_config(session_config)
                logger.info("Session config turn_detection=%s", session_dict.get("turn_detection", "NOT SET"))
                await conn.session.update(session=session_dict)
                logger.info(
                    "DashScope session initialized (model=%s, voice=%s)",
                    model,
                    self.get_current_voice(),
                )
                self._persist_credentials_if_needed()
            except Exception:
                logger.exception("DashScope session.update failed; aborting startup")
                raise

            self.input_transcript_chunks_by_item = __import__(
                "reachy_mini_conversation_app.base_realtime",
                fromlist=["InputTranscriptChunksByItem"],
            ).InputTranscriptChunksByItem()

            self.connection = conn  # type: ignore[assignment]
            try:
                self._connected_event.set()
            except Exception:
                pass

            response_sender_task: asyncio.Task[None] | None = None
            try:
                self.tool_manager.start_up(tool_callbacks=[self._handle_tool_result])
                response_sender_task = asyncio.create_task(
                    self._response_sender_loop(), name="response-sender"
                )

                async for event in self.connection:
                    await self._dispatch_event(event)

            finally:
                if response_sender_task and not response_sender_task.done():
                    response_sender_task.cancel()
                    try:
                        await response_sender_task
                    except asyncio.CancelledError:
                        pass
                self.connection = None
                try:
                    self._connected_event.clear()
                except Exception:
                    pass
                await self.tool_manager.shutdown()

    # ── Session config translation ──────────────────────────────

    @staticmethod
    def _translate_session_config(session: dict[str, Any]) -> dict[str, Any]:
        """Translate OpenAI-style session config to DashScope wire format.

        The DashScope Realtime API is largely OpenAI-compatible, but some
        fields need adjustment (e.g. audio format names, voice casing).
        """
        result = dict(session)

        # Use server_vad with 1.5s silence for wake word mode
        # Server will auto-detect speech end and return transcriptions
        result["turn_detection"] = {
            "type": "server_vad",
            "silence_duration_ms": 1500,
        }

        # DashScope expects plain format strings
        audio = result.get("audio", {})
        if isinstance(audio, dict):
            inp = audio.get("input", {})
            if isinstance(inp, dict) and "format" in inp:
                fmt = inp["format"]
                if isinstance(fmt, dict):
                    inp["format"] = "pcm"
            out = audio.get("output", {})
            if isinstance(out, dict) and "format" in out:
                fmt = out["format"]
                if isinstance(fmt, dict):
                    out["format"] = "pcm"

        # Ensure voice is title-case (DashScope voices are PascalCase)
        voice = result.get("audio", {}).get("output", {}).get("voice")
        if voice and isinstance(voice, str):
            result["audio"]["output"]["voice"] = voice[0].upper() + voice[1:]

        return result

    # ── Event dispatcher (mirrors base_realtime event handling) ──

    async def _dispatch_event(self, event: Any) -> None:
        """Route a DashScope event through the base handler's logic.

        We call the same helper methods the base class uses for each event
        type so cost tracking, transcript debouncing, movement, and chatbot
        updates all work correctly.
        """
        etype: str = getattr(event, "type", "")
        logger.debug("DashScope event: %s", etype)

        # ── User speech detection ──
        if etype == "input_audio_buffer.speech_started":
            self.is_idle_tool_call = False
            self._mark_activity("user_speech_started")
            self._turn_user_done_at = None
            self._turn_response_created_at = None
            self._turn_first_audio_at = None
            if self._clear_queue:
                self._clear_queue()
            self.deps.movement_manager.set_listening(True)

        elif etype == "input_audio_buffer.speech_stopped":
            self._mark_activity("user_speech_stopped")
            self.deps.movement_manager.set_listening(False)

        # ── Response lifecycle ──
        elif etype == "response.created":
            self._mark_activity("response_created")
            self._response_done_event.clear()
            self._response_started_or_rejected_event.set()
            if self._turn_user_done_at is not None and self._turn_response_created_at is None:
                import time as _time
                self._turn_response_created_at = _time.perf_counter()
                delta_ms = (self._turn_response_created_at - self._turn_user_done_at) * 1000
                logger.info("Turn latency: response.created %.0f ms after user transcript", delta_ms)

        elif etype == "response.done":
            self._response_done_event.set()
            self._response_started_or_rejected_event.set()
            # Reset wake word after response completes
            # Wake word disabled - always stay active
            self._wake_word_detected = True
            # Cost tracking
            resp = getattr(event, "response", None)
            usage = getattr(resp, "usage", None) if resp else None
            if usage:
                cost = self._compute_response_cost(usage)
                self.cumulative_cost += cost
                logger.debug("Cost: $%.4f | Cumulative: $%.4f", cost, self.cumulative_cost)

        # ── Audio output ──
        elif etype == "response.audio.delta":
            if not self._robot_speaking:
                self._robot_speaking = True
                logger.info("Robot started speaking (barge-in suppressed)")
            if self._turn_first_audio_at is None:
                import time as _time
                self._turn_first_audio_at = _time.perf_counter()
            delta_b64: str = getattr(event, "delta", "")
            if delta_b64:
                audio_bytes = base64.b64decode(delta_b64)
                import numpy as np
                pcm = np.frombuffer(audio_bytes, dtype=np.int16)
                await self.output_queue.put((self.output_sample_rate, pcm))

        elif etype == "response.audio.done":
            self._robot_speaking = False
            self._post_speech_silence = 0
            logger.info("Robot finished speaking (barge-in re-enabled)")

        # ── Transcript: model output ──
        elif etype == "response.audio_transcript.delta":
            self._mark_activity("assistant_transcription_delta")
            delta = getattr(event, "delta", "")
            if delta:
                from fastrtc import AdditionalOutputs
                await self.output_queue.put(
                    AdditionalOutputs({"role": "assistant_partial", "content": delta})
                )

        elif etype == "response.audio_transcript.done":
            transcript = getattr(event, "transcript", "")
            if transcript:
                from fastrtc import AdditionalOutputs
                await self.output_queue.put(
                    AdditionalOutputs({"role": "assistant", "content": transcript})
                )
                logger.info("[LLM] %s", transcript)

        # ── Transcript: user input ──
        elif etype == "conversation.item.input_audio_transcription.delta":
            self._mark_activity("user_transcription_delta")
            item_id = getattr(event, "item_id", "")
            delta = getattr(event, "delta", "") or ""
            
            # Check for wake word in standby mode
            if not self._wake_word_detected and not self._robot_speaking:
                if self._WAKE_WORD.lower() in delta.lower():
                    self._wake_word_detected = True
                    logger.info(f"Wake word '{self._WAKE_WORD}' detected! Listening for command...")
            
            input_transcript = self.input_transcript_chunks_by_item
            self._record_partial_transcript_delta(input_transcript, item_id, delta)

        elif etype == "conversation.item.input_audio_transcription.completed":
            transcript = getattr(event, "transcript", "")
            if transcript:
                # Wake word check: if not already activated, check for wake word
                if not self._wake_word_detected and not self._robot_speaking:
                    # Check for wake word (English or Chinese phonetic equivalents)
                    wake_variants = [
                        self._WAKE_WORD.lower(),
                        "reachy",
                        "rui qi", "ruiqi",
                        "围棋", "瑞奇", "锐奇", "瑞琪",  # Chinese phonetic equivalents
                        "ruì qí", "ruìqí",
                    ]
                    transcript_lower = transcript.lower()
                    if any(w in transcript_lower for w in wake_variants):
                        self._wake_word_detected = True
                        logger.info(f"Wake word detected in: '{transcript}'")
                    else:
                        # No wake word - cancel the response
                        logger.info(f"No wake word in: '{transcript}' - cancelling response")
                        try:
                            await self.connection._send({"type": "response.cancel"})
                        except Exception as exc:
                            logger.warning(f"Failed to cancel response: {exc}")
                        return
                
                from fastrtc import AdditionalOutputs
                await self.output_queue.put(
                    AdditionalOutputs({"role": "user", "content": transcript})
                )
                logger.info("[User] %s", transcript)
                self._mark_activity("user_transcription_done")
                import time as _time
                self._turn_user_done_at = _time.perf_counter()

        # ── Tool calling ──
        elif etype == "response.function_call_arguments.done":
            call_id = getattr(event, "call_id", "")
            name = getattr(event, "name", "")
            arguments_raw = getattr(event, "arguments", "{}")
            logger.info("Tool call: %s (id=%s)", name, call_id)
            try:
                arguments = json.loads(arguments_raw) if isinstance(arguments_raw, str) else arguments_raw
            except json.JSONDecodeError:
                arguments = {}
            self.tool_manager.submit(
                tool_name=name,
                tool_id=call_id,
                arguments=arguments,
            )

        # ── Errors ──
        elif etype == "error":
            err = getattr(event, "error", None)
            code = getattr(err, "code", "?") if err else "?"
            msg = getattr(err, "message", str(err)) if err else str(event)
            logger.error("DashScope error [%s]: %s", code, msg)

        # ── Session events (informational) ──
        elif etype in ("session.created", "session.updated"):
            logger.debug("Session event: %s", etype)

        # ── Rejection handling ──
        elif etype == "response.cancelled" or (etype == "error" and "active_response" in str(event)):
            self._last_response_rejected = True
            self._response_started_or_rejected_event.set()
