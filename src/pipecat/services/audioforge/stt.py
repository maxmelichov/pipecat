#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""audioforge speech-to-text service.

audioforge (https://github.com/maxmelichov/nvidia-audio-models, ``audioforge/serve.py``) is a
self-hosted WebSocket server that runs NVIDIA's streaming FastConformer ASR with learned VAD and
end-of-turn heads plus NVIDIA Streaming Sortformer diarization on one 80 ms clock, CPU-only. One
server session per pipeline is shared by three Pipecat pieces through an :class:`AudioforgeHub`:

- :class:`AudioforgeSTTService` owns the socket. Server ``partial`` events become
  :class:`~pipecat.frames.frames.InterimTranscriptionFrame`, ``final`` events become
  :class:`~pipecat.frames.frames.TranscriptionFrame` with ``finalized=True``;
  ``frame.metadata["audioforge"]`` carries ``{"t", "speaker", "kind"}`` (``speaker`` is the
  diarizer column of the primary speaker, ``t`` the server's audio time in seconds).
- :class:`~pipecat.audio.vad.audioforge_vad.AudioforgeVADAnalyzer` reads the server's per-frame
  VAD probability from the hub and runs Pipecat's own start/stop hysteresis on it.
- :class:`~pipecat.audio.turn.audioforge_turn.AudioforgeTurnAnalyzer` ends the user turn on the
  server's ``turn_end`` events, which are decided on the diarizer's primary-speaker track (other
  speakers do not delay them) and cut the transcript at exactly the same point.

The server decides the turn boundaries and sends each final together with its ``turn_end``, so
there is no separate speech-end-to-final wait: ``supports_ttfs`` is False.

Protocol framing comes from the ``audioforge-client`` package (``pip install "pipecat-ai[audioforge]"``).
"""

import asyncio
import bisect
import json
import time
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass, field
from typing import Any

from loguru import logger
from websockets.protocol import State

from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    Frame,
    InterimTranscriptionFrame,
    StartFrame,
    TranscriptionFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.settings import STTSettings
from pipecat.services.stt_service import WebsocketSTTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.types import NOT_GIVEN, NotGiven, assert_given

try:
    from audioforge_client import (
        AGENT_END_MODES,
        ENROLL_MODES,
        FINAL_SOURCES,
        NUM_SPEAKERS,
        POLICIES,
        Enrolled,
        Final,
        FrameEvent,
        LanguageEvent,
        Partial,
        Ready,
        ServerError,
        Stats,
        TurnEnd,
        Unknown,
        config_message,
        parse_message,
    )
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use audioforge, you need to `pip install pipecat-ai[audioforge]`.")
    raise Exception(f"Missing module: {e}")


@dataclass
class ReceivedTurnEnd:
    """A server ``turn_end`` event with the time it arrived.

    Parameters:
        event: The parsed server event (``t`` is the decision time in audio seconds,
            ``policy`` the rule that fired, ``p`` the head probability if any,
            ``silence_ms`` the primary speaker's silence at the decision).
        perf: ``time.perf_counter()`` at arrival.
        wall: ``time.time()`` at arrival.
    """

    event: TurnEnd
    perf: float
    wall: float

    @property
    def t(self) -> float:
        """Decision time, in seconds of audio since the session started."""
        return self.event.t

    @property
    def policy(self) -> str:
        """The server policy that produced the decision."""
        return self.event.policy

    @property
    def p(self) -> float | None:
        """The turn head's probability at the decision, if the policy used it."""
        return self.event.p

    @property
    def silence_ms(self) -> int:
        """The primary speaker's silence (ms) at the decision."""
        return self.event.silence_ms


class AudioforgeHub:
    """Per-session state shared by the STT service (which owns the socket) and the analyzers.

    Plain attributes, written only from the event loop (the STT's receive task). The VAD analyzer
    reads ``vad`` from Pipecat's analyzer thread, a single float read. ``policy`` is the turn
    policy the STT asks the server for when it was not given one explicitly (the turn analyzer
    sets it).
    """

    def __init__(self, policy: str | None = None):
        """Initialize the hub.

        Args:
            policy: Turn policy requested by an analyzer; None until one registers.
        """
        self.policy = policy
        self.final_source = "stream"
        self._turn_listeners: list[Callable[[ReceivedTurnEnd], None]] = []
        self._frame_listeners: list[Callable[[float, float], None]] = []
        self._reset_listeners: list[Callable[[], None]] = []
        self.sample_rate = 16000
        self.reset()

    def reset(self):
        """Forget the session state (a new server session starts at audio time 0).

        Listeners are kept and told, so analyzers drop decisions of the old session.
        """
        self.ready: Ready | None = None
        self.stats: Stats | None = None
        self.vad = 0.0
        self.eot: float | None = None
        self.speakers: list[float] = [0.0] * NUM_SPEAKERS
        self.primary: int | None = None
        self.frame_t = 0.0
        self.n_frames = 0
        self.frame_vad: list[tuple[float, float]] = []
        self.turn_ends: list[ReceivedTurnEnd] = []
        self.finals: list[Final] = []
        self.offline_final_t: set[float] = set()
        self.partials = 0
        self.enrolled_column: int | None = None
        self.language: str | None = None
        self.errors: list[ServerError] = []
        self.enroll_sent = 0
        self._sent = [0]
        self._sent_perf = [0.0]
        for fn in list(self._reset_listeners):
            fn()

    def add_turn_listener(self, fn: Callable[[ReceivedTurnEnd], None]):
        """Register ``fn(turn_end)`` for every server ``turn_end`` event."""
        self._turn_listeners.append(fn)

    def add_frame_listener(self, fn: Callable[[float, float], None]):
        """Register ``fn(t, vad)`` for every server ``frame`` event."""
        self._frame_listeners.append(fn)

    def add_reset_listener(self, fn: Callable[[], None]):
        """Register ``fn()`` for every session reset (connect / reconnect)."""
        self._reset_listeners.append(fn)

    def note_sent(self, n_samples: int, sample_rate: int, perf: float | None = None):
        """Record that ``n_samples`` more samples were sent (for latency accounting)."""
        self.sample_rate = sample_rate
        self._sent.append(self._sent[-1] + n_samples)
        self._sent_perf.append(time.perf_counter() if perf is None else perf)
        if len(self._sent) > 120000:
            del self._sent[:20000], self._sent_perf[:20000]

    def sent_perf(self, t: float) -> float | None:
        """``perf_counter`` time at which the audio up to ``t`` seconds had been sent."""
        i = bisect.bisect_left(self._sent, int(round(t * self.sample_rate)))
        return self._sent_perf[i] if 0 < i < len(self._sent) else None

    def on_frame(self, f: FrameEvent):
        """Apply one server ``frame`` event."""
        self.vad = f.vad
        self.eot = f.eot
        if f.speakers:
            self.speakers = list(f.speakers)
        self.primary = f.primary
        self.frame_t = f.t
        self.n_frames += 1
        self.frame_vad.append((f.t, f.vad))
        if len(self.frame_vad) > 120000:
            del self.frame_vad[:20000]
        for fn in list(self._frame_listeners):
            fn(f.t, f.vad)

    def on_turn_end(self, ev: TurnEnd, perf: float | None = None) -> ReceivedTurnEnd:
        """Apply one server ``turn_end`` event and notify the turn listeners."""
        rec = ReceivedTurnEnd(ev, time.perf_counter() if perf is None else perf, time.time())
        self.turn_ends.append(rec)
        for fn in list(self._turn_listeners):
            fn(rec)
        return rec


@dataclass
class AudioforgeSTTSettings(STTSettings):
    """Settings for AudioforgeSTTService.

    Changing any of them at runtime reconnects: the server takes its configuration once per
    session, and a new session restarts the diarizer's speaker identities.

    Parameters:
        turn_policy: Server turn policy: ``timeout`` (silence of the diarizer's primary
            speaker >= ``timeout_ms``; the measured default), ``hybrid_dyn`` / ``hybrid_silero``
            (turn head OR an any-speaker Silero silence; fewer cut-ins, more dead air), ``hybrid``,
            ``head``, ``both`` or ``timeout_quiet``.
        timeout_ms: Silence (ms) for the timeout policy.
        eot_threshold: Turn-head threshold for the head-based policies; None leaves the
            server's default for the policy.
    """

    turn_policy: str | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    timeout_ms: int | NotGiven = field(default_factory=lambda: NOT_GIVEN)
    eot_threshold: float | None | NotGiven = field(default_factory=lambda: NOT_GIVEN)


class AudioforgeSTTService(WebsocketSTTService):
    """Streaming speech-to-text on a self-hosted audioforge server.

    Pushes interim and final transcription frames cut at the server's turn decisions and feeds
    an :class:`AudioforgeHub` that the audioforge VAD and turn analyzers read. Use
    :meth:`create_vad_analyzer` and :meth:`create_turn_analyzer` to build analyzers on the
    service's hub.

    Event handlers (in addition to the STT service ones):

    - ``on_language_detected(service, language: str, confidence: float)``: the server's ``--lid``
      language identification decided or changed the spoken language.
    """

    Settings = AudioforgeSTTSettings
    _settings: Settings

    def __init__(
        self,
        *,
        url: str = "ws://127.0.0.1:8765",
        hub: AudioforgeHub | None = None,
        sample_rate: int | None = None,
        enroll: str | None = None,
        final_source: str = "stream",
        end_timeout: float = 5.0,
        settings: Settings | None = None,
        **kwargs,
    ):
        """Initialize the audioforge STT service.

        Args:
            url: The audioforge server (``python -m audioforge.serve``).
            hub: Shared state for the analyzers. Created if None.
            sample_rate: Audio sample rate. The server accepts any rate in its config message.
            enroll: Primary-speaker enrollment trigger; the server must run with the same
                ``--enroll`` mode. None (default): nothing is sent, the server's default binding.
                ``"after_agent_arm"`` / ``"after_agent"``: send ``agent_end`` at every
                :class:`~pipecat.frames.frames.BotStoppedSpeakingFrame`, so the server binds the
                primary speaker to whoever answers the bot. ``"explicit"``: send nothing until
                :meth:`enroll` is called.
            final_source: With a server running ``--final-asr`` (a second, offline model that
                re-transcribes each turn): ``"stream"`` (default) pushes the streaming finals and
                ignores the offline ones; ``"offline"`` pushes the offline finals instead and the
                turn analyzer holds the turn until the turn's offline final has arrived, so the
                LLM sees the better transcript. Servers without ``--final-asr`` are unaffected.
            end_timeout: Seconds to wait for the server's flush final and stats on
                :class:`~pipecat.frames.frames.EndFrame`.
            settings: Runtime-updatable settings (turn policy, timeout, head threshold).
            **kwargs: Additional arguments passed to the STT service.
        """
        default_settings = self.Settings(
            model="audioforge",
            language=None,
            turn_policy="timeout",
            timeout_ms=1000,
            eot_threshold=None,
        )
        if settings is not None:
            default_settings.apply_update(settings)

        super().__init__(sample_rate=sample_rate, settings=default_settings, **kwargs)

        if enroll is not None and enroll not in ENROLL_MODES:
            raise ValueError(f"enroll must be None or one of {ENROLL_MODES}")
        if final_source not in FINAL_SOURCES:
            raise ValueError(f"final_source must be one of {FINAL_SOURCES}")
        if self._settings.turn_policy not in POLICIES:
            raise ValueError(f"turn_policy must be one of {POLICIES}")

        self._url = url
        self.hub = hub or AudioforgeHub()
        self.hub.final_source = final_source
        self._enroll = enroll
        self._final_source = final_source
        self._end_timeout = float(end_timeout)
        self._receive_task: asyncio.Task | None = None
        self._stats_event: asyncio.Event | None = None
        self._pending_controls: list[str] = []
        self._detected_language: Language | None = None

        self._register_event_handler("on_language_detected")

    # ------------------------------------------------------------------ convenience

    def create_vad_analyzer(self, **kwargs):
        """Create an :class:`~pipecat.audio.vad.audioforge_vad.AudioforgeVADAnalyzer` on this hub."""
        from pipecat.audio.vad.audioforge_vad import AudioforgeVADAnalyzer

        return AudioforgeVADAnalyzer(self.hub, **kwargs)

    def create_turn_analyzer(self, **kwargs):
        """Create an :class:`~pipecat.audio.turn.audioforge_turn.AudioforgeTurnAnalyzer` on this hub."""
        from pipecat.audio.turn.audioforge_turn import AudioforgeTurnAnalyzer

        return AudioforgeTurnAnalyzer(self.hub, **kwargs)

    @property
    def turn_policy(self) -> str:
        """The policy requested from the server (the settings' value)."""
        return str(self._settings.turn_policy)

    @property
    def supports_ttfs(self) -> bool:
        """False: each final arrives with its turn_end, there is no speech-end-to-final wait."""
        return False

    def can_generate_metrics(self) -> bool:
        """Whether this service generates processing metrics.

        Returns:
            False. The server's decisions carry no TTFB in Pipecat's sense.
        """
        return False

    # ------------------------------------------------------------------ lifecycle

    async def start(self, frame: StartFrame):
        """Start the service and connect to the server.

        Args:
            frame: The start frame.
        """
        await super().start(frame)
        if self.hub.policy and self.hub.policy != self.turn_policy and self.turn_policy != "both":
            logger.warning(
                f"{self}: server turn_policy {self.turn_policy!r} will not emit the "
                f"{self.hub.policy!r} turn_end events the turn analyzer waits for"
            )
        await self._connect()

    async def stop(self, frame: EndFrame):
        """Flush the server (last final + stats) and stop.

        Args:
            frame: The end frame.
        """
        await self._flush_and_wait()
        await super().stop(frame)

    async def cancel(self, frame: CancelFrame):
        """Cancel immediately without flushing.

        Args:
            frame: The cancel frame.
        """
        await super().cancel(frame)

    async def _update_settings(self, delta: Settings) -> dict[str, Any]:
        """Apply a settings delta and reconnect if anything changed.

        Args:
            delta: A settings delta.

        Returns:
            Dict mapping changed field names to their previous values.
        """
        changed = await super()._update_settings(delta)
        if "turn_policy" in changed and self._settings.turn_policy not in POLICIES:
            logger.warning(f"{self}: unknown turn_policy {self._settings.turn_policy!r}; keeping")
            self._settings.turn_policy = changed["turn_policy"]
            changed.pop("turn_policy")
        if changed:
            await self._request_reconnect()
        return changed

    async def _flush_and_wait(self):
        """Send ``{"type": "end"}`` and wait for the flush final and stats (the server then closes)."""
        ws = self._websocket
        if ws is None or ws.state is not State.OPEN:
            return
        # The server closes after "stats": that close is not an error to reconnect on.
        self._disconnecting = True
        try:
            await ws.send(json.dumps({"type": "end"}))
            if self._stats_event is not None:
                await asyncio.wait_for(self._stats_event.wait(), self._end_timeout)
        except TimeoutError:
            logger.warning(f"{self}: no stats message within {self._end_timeout}s after end")
        except Exception as e:
            logger.warning(f"{self}: end flush failed: {e}")

    # ------------------------------------------------------------------ enrollment

    async def _send_control(self, kind: str) -> bool:
        ws = self._websocket
        if ws is None or ws.state is not State.OPEN:
            # Not connected yet: sent right after the next config message.
            self._pending_controls.append(kind)
            return True
        try:
            await ws.send(json.dumps({"type": kind}))
            self.hub.enroll_sent += 1
            return True
        except Exception as e:
            logger.warning(f"{self}: {kind} send failed: {e}")
            return False

    async def enroll(self) -> bool:
        """Ask the server to enroll the primary speaker on the next utterance.

        Only for ``enroll="explicit"`` (server ``--enroll explicit``); call it after the user
        was asked to speak.

        Returns:
            Whether the request was sent or queued.
        """
        if self._enroll != "explicit":
            logger.warning(f"{self}: enroll() called with enroll={self._enroll!r}; nothing sent")
            return False
        return await self._send_control("enroll")

    async def agent_end(self) -> bool:
        """Tell the server the bot's speech ended (``--enroll after_agent[_arm]``).

        Sent automatically at every ``BotStoppedSpeakingFrame`` when ``enroll`` is one of the
        agent-end modes; call it yourself for bot speech Pipecat does not see.

        Returns:
            Whether the message was sent or queued.
        """
        return await self._send_control("agent_end")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process a frame; the bot's end of speech arms the server's enrollment.

        Args:
            frame: The frame to process.
            direction: The direction of frame processing.
        """
        if isinstance(frame, BotStoppedSpeakingFrame) and self._enroll in AGENT_END_MODES:
            await self._send_control("agent_end")
        await super().process_frame(frame, direction)

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        """Send audio to the server.

        Args:
            audio: Raw int16 PCM bytes.

        Yields:
            None. Transcriptions arrive over the websocket.
        """
        ws = self._websocket
        if ws is not None and ws.state is State.OPEN:
            try:
                await ws.send(audio)
                self.hub.note_sent(len(audio) // 2, self.sample_rate or self.hub.sample_rate)
            except Exception as e:
                logger.warning(f"{self}: send failed: {e}")
        yield None

    # ------------------------------------------------------------------ websocket

    async def _connect(self):
        await super()._connect()
        await self._connect_websocket()
        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

    async def _disconnect(self):
        await super()._disconnect()
        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None
        await self._disconnect_websocket()

    async def _connect_websocket(self):
        try:
            if self._websocket and self._websocket.state is State.OPEN:
                return
            logger.debug(f"{self}: connecting to {self._url} (turn_policy {self.turn_policy})")
            self.hub.reset()
            self._stats_event = asyncio.Event()
            ws = await self._websocket_connect(self._url, max_size=2**22, ping_interval=None)
            self._websocket = ws
            # Before StartFrame the service has no pipeline rate yet: the server default then.
            sample_rate = self.sample_rate or self.hub.sample_rate
            cfg = config_message(
                turn_policy=self.turn_policy,
                timeout_ms=int(assert_given(self._settings.timeout_ms)),
                eot_threshold=assert_given(self._settings.eot_threshold),
                sample_rate=sample_rate,
            )
            self.hub.sample_rate = sample_rate
            await ws.send(json.dumps(cfg))
            for kind in self._pending_controls:
                await ws.send(json.dumps({"type": kind}))
                self.hub.enroll_sent += 1
            self._pending_controls = []
            await self._call_event_handler("on_connected")
        except Exception as e:
            self._websocket = None
            await self.push_error(
                error_msg=f"Unable to connect to audioforge at {self._url}: {e}", exception=e
            )

    async def _disconnect_websocket(self):
        try:
            if self._websocket:
                await self._websocket.close()
        except Exception as e:
            logger.warning(f"{self}: error closing websocket: {e}")
        finally:
            if self._websocket is not None:
                self._websocket = None
                await self._call_event_handler("on_disconnected")

    async def _receive_messages(self):
        ws = self._websocket
        if ws is None:
            return
        async for raw in ws:
            if isinstance(raw, (bytes, bytearray)):
                continue
            for event in parse_message(raw):
                await self.handle_server_event(event)

    async def handle_server_event(self, event: Any):
        """Map one parsed server event to hub updates and Pipecat frames.

        Public so tests can feed canned events.

        Args:
            event: An ``audioforge_client`` event (``FrameEvent``, ``Partial``, ``TurnEnd``,
                ``Final``, ``Ready``, ``Enrolled``, ``LanguageEvent``, ``Stats``, ``ServerError``
                or ``Unknown``).
        """
        if isinstance(event, FrameEvent):
            self.hub.on_frame(event)
        elif isinstance(event, Partial):
            self.hub.partials += 1
            text = event.text.strip()
            if text:
                frame = InterimTranscriptionFrame(
                    text=text,
                    user_id=self._user_id,
                    timestamp=time_now_iso8601(),
                    language=self._detected_language,
                    result=event.raw,
                )
                frame.metadata["audioforge"] = {
                    "t": event.t,
                    "speaker": self.hub.primary,
                    "kind": "partial",
                }
                await self.push_frame(frame)
        elif isinstance(event, TurnEnd):
            self.hub.on_turn_end(event)
        elif isinstance(event, Final):
            self.hub.finals.append(event)
            if event.is_offline:
                self.hub.offline_final_t.add(event.t)
            use = event.source is None or (event.source == "stream") == (
                self._final_source == "stream"
            )
            text = event.text.strip()
            if text and use:
                await self.emit_stt_usage_metrics()
                frame = TranscriptionFrame(
                    text=text,
                    user_id=self._user_id,
                    timestamp=time_now_iso8601(),
                    language=self._detected_language,
                    result=event.raw,
                    finalized=True,
                )
                frame.metadata["audioforge"] = {
                    "t": event.t,
                    "speaker": event.speaker,
                    "kind": "final",
                    "source": event.source,
                }
                await self.push_frame(frame)
        elif isinstance(event, Ready):
            self.hub.ready = event
            if self._enroll and event.enroll != self._enroll:
                logger.warning(
                    f"{self}: enroll={self._enroll!r} but the server runs --enroll "
                    f"{event.enroll or 'dominant'!r}; its trigger messages will be ignored"
                )
        elif isinstance(event, Enrolled):
            self.hub.enrolled_column = event.column
        elif isinstance(event, LanguageEvent):
            self.hub.language = event.language
            try:
                self._detected_language = Language(event.language)
            except ValueError:
                self._detected_language = None
            await self._call_event_handler("on_language_detected", event.language, event.confidence)
        elif isinstance(event, Stats):
            self.hub.stats = event
            if self._stats_event is not None:
                self._stats_event.set()
        elif isinstance(event, ServerError):
            self.hub.errors.append(event)
            text = f"{self}: audioforge server error {event.message}"
            if event.fatal:
                logger.error(text)
                await self.push_error(error_msg=text)
            else:
                logger.warning(text)
        elif isinstance(event, Unknown):
            logger.warning(f"{self}: unknown server message type {event.kind!r}")
