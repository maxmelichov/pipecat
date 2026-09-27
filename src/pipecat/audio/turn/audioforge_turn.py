#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Turn analyzer driven by the audioforge server's end-of-turn decisions.

The audioforge server (see :mod:`pipecat.services.audioforge.stt`) decides end of turn on the
diarizer's primary-speaker track and sends a ``turn_end`` event followed by the turn's ``final``
transcript. This analyzer turns those events into
:class:`~pipecat.audio.turn.base_turn_analyzer.EndOfTurnState` for
:class:`~pipecat.turns.user_stop.TurnAnalyzerUserTurnStopStrategy`:

- A received ``turn_end`` of the analyzer's policy latches until Pipecat ends the turn
  (:meth:`AudioforgeTurnAnalyzer.clear`).
- With ``wait_for_silence`` (default) the analyzer reports ``COMPLETE`` only while Pipecat's VAD
  hears no speech. Pipecat starts LLM inference the moment an analyzer reports ``COMPLETE`` but its
  user turn controller refuses to end the turn while the VAD hears speech, so without the gate the
  bot would start answering while someone is still audible.
- The latch is dropped once the server's VAD has reported speech on ``resume_ms`` of audio after
  the decision time: someone kept or started talking after the decision, so it is stale. This rule
  runs on the server's frame clock, so it does not depend on socket timing.

Pipecat's turn controller treats whoever is audible as "the user". On multi-party audio someone
else talking right after the server's ``turn_end`` therefore either holds the turn until the VAD
stops or makes the decision stale, and the turn then waits for the server's next ``turn_end``.
"""

from loguru import logger

from pipecat.audio.turn.base_turn_analyzer import BaseTurnAnalyzer, BaseTurnParams, EndOfTurnState
from pipecat.metrics.metrics import MetricsData

try:
    from audioforge_client import FRAME_MS, POLICIES
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use audioforge, you need to `pip install pipecat-ai[audioforge]`.")
    raise Exception(f"Missing module: {e}")

from pipecat.services.audioforge.stt import AudioforgeHub, ReceivedTurnEnd


class AudioforgeTurnParams(BaseTurnParams):
    """Configuration parameters for the audioforge turn analyzer.

    Parameters:
        policy: Which server ``turn_end`` events end the turn: ``timeout`` (default; the plain
            silence timeout on the primary speaker), ``hybrid_dyn`` / ``hybrid_silero`` /
            ``hybrid`` (the server's head-or-silence rules), ``head`` (the turn head alone) or
            ``both`` (any event). The STT service requests the same policy from the server unless
            it was given one explicitly, so the finals are cut where this analyzer ends turns.
        wait_for_silence: Report ``COMPLETE`` only while Pipecat's VAD hears no speech.
        resume_ms: Milliseconds of server-side speech after a decision that make it stale
            (0 disables).
    """

    policy: str = "timeout"
    wait_for_silence: bool = True
    resume_ms: int = 240


class AudioforgeTurnAnalyzer(BaseTurnAnalyzer):
    """End of turn from the audioforge server's ``turn_end`` events.

    Build it with :meth:`AudioforgeSTTService.create_turn_analyzer
    <pipecat.services.audioforge.stt.AudioforgeSTTService.create_turn_analyzer>` or pass the
    service's ``hub``, and plug it into
    :class:`~pipecat.turns.user_stop.TurnAnalyzerUserTurnStopStrategy`.
    """

    def __init__(
        self,
        hub: AudioforgeHub,
        *,
        sample_rate: int | None = None,
        params: AudioforgeTurnParams | None = None,
    ):
        """Initialize the analyzer.

        Args:
            hub: The hub of the :class:`~pipecat.services.audioforge.stt.AudioforgeSTTService`
                that owns the server session.
            sample_rate: Optional fixed sample rate.
            params: Analyzer parameters.

        Raises:
            ValueError: If ``params.policy`` is not a server policy.
        """
        super().__init__(sample_rate=sample_rate)
        self._params = params or AudioforgeTurnParams()
        if self._params.policy not in POLICIES:
            raise ValueError(f"policy must be one of {POLICIES}")
        self._hub = hub
        if hub.policy is None:
            hub.policy = self._params.policy
        self._pending: ReceivedTurnEnd | None = None
        self._resume_frames = 0
        self._speech = False
        self.received: list[ReceivedTurnEnd] = []
        self.discarded: list[ReceivedTurnEnd] = []
        hub.add_turn_listener(self._on_turn_end)
        hub.add_frame_listener(self._on_frame)
        hub.add_reset_listener(self._on_reset)

    def _on_reset(self):
        self._pending = None
        self._resume_frames = 0

    def _on_turn_end(self, ev: ReceivedTurnEnd):
        if self._params.policy not in ("both", ev.policy):
            return
        self.received.append(ev)
        self._pending = ev
        self._resume_frames = 0
        # Frames of the same batch can precede the event: count speech already past the decision.
        for t, vad in reversed(self._hub.frame_vad):
            if t <= ev.t + 1e-6:
                break
            self._resume_frames += vad >= 0.5
        self._maybe_discard()

    def _on_frame(self, t: float, vad: float):
        if self._pending is not None and t > self._pending.t + 1e-6 and vad >= 0.5:
            self._resume_frames += 1
            self._maybe_discard()

    def _maybe_discard(self):
        if (
            self._pending is not None
            and self._params.resume_ms > 0
            and self._resume_frames * FRAME_MS >= self._params.resume_ms
        ):
            logger.debug(f"{self}: speech after turn_end t={self._pending.t}; dropping it")
            self.discarded.append(self._pending)
            self._pending = None

    @property
    def pending(self) -> ReceivedTurnEnd | None:
        """The latched server decision waiting for Pipecat to end the turn, if any."""
        return self._pending

    @property
    def speech_triggered(self) -> bool:
        """Whether speech was seen since the last :meth:`clear`."""
        return self._speech

    @property
    def params(self) -> AudioforgeTurnParams:
        """The analyzer parameters."""
        return self._params

    def _final_ready(self) -> bool:
        """With ``final_source="offline"`` on a ``--final-asr`` server: the turn's offline final arrived."""
        hub = self._hub
        if hub.final_source == "stream" or hub.ready is None or not hub.ready.final_asr:
            return True
        return self._pending is not None and any(
            abs(t - self._pending.t) < 1e-6 for t in hub.offline_final_t
        )

    def append_audio(self, buffer: bytes, is_speech: bool) -> EndOfTurnState:
        """Append audio and report whether the server's latched decision can be applied.

        Args:
            buffer: Audio data (unused: the decision comes from the server).
            is_speech: Whether Pipecat's VAD hears speech in it.

        Returns:
            ``COMPLETE`` when a decision is latched, the VAD gate allows it and the turn's final
            is available; otherwise ``INCOMPLETE``.
        """
        if is_speech:
            self._speech = True
        if (
            self._pending is None
            or (is_speech and self._params.wait_for_silence)
            or not self._final_ready()
        ):
            return EndOfTurnState.INCOMPLETE
        return EndOfTurnState.COMPLETE

    async def analyze_end_of_turn(self) -> tuple[EndOfTurnState, MetricsData | None]:
        """Analyze end of turn (called when Pipecat's VAD stops).

        Returns:
            ``COMPLETE`` when a server decision is latched and its final is available, else
            ``INCOMPLETE``; no metrics.
        """
        ok = self._pending is not None and self._final_ready()
        return (EndOfTurnState.COMPLETE if ok else EndOfTurnState.INCOMPLETE), None

    def clear(self):
        """Reset the analyzer: the turn ended (or was abandoned) on the Pipecat side."""
        self._pending = None
        self._speech = False
