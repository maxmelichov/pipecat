#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""VAD analyzer fed by the audioforge server's per-frame VAD probability.

The audioforge server (see :mod:`pipecat.services.audioforge.stt`) sends one ``frame`` event per
80 ms of audio with the probability of its learned VAD head. This analyzer returns that
probability as the confidence of each 80 ms step, and Pipecat's own start/stop hysteresis
(:class:`~pipecat.audio.vad.vad_analyzer.VADParams` ``confidence`` / ``start_secs`` /
``stop_secs``) runs on top of it unchanged. No model runs in the Pipecat process.

The probability comes back over the socket, so the value used for the audio just delivered is the
newest one the server has sent: about one to two frames (80-160 ms at real time) older than that
audio. ``VADParams.min_volume`` is 0 unless set explicitly: the learned VAD decides, and Pipecat's
loudness gate would make the result depend on the recording level.
"""

from loguru import logger

from pipecat.audio.vad.vad_analyzer import VADAnalyzer, VADParams

try:
    from audioforge_client import FRAME_MS
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use audioforge, you need to `pip install pipecat-ai[audioforge]`.")
    raise Exception(f"Missing module: {e}")

from pipecat.services.audioforge.stt import AudioforgeHub


class AudioforgeVADAnalyzer(VADAnalyzer):
    """Voice activity from the audioforge server's VAD head, with Pipecat's hysteresis on top.

    Build it with :meth:`AudioforgeSTTService.create_vad_analyzer
    <pipecat.services.audioforge.stt.AudioforgeSTTService.create_vad_analyzer>` or pass the
    service's ``hub``. It hears every speaker on the input, not only the primary one.
    """

    def __init__(
        self,
        hub: AudioforgeHub,
        *,
        sample_rate: int | None = None,
        params: VADParams | None = None,
    ):
        """Initialize the analyzer.

        Args:
            hub: The hub of the :class:`~pipecat.services.audioforge.stt.AudioforgeSTTService`
                that owns the server session.
            sample_rate: Audio sample rate in Hz. If None, will be set later.
            params: VAD parameters. ``min_volume`` defaults to 0 here (see the module docstring).
        """
        if params is None:
            params = VADParams(min_volume=0.0)
        elif "min_volume" not in params.model_fields_set:
            params = params.model_copy(update={"min_volume": 0.0})
        super().__init__(sample_rate=sample_rate, params=params)
        self._hub = hub
        self.steps = 0

    def num_frames_required(self) -> int:
        """Get the number of audio frames required for analysis.

        Returns:
            One server frame (80 ms) per analyzer step.
        """
        return int((self.sample_rate or 16000) * FRAME_MS / 1000)

    def voice_confidence(self, buffer: bytes) -> float:
        """Calculate voice activity confidence for the given audio buffer.

        Args:
            buffer: Audio buffer (unused: the confidence is the server's newest VAD probability).

        Returns:
            The server's VAD probability in [0, 1].
        """
        self.steps += 1
        return float(self._hub.vad)
