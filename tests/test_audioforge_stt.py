#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the audioforge STT service, against the client package's scripted fake server."""

import asyncio
import json
from unittest.mock import AsyncMock

import pytest
from audioforge_client import Final, LanguageEvent, Partial, Ready, ServerError, Stats, TurnEnd
from audioforge_client.testing import FakeAudioforgeServer, speech_pcm
from websockets.protocol import State

from pipecat.frames.frames import InterimTranscriptionFrame, TranscriptionFrame
from pipecat.services.audioforge.stt import AudioforgeHub, AudioforgeSTTService
from pipecat.transcriptions.language import Language


class _FakeWebsocket:
    def __init__(self, messages, *, state=State.OPEN):
        self._messages = messages
        self.state = state
        self.send = AsyncMock()

    async def close(self):
        self.state = State.CLOSED

    def __aiter__(self):
        return self._iter()

    async def _iter(self):
        for m in self._messages:
            yield m


def _capture(service):
    frames = []

    async def push_frame(frame, direction=None):
        frames.append(frame)

    service.push_frame = push_frame
    return frames


async def _stream(service, pcm, chunk=640):
    """Send ``pcm`` through run_stt in ``chunk``-byte pieces (20 ms at 16 kHz)."""
    for i in range(0, len(pcm), chunk):
        async for _ in service.run_stt(pcm[i : i + chunk]):
            pass
        await asyncio.sleep(0)


def test_defaults_and_validation():
    service = AudioforgeSTTService()
    assert service.turn_policy == "timeout"
    assert service._settings.timeout_ms == 1000
    assert service._settings.eot_threshold is None
    assert service.supports_ttfs is False
    assert isinstance(service.hub, AudioforgeHub)

    with pytest.raises(ValueError):
        AudioforgeSTTService(enroll="sometimes")
    with pytest.raises(ValueError):
        AudioforgeSTTService(final_source="both")
    with pytest.raises(ValueError):
        AudioforgeSTTService(settings=AudioforgeSTTService.Settings(turn_policy="never"))


def test_turn_analyzer_sets_the_hub_policy():
    service = AudioforgeSTTService()
    analyzer = service.create_turn_analyzer()
    assert service.hub.policy == "timeout"
    assert analyzer.params.policy == "timeout"
    vad = service.create_vad_analyzer()
    assert vad.params.min_volume == 0.0


@pytest.mark.asyncio
async def test_round_trip_against_fake_server():
    async with FakeAudioforgeServer(timeout_ms=480, language="en") as srv:
        service = AudioforgeSTTService(
            url=srv.url,
            enroll="after_agent_arm",
            settings=AudioforgeSTTService.Settings(timeout_ms=480),
        )
        frames = _capture(service)
        # A control message before the connection exists is sent right after the config.
        assert await service.agent_end()

        await service._connect_websocket()
        assert service._websocket is not None
        receiver = asyncio.create_task(service._receive_messages())

        await _stream(service, speech_pcm(1.0, 1.0))
        await service._flush_and_wait()
        await asyncio.wait_for(receiver, 5)

    assert srv.configs[0]["turn_policy"] == "timeout"
    assert srv.configs[0]["timeout_ms"] == 480
    assert srv.controls[0][1:] == ("agent_end", 0)

    interims = [f for f in frames if isinstance(f, InterimTranscriptionFrame)]
    finals = [f for f in frames if isinstance(f, TranscriptionFrame)]
    assert interims and finals
    assert finals[0].finalized is True
    assert finals[0].metadata["audioforge"]["kind"] == "final"
    assert finals[0].metadata["audioforge"]["speaker"] == 0
    # The final is cut where the server's turn_end fired.
    assert len(service.hub.turn_ends) == 1
    assert finals[0].metadata["audioforge"]["t"] == service.hub.turn_ends[0].t
    assert service.hub.turn_ends[0].policy == "timeout"
    # --lid: the language event tags the frames that follow it.
    assert service.hub.language == "en"
    assert finals[-1].language == Language.EN
    assert isinstance(service.hub.ready, Ready)
    assert isinstance(service.hub.stats, Stats)
    assert service.hub.n_frames == 25


@pytest.mark.asyncio
async def test_offline_final_source_uses_the_offline_transcript():
    async with FakeAudioforgeServer(timeout_ms=480, final_asr=True) as srv:
        service = AudioforgeSTTService(url=srv.url, final_source="offline")
        frames = _capture(service)
        await service._connect_websocket()
        receiver = asyncio.create_task(service._receive_messages())
        await _stream(service, speech_pcm(1.0, 1.0))
        await service._flush_and_wait()
        await asyncio.wait_for(receiver, 5)

    finals = [f for f in frames if isinstance(f, TranscriptionFrame)]
    assert finals and all(f.metadata["audioforge"]["source"] == "tdt_v3" for f in finals)
    assert finals[0].text == finals[0].text.upper()  # the fake's offline twin is upper-cased
    assert service.hub.offline_final_t


@pytest.mark.asyncio
async def test_receive_messages_parses_batches_and_errors():
    service = AudioforgeSTTService()
    frames = _capture(service)
    messages = [
        json.dumps({"type": "ready", "model": "m", "chunk_ms": 160, "frame_ms": 80}),
        json.dumps(
            {
                "type": "frames",
                "items": [
                    {"t": 0.08, "vad": 0.9, "eot": 0.1, "speakers": [0.9, 0, 0, 0], "primary": 0},
                    {"t": 0.16, "vad": 0.1, "eot": 0.2, "speakers": [0.1, 0, 0, 0], "primary": 0},
                ],
            }
        ),
        json.dumps({"type": "partial", "t": 0.16, "text": "hello"}),
        json.dumps({"type": "error", "code": "bad_config", "detail": "x", "fatal": False}),
        json.dumps(
            {"type": "turn_end", "t": 1.2, "policy": "timeout", "p": None, "silence_ms": 1000}
        ),
        json.dumps({"type": "final", "t": 1.2, "text": "hello world", "speaker": 0}),
        json.dumps({"type": "stats", "rtf": 0.5}),
    ]
    service._websocket = _FakeWebsocket(messages)
    service._stats_event = asyncio.Event()
    await service._receive_messages()

    assert service.hub.n_frames == 2 and service.hub.vad == 0.1 and service.hub.primary == 0
    assert [type(f) for f in frames] == [InterimTranscriptionFrame, TranscriptionFrame]
    assert frames[1].text == "hello world"
    assert service.hub.errors and service.hub.errors[0].code == "bad_config"
    assert service._stats_event.is_set()


@pytest.mark.asyncio
async def test_handle_server_event_units():
    service = AudioforgeSTTService()
    frames = _capture(service)
    await service.handle_server_event(Partial(t=0.5, text="   "))  # empty partials are dropped
    await service.handle_server_event(Final(t=1.0, text="", speaker=None))  # so are empty finals
    assert frames == []
    await service.handle_server_event(LanguageEvent(t=1.0, language="xx-unknown", confidence=0.9))
    assert service.hub.language == "xx-unknown" and service._detected_language is None
    await service.handle_server_event(TurnEnd(t=2.0, policy="hybrid_dyn", p=0.999, silence_ms=560))
    assert service.hub.turn_ends[-1].policy == "hybrid_dyn"
    await service.handle_server_event(ServerError(message="overloaded", code="overloaded"))
    assert service.hub.errors[-1].fatal is False
