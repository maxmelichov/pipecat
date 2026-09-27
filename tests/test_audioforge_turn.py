#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for the audioforge turn and VAD analyzers (hub-driven, no socket)."""

import pytest
from audioforge_client import FrameEvent, Ready, TurnEnd

from pipecat.audio.turn.audioforge_turn import AudioforgeTurnAnalyzer, AudioforgeTurnParams
from pipecat.audio.turn.base_turn_analyzer import EndOfTurnState
from pipecat.audio.vad.audioforge_vad import AudioforgeVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.services.audioforge.stt import AudioforgeHub


def _frame(hub, t, vad):
    hub.on_frame(FrameEvent(t=t, vad=vad, eot=None, speakers=(vad, 0, 0, 0), primary=0))


def _turn_end(hub, t, policy="timeout"):
    return hub.on_turn_end(TurnEnd(t=t, policy=policy, p=None, silence_ms=1000))


def test_params_validation_and_hub_policy():
    hub = AudioforgeHub()
    with pytest.raises(ValueError):
        AudioforgeTurnAnalyzer(hub, params=AudioforgeTurnParams(policy="soon"))
    analyzer = AudioforgeTurnAnalyzer(hub, params=AudioforgeTurnParams(policy="hybrid_dyn"))
    assert hub.policy == "hybrid_dyn"
    assert analyzer.params.wait_for_silence is True and analyzer.params.resume_ms == 240
    # The first analyzer owns the hub policy; a second one does not overwrite it.
    AudioforgeTurnAnalyzer(hub, params=AudioforgeTurnParams(policy="timeout"))
    assert hub.policy == "hybrid_dyn"


@pytest.mark.asyncio
async def test_latch_waits_for_silence_and_clears():
    hub = AudioforgeHub()
    analyzer = AudioforgeTurnAnalyzer(hub)
    assert analyzer.append_audio(b"", False) == EndOfTurnState.INCOMPLETE
    assert analyzer.speech_triggered is False

    _turn_end(hub, 2.0)
    assert analyzer.pending is not None
    # Pipecat's VAD still hears speech: hold.
    assert analyzer.append_audio(b"", True) == EndOfTurnState.INCOMPLETE
    assert analyzer.speech_triggered is True
    # Silence: the latched decision applies.
    assert analyzer.append_audio(b"", False) == EndOfTurnState.COMPLETE
    state, metrics = await analyzer.analyze_end_of_turn()
    assert state == EndOfTurnState.COMPLETE and metrics is None

    analyzer.clear()
    assert analyzer.pending is None and analyzer.speech_triggered is False
    assert analyzer.append_audio(b"", False) == EndOfTurnState.INCOMPLETE
    assert (await analyzer.analyze_end_of_turn())[0] == EndOfTurnState.INCOMPLETE


def test_without_silence_gate():
    hub = AudioforgeHub()
    analyzer = AudioforgeTurnAnalyzer(hub, params=AudioforgeTurnParams(wait_for_silence=False))
    _turn_end(hub, 2.0)
    assert analyzer.append_audio(b"", True) == EndOfTurnState.COMPLETE


def test_speech_after_the_decision_makes_it_stale():
    hub = AudioforgeHub()
    analyzer = AudioforgeTurnAnalyzer(hub)
    _turn_end(hub, 2.0)
    _frame(hub, 2.08, 0.9)
    _frame(hub, 2.16, 0.9)
    assert analyzer.pending is not None  # 160 ms < resume_ms
    _frame(hub, 2.24, 0.9)
    assert analyzer.pending is None and len(analyzer.discarded) == 1
    assert analyzer.append_audio(b"", False) == EndOfTurnState.INCOMPLETE

    # Frames that arrived before the event but lie after its decision time count too.
    _frame(hub, 4.08, 0.9)
    _frame(hub, 4.16, 0.9)
    _frame(hub, 4.24, 0.9)
    _turn_end(hub, 4.0)
    assert analyzer.pending is None and len(analyzer.discarded) == 2

    # Speech before the decision time does not.
    _frame(hub, 5.9, 0.9)
    _turn_end(hub, 6.0)
    assert analyzer.pending is not None
    assert len(analyzer.received) == 3


def test_policy_filter_and_both():
    hub = AudioforgeHub()
    dyn = AudioforgeTurnAnalyzer(hub, params=AudioforgeTurnParams(policy="hybrid_dyn"))
    both = AudioforgeTurnAnalyzer(hub, params=AudioforgeTurnParams(policy="both"))
    _turn_end(hub, 1.0, "timeout")
    assert dyn.pending is None and both.pending is not None
    _turn_end(hub, 2.0, "hybrid_dyn")
    assert dyn.pending is not None and dyn.pending.t == 2.0


def test_offline_final_gates_completion():
    hub = AudioforgeHub()
    hub.final_source = "offline"
    hub.ready = Ready(model="m", chunk_ms=160, frame_ms=80, final_asr="tdt_v3")
    analyzer = AudioforgeTurnAnalyzer(hub)
    _turn_end(hub, 3.0)
    assert analyzer.append_audio(b"", False) == EndOfTurnState.INCOMPLETE
    hub.offline_final_t.add(3.0)
    assert analyzer.append_audio(b"", False) == EndOfTurnState.COMPLETE

    # final_source="stream" never waits, whatever the server runs.
    hub2 = AudioforgeHub()
    hub2.ready = Ready(model="m", chunk_ms=160, frame_ms=80, final_asr="tdt_v3")
    a2 = AudioforgeTurnAnalyzer(hub2)
    _turn_end(hub2, 3.0)
    assert a2.append_audio(b"", False) == EndOfTurnState.COMPLETE


def test_session_reset_drops_the_latch():
    hub = AudioforgeHub()
    analyzer = AudioforgeTurnAnalyzer(hub)
    _turn_end(hub, 2.0)
    hub.reset()
    assert analyzer.pending is None and hub.turn_ends == []


def test_vad_analyzer_reads_the_hub():
    hub = AudioforgeHub()
    vad = AudioforgeVADAnalyzer(hub)
    vad.set_sample_rate(16000)
    assert vad.num_frames_required() == 1280
    assert vad.params.min_volume == 0.0
    assert vad.voice_confidence(b"\x00" * 2560) == 0.0
    _frame(hub, 0.08, 0.93)
    assert vad.voice_confidence(b"\x00" * 2560) == pytest.approx(0.93)
    assert vad.steps == 2

    explicit = AudioforgeVADAnalyzer(hub, params=VADParams(min_volume=0.3, stop_secs=0.4))
    assert explicit.params.min_volume == 0.3 and explicit.params.stop_secs == 0.4
    implicit = AudioforgeVADAnalyzer(hub, params=VADParams(stop_secs=0.4))
    assert implicit.params.min_volume == 0.0 and implicit.params.stop_secs == 0.4
