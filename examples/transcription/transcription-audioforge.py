#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Streaming transcription with the self-hosted audioforge server.

Start the server first (see https://github.com/maxmelichov/nvidia-audio-models):

    python -m audioforge.serve --asr runs/stage1_served.afm --diar runs/nemo_sortformer_v2.afm

then run this bot with `pip install "pipecat-ai[audioforge]"`. Set AUDIOFORGE_URL to point at a
server that is not on ws://127.0.0.1:8765.
"""

import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.evals.transport import EvalTransportParams
from pipecat.frames.frames import Frame, InterimTranscriptionFrame, TranscriptionFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker, ProcessorUnusablePolicy
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.audioforge.stt import AudioforgeSTTService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)


class TranscriptionLogger(FrameProcessor):
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            info = frame.metadata.get("audioforge", {})
            print(f"Transcription (speaker {info.get('speaker')}, t={info.get('t')}): {frame.text}")
        elif isinstance(frame, InterimTranscriptionFrame):
            print(f"Interim transcription: {frame.text}")

        # Push all frames through
        await self.push_frame(frame, direction)


# We use lambdas to defer transport parameter creation until the transport
# type is selected at runtime.
transport_params = {
    "eval": lambda: EvalTransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
    ),
    "daily": lambda: DailyParams(audio_in_enabled=True),
    "twilio": lambda: FastAPIWebsocketParams(audio_in_enabled=True),
    "webrtc": lambda: TransportParams(audio_in_enabled=True),
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    logger.info("Starting bot")

    stt = AudioforgeSTTService(url=os.getenv("AUDIOFORGE_URL", "ws://127.0.0.1:8765"))

    @stt.event_handler("on_language_detected")
    async def on_language_detected(service, language, confidence):
        # Only with a server started with --lid.
        print(f"Language: {language} ({confidence:.2f})")

    tl = TranscriptionLogger()

    # The VAD probability comes from the same server session: no local VAD model.
    vad_processor = VADProcessor(vad_analyzer=stt.create_vad_analyzer())

    pipeline = Pipeline([transport.input(), stt, vad_processor, tl, transport.output()])

    worker = PipelineWorker(
        pipeline,
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
        processor_unusable_policy=ProcessorUnusablePolicy.END,
    )

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)
    await runner.add_workers(worker)

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await runner.cancel()

    await runner.run()


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
