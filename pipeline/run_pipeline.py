"""
End-to-end voice agent pipeline.

    Browser mic → Deepgram STT → OpenAI LLM → QwenTTSService → Browser speaker

Run:
    python pipeline/run_pipeline.py -t daily
"""

import asyncio
import sys
import os

os.environ.setdefault("HF_HOME", "/workspace/.hf_home")

sys.path.insert(0, os.path.dirname(__file__))

from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    EndFrame, Frame, TranscriptionFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.runner.run import main as runner_main
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.transports.base_transport import BaseTransport
from pipecat.transports.daily.transport import DailyParams
from pipecat.workers.runner import WorkerRunner

from pipecat.services.google.vertex.llm import GoogleLLMService
from tts_service import QwenTTSService

transport_params = {
    "daily": lambda: DailyParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_enabled=True,
        vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.8, start_secs=0.2, confidence=0.7)),
        audio_out_sample_rate=24000,
    ),
}


# ---------------------------------------------------------------------------
# Bot
# ---------------------------------------------------------------------------

async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    stt = DeepgramSTTService(api_key=os.getenv("DEEPGRAM_API_KEY"))
    llm = GoogleLLMService(
        api_key=os.getenv("GEMINI_API_KEY"),
        settings=GoogleLLMService.Settings(
            model="gemini-2.0-flash",
            system_instruction="You are a helpful voice assistant. Keep responses concise and conversational.",
        ),
    )
    tts = QwenTTSService(language="English", device="cuda")

    pipeline = Pipeline([
        transport.input(),
        stt,
        llm,
        tts,
        transport.output(),
    ])

    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
        ),
    )

    @transport.event_handler("on_client_connected")
    async def on_connected(transport, client):
        from pipecat.frames.frames import TTSSpeakFrame
        logger.info("Client connected — sending greeting")
        await asyncio.sleep(2)
        await worker.queue_frames([TTSSpeakFrame("Hello! I am your voice assistant powered by Qwen3 TTS. How can I help you today?")])

    @transport.event_handler("on_client_disconnected")
    async def on_disconnect(transport, client):
        await worker.cancel()

    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)
    await runner.add_workers(worker)
    await runner.run()


async def bot(runner_args: RunnerArguments):
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    runner_main()
