"""
End-to-end voice agent pipeline.

    Browser mic → VAD → Whisper STT → Echo LLM → QwenTTSService → Browser speaker

Run:
    pip install "pipecat-ai[webrtc,runner,silero]"
    python pipeline/run_pipeline.py

Then open http://<server-ip>:7860/client in your browser.
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    EndFrame, Frame, LLMFullResponseEndFrame, LLMFullResponseStartFrame,
    TextFrame, TranscriptionFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.runner.run import main as runner_main
from pipecat.runner.types import SmallWebRTCRunnerArguments
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport

from tts_service import QwenTTSService


# ---------------------------------------------------------------------------
# Faster-Whisper STT
# ---------------------------------------------------------------------------

class WhisperSTTService(SegmentedSTTService):
    def __init__(self, model_size: str = "base.en", device: str = "cuda", **kwargs):
        super().__init__(**kwargs)
        from faster_whisper import WhisperModel
        logger.info(f"Loading Whisper {model_size}...")
        self._whisper = WhisperModel(model_size, device=device, compute_type="float16")

    async def run_stt(self, audio: bytes):
        import io, wave
        import numpy as np

        with io.BytesIO(audio) as f:
            with wave.open(f) as wf:
                raw = wf.readframes(wf.getnframes())
                pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

        loop = asyncio.get_event_loop()
        segments, _ = await loop.run_in_executor(
            None, lambda: self._whisper.transcribe(pcm, beam_size=1, language="en")
        )
        text = " ".join(s.text.strip() for s in segments).strip()
        if text:
            logger.info(f"Transcribed: {text!r}")
            yield TranscriptionFrame(text=text, user_id="user", timestamp=0)


# ---------------------------------------------------------------------------
# Echo LLM — passes transcription straight to TTS (demo, replace with real LLM)
# ---------------------------------------------------------------------------

class EchoLLM(FrameProcessor):
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame):
            await self.push_frame(LLMFullResponseStartFrame())
            await self.push_frame(TextFrame(text=frame.text))
            await self.push_frame(LLMFullResponseEndFrame())
        else:
            await self.push_frame(frame, direction)


# ---------------------------------------------------------------------------
# Bot entry point — called by the Pipecat runner per WebRTC connection
# ---------------------------------------------------------------------------

async def bot(runner_args: SmallWebRTCRunnerArguments):
    transport = SmallWebRTCTransport(
        webrtc_connection=runner_args.webrtc_connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
            audio_out_sample_rate=24000,
        ),
    )

    stt = WhisperSTTService(model_size="base.en", device="cuda")
    llm = EchoLLM()
    tts = QwenTTSService(language="English", device="cuda")

    pipeline = Pipeline([
        transport.input(),
        stt,
        llm,
        tts,
        transport.output(),
    ])

    task = PipelineTask(pipeline, PipelineParams(allow_interruptions=True))

    @transport.event_handler("on_client_disconnected")
    async def on_disconnect(transport, client):
        await task.queue_frame(EndFrame())

    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    runner_main()
