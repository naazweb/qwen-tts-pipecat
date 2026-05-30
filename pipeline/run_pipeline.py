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

os.environ.setdefault("HF_HOME", "/workspace/.hf_home")

sys.path.insert(0, os.path.dirname(__file__))

from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    EndFrame, Frame, LLMFullResponseEndFrame, LLMFullResponseStartFrame,
    TextFrame, TranscriptionFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.runner.run import main as runner_main, app as pipecat_app
from pipecat.runner.types import SmallWebRTCRunnerArguments
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.connection import SmallWebRTCConnection, IceServer
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from fastapi import Request
from fastapi.responses import JSONResponse

# Free TURN server — forces relay over TCP when UDP is blocked (e.g. Vast.ai)
TURN_SERVERS = [
    IceServer(
        urls=["turn:openrelay.metered.ca:80", "turn:openrelay.metered.ca:443"],
        username="openrelayproject",
        credential="openrelayproject",
    ),
    IceServer(urls=["stun:stun.l.google.com:19302"]),
]

from openai import AsyncOpenAI
from tts_service import QwenTTSService


# Inject TURN servers into every /start response so the browser uses relay candidates
@pipecat_app.middleware("http")
async def inject_ice_config(request: Request, call_next):
    response = await call_next(request)
    if request.url.path == "/start" and request.method == "POST":
        import json
        from starlette.responses import Response
        body = b""
        async for chunk in response.body_iterator:
            body += chunk
        try:
            data = json.loads(body)
            data["iceConfig"] = {
                "iceServers": [
                    {
                        "urls": ["turn:openrelay.metered.ca:80", "turn:openrelay.metered.ca:443"],
                        "username": "openrelayproject",
                        "credential": "openrelayproject",
                    },
                    {"urls": ["stun:stun.l.google.com:19302"]},
                ]
            }
            return Response(
                content=json.dumps(data),
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type="application/json",
            )
        except Exception:
            return Response(content=body, status_code=response.status_code,
                          headers=dict(response.headers))
    return response


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

        logger.info(f"STT received audio: {len(pcm)/16000:.2f}s, rms={float(np.sqrt(np.mean(pcm**2))):.4f}")

        loop = asyncio.get_event_loop()
        segments, _ = await loop.run_in_executor(
            None, lambda: self._whisper.transcribe(pcm, beam_size=1, language="en")
        )
        text = " ".join(s.text.strip() for s in segments).strip()
        logger.info(f"Whisper output: {text!r}")
        if text:
            logger.info(f"Transcribed: {text!r}")
            yield TranscriptionFrame(text=text, user_id="user", timestamp=0)


# ---------------------------------------------------------------------------
# OpenAI LLM — streams response tokens directly to TTS
# ---------------------------------------------------------------------------

class OpenAILLM(FrameProcessor):
    def __init__(self, model: str = "gpt-4o-mini", system: str = "You are a helpful voice assistant. Keep responses concise and conversational."):
        super().__init__()
        self._client = AsyncOpenAI()
        self._model = model
        self._system = system
        self._history = [{"role": "system", "content": system}]

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame):
            self._history.append({"role": "user", "content": frame.text})
            await self.push_frame(LLMFullResponseStartFrame())
            full = ""
            async with self._client.chat.completions.stream(
                model=self._model,
                messages=self._history,
            ) as stream:
                async for chunk in stream:
                    token = chunk.choices[0].delta.content if chunk.choices else None
                    if token:
                        full += token
                        await self.push_frame(TextFrame(text=token))
            self._history.append({"role": "assistant", "content": full})
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
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.8, start_secs=0.2, confidence=0.7)),
            audio_out_sample_rate=24000,
        ),
    )

    stt = WhisperSTTService(model_size="base.en", device="cuda")
    llm = OpenAILLM()
    tts = QwenTTSService(language="English", device="cuda")

    pipeline = Pipeline([
        transport.input(),
        stt,
        llm,
        tts,
        transport.output(),
    ])

    task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=True))

    @transport.event_handler("on_client_connected")
    async def on_connected(transport, client):
        from pipecat.frames.frames import TTSSpeakFrame
        logger.info("Client connected — sending greeting")
        await asyncio.sleep(5)
        logger.info("Queuing greeting TTS frame")
        await task.queue_frame(TTSSpeakFrame("Hello! I am your voice assistant powered by Qwen3 TTS. How can I help you today?"))

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
