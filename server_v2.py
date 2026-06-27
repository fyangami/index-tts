import argparse
import asyncio
import io
import json
import os
import sys
import time
import uuid
import warnings
from functools import partial
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

import torch
import torchaudio
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from indextts.infer_v2 import IndexTTS2

VOICES_DIR = Path("voices")
VOICES_DIR.mkdir(exist_ok=True)
SAMPLING_RATE = 22050

app = FastAPI(title="IndexTTS2 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

tts: Optional[IndexTTS2] = None


def get_tts():
    global tts
    if tts is None:
        raise HTTPException(status_code=503, detail="TTS engine not initialized")
    return tts


# ---------------------------------------------------------------------------
# OpenAI-compatible request / response models
# ---------------------------------------------------------------------------

class TTSRequest(BaseModel):
    model: str = "tts-1"
    input: str = Field(..., min_length=1)
    voice: str = Field(..., min_length=1)
    response_format: str = "wav"
    speed: float = 1.0
    stream: bool = False


class VoiceInfo(BaseModel):
    voice_id: str
    created_at: float
    duration_seconds: Optional[float] = None


class VoiceListResponse(BaseModel):
    voices: list[VoiceInfo]


# ---------------------------------------------------------------------------
# SSE helpers for streaming
# ---------------------------------------------------------------------------

def sse_event(data: bytes) -> bytes:
    return b"data: " + data + b"\n\n"


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def _convert_format(wav_tensor: torch.Tensor, target_format: str) -> bytes:
    buf = io.BytesIO()
    wav_int16 = wav_tensor.type(torch.int16)

    if target_format == "wav":
        torchaudio.save(buf, wav_int16, SAMPLING_RATE, format="wav")
    elif target_format == "mp3":
        torchaudio.save(buf, wav_int16, SAMPLING_RATE, format="mp3")
    elif target_format == "flac":
        torchaudio.save(buf, wav_int16, SAMPLING_RATE, format="flac")
    elif target_format in ("opus", "aac", "pcm"):
        if target_format == "pcm":
            return wav_int16.numpy().tobytes()
        torchaudio.save(buf, wav_int16, SAMPLING_RATE, format=target_format)
    else:
        torchaudio.save(buf, wav_int16, SAMPLING_RATE, format="wav")

    return buf.getvalue()


def _get_voice_path(voice_id: str) -> Path:
    for ext in ("wav", "mp3", "flac", "ogg", "opus", "m4a"):
        p = VOICES_DIR / f"{voice_id}.{ext}"
        if p.exists():
            return p
    raise HTTPException(status_code=404, detail=f"Voice '{voice_id}' not found")


def _scan_voices() -> list[VoiceInfo]:
    results = []
    for f in VOICES_DIR.iterdir():
        if f.is_file() and f.suffix.lower() in (".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a"):
            voice_id = f.stem
            created_at = f.stat().st_ctime
            dur = None
            try:
                info = torchaudio.info(str(f))
                dur = info.num_frames / info.sample_rate
            except Exception:
                pass
            results.append(VoiceInfo(voice_id=voice_id, created_at=created_at, duration_seconds=dur))
    return sorted(results, key=lambda v: v.created_at, reverse=True)


# ---------------------------------------------------------------------------
# Voice management endpoints
# ---------------------------------------------------------------------------

@app.get("/v1/audio/voices", response_model=VoiceListResponse)
def list_voices():
    return VoiceListResponse(voices=_scan_voices())


@app.post("/v1/audio/voices", status_code=201)
async def upload_voice(file: UploadFile = File(...), voice_id: str = Form(None)):
    if voice_id is None:
        voice_id = str(uuid.uuid4())[:8]
    else:
        existing = list(VOICES_DIR.glob(f"{voice_id}.*"))
        if existing:
            raise HTTPException(status_code=409, detail=f"Voice '{voice_id}' already exists")

    ext = Path(file.filename).suffix if file.filename else ".wav"
    if ext.lower() not in (".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a"):
        ext = ".wav"

    dest = VOICES_DIR / f"{voice_id}{ext}"
    content = await file.read()
    dest.write_bytes(content)

    info = None
    try:
        audio_info = torchaudio.info(str(dest))
        info = VoiceInfo(
            voice_id=voice_id,
            created_at=dest.stat().st_ctime,
            duration_seconds=audio_info.num_frames / audio_info.sample_rate,
        )
    except Exception:
        info = VoiceInfo(voice_id=voice_id, created_at=dest.stat().st_ctime)

    return info


@app.delete("/v1/audio/voices/{voice_id}")
def delete_voice(voice_id: str):
    voice_path = _get_voice_path(voice_id)
    voice_path.unlink()
    return {"status": "deleted", "voice_id": voice_id}


# ---------------------------------------------------------------------------
# OpenAI-compatible TTS endpoint
# ---------------------------------------------------------------------------

@app.post("/v1/audio/speech")
async def speech(request: TTSRequest):
    tts = get_tts()
    voice_path = _get_voice_path(request.voice)

    text = request.input

    kwargs = {
        "do_sample": True,
        "top_p": 0.8,
        "top_k": 30,
        "temperature": 0.8,
        "length_penalty": 0.0,
        "num_beams": 3,
        "repetition_penalty": 10.0,
        "max_mel_tokens": 1500,
    }

    if request.stream:
        return _streaming_response(tts, str(voice_path), text, kwargs)
    else:
        audio_bytes = await _non_streaming_response(tts, str(voice_path), text, kwargs, request.response_format)
        media_type_map = {
            "wav": "audio/wav",
            "mp3": "audio/mpeg",
            "flac": "audio/flac",
            "opus": "audio/opus",
            "aac": "audio/aac",
            "pcm": "audio/L16;rate=22050;channels=1",
        }
        fmt = request.response_format
        return Response(
            content=audio_bytes,
            media_type=media_type_map.get(fmt, "audio/wav"),
            headers={"Content-Disposition": f"attachment; filename=speech.{fmt}"},
        )


async def _non_streaming_response(tts: IndexTTS2, voice_path: str, text: str, kwargs: dict, fmt: str):
    loop = asyncio.get_event_loop()

    def run_infer():
        return tts.infer(
            spk_audio_prompt=voice_path,
            text=text,
            output_path=None,
            verbose=False,
            **kwargs,
        )

    result = await loop.run_in_executor(None, run_infer)
    if result is None:
        raise HTTPException(status_code=500, detail="TTS inference returned no audio")

    sr, wav_data = result
    wav_tensor = torch.tensor(wav_data.T, dtype=torch.float32)

    audio_bytes = _convert_format(wav_tensor, fmt)
    return audio_bytes


def _streaming_response(tts: IndexTTS2, voice_path: str, text: str, kwargs: dict):

    async def event_stream():
        loop = asyncio.get_event_loop()
        gen = tts.infer_generator(
            spk_audio_prompt=voice_path,
            text=text,
            output_path=None,
            verbose=False,
            stream_return=True,
            **kwargs,
        )

        while True:
            try:
                chunk = await loop.run_in_executor(None, partial(next, gen))
            except StopIteration:
                break

            if chunk is None:
                continue
            if isinstance(chunk, tuple) and len(chunk) == 2:
                sr, wav_data = chunk
                wav_tensor = torch.tensor(wav_data.T, dtype=torch.float32)
            elif isinstance(chunk, torch.Tensor):
                wav_tensor = chunk
            else:
                continue

            wav_int16 = wav_tensor.type(torch.int16)
            buf = io.BytesIO()
            torchaudio.save(buf, wav_int16, SAMPLING_RATE, format="wav")

            yield sse_event(buf.getvalue())

        yield b"data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/v1/health")
def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

def init_engine(model_dir: str, fp16: bool, deepspeed: bool, cuda_kernel: bool, torch_compile: bool):
    global tts
    if tts is not None:
        return

    cfg_path = os.path.join(model_dir, "config.yaml")
    if not os.path.exists(cfg_path):
        print(f"Config not found at {cfg_path}, downloading model...")
        from indextts.utils.model_download import ensure_config_available, ensure_models_available
        ensure_config_available(model_dir)
        ensure_models_available(model_dir)

    tts = IndexTTS2(
        cfg_path=cfg_path,
        model_dir=model_dir,
        use_fp16=fp16,
        use_deepspeed=deepspeed,
        use_cuda_kernel=cuda_kernel,
        use_torch_compile=torch_compile,
    )
    print(">> TTS engine initialized")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IndexTTS2 OpenAI-compatible API server")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model_dir", type=str, default="./checkpoints")
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--deepspeed", action="store_true", default=False)
    parser.add_argument("--cuda_kernel", action="store_true", default=False)
    parser.add_argument("--torch_compile", action="store_true", default=False)
    args = parser.parse_args()

    init_engine(
        model_dir=args.model_dir,
        fp16=args.fp16,
        deepspeed=args.deepspeed,
        cuda_kernel=args.cuda_kernel,
        torch_compile=args.torch_compile,
    )

    uvicorn.run(app, host=args.host, port=args.port)
