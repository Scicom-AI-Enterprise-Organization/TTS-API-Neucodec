"""Minimal LiveKit agent for TTS stress testing: no STT, no LLM, no VAD.

Text arrives on the "tts-input" text-stream topic and is spoken verbatim via
session.say() -> openai.TTS plugin -> the TTS API (TTS_BASE_URL). This keeps the
exact production TTS path (AgentSession audio output, plugin streaming, 24kHz
publish) while removing every other moving part.

Run:  python stress_agent.py dev
Env:  LIVEKIT_URL / LIVEKIT_API_KEY / LIVEKIT_API_SECRET / TTS_BASE_URL / TTS_VOICE
      TTS_INTERLEAVE=true|false  -- send X-Interleave-Id (default true), see below
"""

import asyncio
import logging
import math
import os

import openai as openai_sdk
from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    RoomInputOptions,
    WorkerOptions,
    cli,
)
from livekit.plugins import openai

load_dotenv(".env")
logger = logging.getLogger("tts-stress-agent")

TTS_BASE_URL = os.environ.get("TTS_BASE_URL", "http://127.0.0.1:9099/v1")
TTS_VOICE = os.environ.get("TTS_VOICE", "husein")
TTS_MODEL = os.environ.get("TTS_MODEL", "TTS-model")
TTS_INTERLEAVE = os.environ.get("TTS_INTERLEAVE", "true").lower() == "true"


def _tts(room_name: str):
    """The TTS plugin, optionally carrying this room's interleave id.

    `StreamAdapter` cuts one reply into several `/v1/audio/speech` calls, and by
    default the call for chunk N+1 carries no trace of chunk N -- the LM starts cold
    and picks a fresh register, pace and energy at every join (INTERLEAVE.md §1).
    The plugin drives the stock `openai` client and cannot add body fields, so the
    id goes in the `X-Interleave-Id` header, which the API reads exactly like the
    `interleave_id` body field (app/main.py, INTERLEAVE.md §6).

    One AgentSession = one TTS instance = one client = one room's id, so every chunk
    of a room is generated in the context of the ones before it (bounded by
    MAX_RETAIN_INTERLEAVE turns and INTERLEAVE_MAX_S seconds server-side) and rooms
    never share history. TTS_INTERLEAVE=false is the cold arm, for the A/B.

    Only worth switching on against an interleave-trained checkpoint -- on anything
    else the same prompt shape is a regression risk (bench/INTERLEAVE_AB.md §7).
    """
    common = dict(model=TTS_MODEL, voice=TTS_VOICE, response_format="pcm")
    if not TTS_INTERLEAVE:
        return openai.TTS(api_key="unused", base_url=TTS_BASE_URL, **common)
    client = openai_sdk.AsyncClient(
        api_key="unused",
        base_url=TTS_BASE_URL,
        default_headers={"X-Interleave-Id": room_name},
        max_retries=0,          # a retried chunk would be stored twice
    )
    return openai.TTS(client=client, **common)


async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}
    logger.info(f"interleave {'ON' if TTS_INTERLEAVE else 'OFF'} for room {ctx.room.name}")

    session = AgentSession(tts=_tts(ctx.room.name))

    def on_text(reader, participant_identity):
        async def speak():
            text = await reader.read_all()
            logger.info(f"say ({participant_identity}): {text[:80]}")
            session.say(text, allow_interruptions=False)

        asyncio.create_task(speak())

    ctx.room.register_text_stream_handler("tts-input", on_text)

    await session.start(
        agent=Agent(instructions="tts stress test - speaks received text verbatim"),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            audio_enabled=False,
            video_enabled=False,
            text_enabled=False,
        ),
    )
    await ctx.connect()


if __name__ == "__main__":
    # the box runs unrelated GPU training -> high CPU load. The default load_fnc
    # reports that load to livekit-server, which then refuses to select the worker
    # ("no servers available"), and load_threshold alone can't fix server-side
    # selection - so report zero load for the stress test.
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            load_fnc=lambda *_: 0.0,
            load_threshold=math.inf,
        )
    )
