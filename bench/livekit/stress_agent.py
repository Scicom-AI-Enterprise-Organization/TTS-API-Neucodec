"""Minimal LiveKit agent for TTS stress testing: no STT, no LLM, no VAD.

Text arrives on the "tts-input" text-stream topic and is spoken verbatim via
session.say() -> openai.TTS plugin -> the TTS API (TTS_BASE_URL). This keeps the
exact production TTS path (AgentSession audio output, plugin streaming, 24kHz
publish) while removing every other moving part.

The plugin is a non-streaming TTS, so livekit wraps it in a StreamAdapter that cuts
each reply into sentence chunks (>= 20 chars) and synthesizes them one request at a
time, in order. Without help every chunk is generated cold and the joins sound
disconnected; the room name is therefore sent as the `X-Context-Id` header (the API's
`request_id`, see README "Speech context") through the openai client's
default_headers, so each chunk is generated in the context of the previous ones.
Set TTS_CONTEXT=0 to compare against the cold-start behaviour.

Run:  python stress_agent.py dev
Env:  LIVEKIT_URL / LIVEKIT_API_KEY / LIVEKIT_API_SECRET / TTS_BASE_URL / TTS_VOICE
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
TTS_CONTEXT = os.environ.get("TTS_CONTEXT", "1") not in ("0", "false", "no")


async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    # one client per room -> one speech-context id per room: every sentence chunk of
    # this session is generated in the context of the ones before it.
    tts_client = openai_sdk.AsyncClient(
        api_key="unused",
        base_url=TTS_BASE_URL,
        default_headers={"X-Context-Id": ctx.room.name} if TTS_CONTEXT else None,
    )
    session = AgentSession(
        tts=openai.TTS(
            model=TTS_MODEL,
            voice=TTS_VOICE,
            client=tts_client,
            response_format="pcm",
        ),
    )

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
