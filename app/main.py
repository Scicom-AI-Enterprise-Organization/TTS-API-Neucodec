from app.env import *

import torch

torch._dynamo.config.recompile_limit = 128
torch.set_float32_matmul_precision('high')

from typing import Literal
import re
import json
import asyncio
import io
import wave
import tempfile
from tqdm import tqdm
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, FileResponse, Response
from pydantic import BaseModel
from huggingface_hub import hf_hub_download
from malaya.text.normalization import to_cardinal
import torch.cuda as cuda
import uuid
import bisect
import threading
import queue as thread_queue
import concurrent.futures
import soundfile as sf
import numpy as np
import aiohttp
import sentry_sdk
import malaya
import fasttext
import fastapi_loki_tempo
from app.rules import *
from app.wrapper import CUDAGraphsWrapper
from app.neucodec import NeuCodec

if len(SENTRY_DSN):
    sentry_sdk.init(dsn=SENTRY_DSN, send_default_pii=True)

app = FastAPI()
fastapi_loki_tempo.patch(app=app)

torch.set_grad_enabled(False)

if torch.cuda.is_available():
    device = "cuda"
else:
    logging.warning("GPU is not available, will run using CPU.")
    device = "cpu"

filename = hf_hub_download(
    repo_id="mesolitica/fasttext-language-detection-bahasa-en", 
    filename="fasttext.ftz"
)
lang_model = fasttext.load_model(filename)
normalizer = malaya.normalize.normalizer()

logging.info('loading audio encoder')

codec = NeuCodec.from_pretrained("neuphonic/neucodec").eval().to(device)
codebook_size = 50
sr = 24000

logging.info('done load audio encoder')

h2d_stream = cuda.Stream()
compute_stream = cuda.Stream()

def fn(padded_token):
    return codec.decode_code(padded_token.unsqueeze(1))

buckets = {}
BUCKET_TOKENS = [int(codebook_size * i) for i in CUDA_GRAPH_BATCH]
BUCKET_TOKENS = sorted(BUCKET_TOKENS)
if len(BUCKET_TOKENS) and device == "cuda":
    if TORCH_COMPILE:
        logging.info("warming up with torch compile")
    else:
        logging.info("warming up with cuda graphs")

    for B in tqdm(BUCKET_BATCHES):
        for T in BUCKET_TOKENS:
            input = torch.zeros((B, T), dtype=torch.long, device=device)
            if TORCH_COMPILE:
                l = lambda x: fn(x)
                l = torch.compile(l)
                for _ in range(3):
                    l(input)
                fn_with_graph = l
            else:
                fn_with_graph = CUDAGraphsWrapper.wrap(fn, [input], stream=compute_stream)
            buckets[(B, T)] = fn_with_graph

    logging.info(f'keys: {buckets.keys()}')

compute_queue = thread_queue.Queue()
batch_queue = thread_queue.Queue()
dynamic_batch_queue = asyncio.Queue()

def thread_safe_set_result(loop, future, value):
    loop.call_soon_threadsafe(future.set_result, value)

def make_pinned_batch(tokens, target_T, dtype=torch.long):
    B = len(tokens)
    pinned = torch.empty((B, target_T), dtype=dtype).pin_memory()
    pinned.fill_(0)

    for i, t in enumerate(tokens):
        n = len(t)
        if isinstance(t, torch.Tensor):
            pinned[i, :n].copy_(t)
        else:
            pinned[i, :n] = torch.tensor(t, dtype=dtype)
    return pinned

def choose_bucket_len(max_len):
    idx = bisect.bisect_left(BUCKET_TOKENS, max_len)
    if idx < len(BUCKET_TOKENS):
        return BUCKET_TOKENS[idx]
    return max_len

def compute_thread_fn(loop):
    while True:
        uuid_str, padded_token, padded_token_len, futures = compute_queue.get()
        logging.info(f'{uuid_str}, enter compute_thread_fn')
        shapes = padded_token.shape

        with cuda.stream(compute_stream):
            compute_stream.wait_stream(h2d_stream)
            if shapes in buckets:
                logging.info(f'{uuid_str}, Hit compute shape {shapes}')
                recon = buckets[shapes](padded_token)
            else:
                recon = fn(padded_token)
            logging.info(f'{uuid_str}, done compute shape {shapes}')

            ys = recon.cpu()
        compute_stream.synchronize()
        for i, fut in enumerate(futures):
            out_len = padded_token_len[i] * 480
            ys_ = ys[i:i+1, :, :out_len]
            loop.call_soon_threadsafe(fut.set_result, ys_)

def batch_thread_fn():
    while True:
        uuid_str, batch = batch_queue.get()
        logging.info(f'{uuid_str}, enter batch_thread_fn')
        futures, tokens = zip(*[(b[0], b[1]) for b in batch])

        padded_token_len = [len(t) for t in tokens]
        max_len = max(padded_token_len)
        target_T = choose_bucket_len(max_len)
        padded_token = make_pinned_batch(tokens, target_T)
        shapes = padded_token.shape
        logging.info(f'{uuid_str}, batch shape {shapes} cpu')

        with cuda.stream(h2d_stream):
            padded_token_gpu = padded_token.to(device, non_blocking=True)
        compute_queue.put((uuid_str, padded_token_gpu, padded_token_len, futures))

async def dynamic_batching():
    need_sleep = True
    while True:
        if need_sleep:
            await asyncio.sleep(MICROSLEEP)
        need_sleep = True
        batch = []
        while not dynamic_batch_queue.empty():
            try:
                request = await asyncio.wait_for(dynamic_batch_queue.get(), timeout=1e-9)
                batch.append(request)
                if len(batch) >= MAX_BATCH_SIZE:
                    need_sleep = False
                    break
            except asyncio.TimeoutError:
                break
        if not len(batch):
            continue

        uuid_str = str(uuid.uuid4())
        logging.info(f'{uuid_str}, dynamic batching size {len(batch)}')
        batch_queue.put((uuid_str, batch))

async def decode_speech_token(speech_token):
    numbers = re.findall(r's_(\d+)', speech_token)
    d = list(map(int, numbers))
    if DYNAMIC_BATCHING:
        future = asyncio.Future()
        await dynamic_batch_queue.put((future, d))
        y_gen = await future
    else:
        audio_codes = torch.tensor(d)[None, None]
        y_gen = codec.decode_code(audio_codes.to(device))

    return (sr, y_gen[0, 0].cpu().numpy())

class TTSRequest(BaseModel):
    input: str = "Hello! How can I help you?"
    voice: str = DEFAULT_SPEAKER
    model: str = MODEL_NAME
    response_format: Literal["pcm", "wav"] = "pcm"
    temperature: float = DEFAULT_TEMPERATURE
    repetition_penalty: float = DEFAULT_REPETITION_PENALTY
    max_tokens: int = DEFAULT_MAX_TOKENS
    stream: bool = True
    playback_speed: float = DEFAULT_PLAYBACK_SPEED
    playback_overlap_speed: float = DEFAULT_PLAYBACK_OVERLAP_SPEED
    normalize_malaysian: bool = True

@app.get('/v1/audio/speaker')
async def speaker():
    return SPEAKERS

@app.post('/v1/audio/speech')
async def tts_stream(data: TTSRequest, request: Request = None):
    speaker = data.voice

    s = data.input
    logging.debug(f'input: {s}')
    if data.normalize_malaysian:

        s = s.replace('\n', ' ')
        s = re.sub(r'[ ]+', ' ', s).strip()
        
        lang = lang_model.predict(s, k = 3)[0][0]
        normalize_in_english = 'english' in lang
        
        def replace_range(match):
            num1 = int(match.group(1))
            num2 = int(match.group(2))
            words1 = to_cardinal(num1, english=normalize_in_english)
            words2 = to_cardinal(num2, english=normalize_in_english)
            phrase = match.group(3)
            if normalize_in_english:
                to = 'to'
            else:
                to = 'hingga'
            return f"{words1} {to} {words2} {phrase}"

        for k, v in before_replace_mapping.items():
            s = s.replace(k, v)
        
        s = expand_contractions(s)
        s = pattern_range.sub(replace_range, s)
        new_s = []
        for w in s.split():
            splitted = split_alpha_num(w).split()
            for i in range(len(splitted)):
                if len(splitted[i]) == 1:
                    splitted[i] = splitted[i].upper()
            splitted = ' '.join(splitted)
            new_s.append(splitted)
        s = ' '.join(new_s)

        logging.debug(f'out from internal normalizer: {s}')
        
        string = normalizer.normalize(
            s, 
            normalize_hingga = False, 
            normalize_text = False, 
            normalize_word_rules = False, 
            normalize_time = True, 
            normalize_cardinal = False,
            normalize_ordinal = False,
            normalize_url = True,
            normalize_email = True,
            normalize_in_english=normalize_in_english,
        )
        s = string['normalize']
        logging.debug(f'out from malaya normalizer: {s}')

        for k, v in replace_mapping.items():
            s = s.replace(k, v)
        
        original_s = s
        s = apply_pronunciation_replacements(s)
        if s != original_s:
            logging.debug(f'Pronunciation replacements applied: "{original_s}" -> "{s}"')
        
        s = re.sub(r'[ ]+', ' ', s).strip()

    prompt = f'<|im_start|>{speaker}: {s}<|speech_start|>'
    logging.debug(f'prompt: {prompt}')
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json',
    }

    json_data = {
        'model': data.model,
        'prompt': prompt,
        'max_tokens': data.max_tokens,
        'temperature': data.temperature,
        'repetition_penalty': data.repetition_penalty,
        'stream': True,
    }

    queue = asyncio.Queue()

    async def generate_audio_stream():
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    TTS_API,
                    headers=headers,
                    json=json_data,
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        logging.error(f"TTS backend error: {resp.status} - {error_text}")
                        return

                    async for line in resp.content:
                        if await request.is_disconnected():
                            break
                        if line.startswith(b"data: "):
                            data_str = line.decode("utf-8").strip()[6:]
                            if data_str == "[DONE]":
                                break
                            try:
                                data_json = json.loads(data_str)
                                delta = data_json["choices"][0]
                                if "text" in delta:
                                    await queue.put({'result': delta["text"]})
                            except json.JSONDecodeError:
                                continue

                    await queue.put(None)
                    
        except Exception as e:
            await queue.put({'error': str(e)})
            return
    
    asyncio.create_task(generate_audio_stream())
    
    chunk_size = int(data.playback_speed * codebook_size)
    overlap = int(data.playback_overlap_speed * codebook_size)
    overlap_chunk = int((overlap / codebook_size) * sr)
    fade_len = int(sr * 0.002)

    async def audio_stream():
        buffer = []
        to_yield = 0
        count = 0

        while True:
            try:
                output = queue.get_nowait()
            except asyncio.QueueEmpty:
                await asyncio.sleep(1e-9)
                continue
                
            if output is None:
                break

            if "error" in output:
                raise HTTPException(status_code=400, detail=output["error"])
            
            output = output["result"]
                
            buffer.append(output)
            
            if len(buffer) % chunk_size == 0:
                _, y = await decode_speech_token("".join(buffer))
                y_ = y[to_yield : -overlap_chunk]
                y_ = np.clip(y_, -1.0, 1.0)
                if len(y_) > fade_len:
                    y_[-fade_len:] *= np.linspace(1, 0, fade_len)

                if DEBUG_AUDIO:
                    sf.write(f'/app/app/{count}.wav', y_, sr)

                yield (y_ * 32767).astype(np.int16).tobytes()
                await asyncio.sleep(0)

                if to_yield == 0:
                    to_yield = len(y) - to_yield - overlap_chunk
        
                buffer = buffer[-chunk_size:]
                count += 1

        if len(buffer):
            _, y = await decode_speech_token("".join(buffer))
            y_ = y[to_yield :]
            y_ = np.clip(y_, -1.0, 1.0)
            if len(y_) > fade_len:
                y_[-fade_len:] *= np.linspace(1, 0, fade_len)
            
            if DEBUG_AUDIO:
                sf.write(f'/app/app/{count}.wav', y_, sr)

            yield (y_ * 32767).astype(np.int16).tobytes()
            await asyncio.sleep(0)

    media_type = f"audio/{data.response_format}"
    func = audio_stream()
    if data.stream:
        return StreamingResponse(func, media_type=media_type)
    else:
        ys = []
        async for y_ in func:
            ys.append(y_)
        merged_bytes = b"".join(ys)

        if data.response_format == 'wav':
            bio = io.BytesIO()
            with wave.open(bio, 'wb') as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(sr)
                w.writeframes(merged_bytes)
            
            wav_bytes = bio.getvalue()
            headers = {
                "Content-Type": "audio/wav",
                "Accept-Ranges": "bytes",
                "Content-Length": str(len(wav_bytes)),
            }
            return Response(content=wav_bytes, headers=headers)
            
        else:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pcm")
            tmp.write(merged_bytes)
            tmp.close()
            return FileResponse(
                path=tmp.name,
                media_type="audio/L16; rate=24000; channels=1",
                filename="merged_audio.pcm"
            )
    
if len(SENTRY_DSN):
    @app.get("/sentry-debug")
    async def trigger_error():
        division_by_zero = 1 / 0

app.state.background_dynamic_batching = asyncio.create_task(dynamic_batching())

loop = asyncio.get_running_loop()
t1 = threading.Thread(
    target=batch_thread_fn,
    daemon=True
)
t1.start()
t2 = threading.Thread(
    target=compute_thread_fn,
    args=(loop,),
    daemon=True
)
t2.start()
app.state.batch_thread = t1
app.state.compute_thread = t2