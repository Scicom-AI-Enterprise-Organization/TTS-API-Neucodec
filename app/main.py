from app.env import *

import torch

torch._dynamo.config.recompile_limit = 128
torch.set_float32_matmul_precision('high')

from typing import Literal
import re
import json
import struct
import asyncio
import io
import wave
import tempfile
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from fastapi import FastAPI, Request, HTTPException, File, Form
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
import librosa
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

vc_compute_queue = thread_queue.Queue()
vc_batch_queue = thread_queue.Queue()
vc_dynamic_batch_queue = asyncio.Queue()

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

        with torch.no_grad():
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

def vc_compute_thread_fn(loop):
    while True:
        uuid_str, ys, futures = vc_compute_queue.get()
        logging.info(f'{uuid_str}, enter vc_compute_thread_fn, batch size {len(ys)}')
        try:
            tokens = batch_encode(ys)
            for i, fut in enumerate(futures):
                loop.call_soon_threadsafe(fut.set_result, tokens[i])
        except Exception as e:
            for fut in futures:
                loop.call_soon_threadsafe(fut.set_exception, e)

def vc_batch_thread_fn(loop):
    while True:
        uuid_str, batch = vc_batch_queue.get()
        logging.info(f'{uuid_str}, enter vc_batch_thread_fn, batch size {len(batch)}')
        futures, ys = zip(*[(b[0], b[1]) for b in batch])
        vc_compute_queue.put((uuid_str, list(ys), futures))

async def vc_dynamic_batching():
    need_sleep = True
    while True:
        if need_sleep:
            await asyncio.sleep(MICROSLEEP)
        need_sleep = True
        batch = []
        while not vc_dynamic_batch_queue.empty():
            try:
                request = await asyncio.wait_for(vc_dynamic_batch_queue.get(), timeout=1e-9)
                batch.append(request)
                if len(batch) >= MAX_BATCH_SIZE:
                    need_sleep = False
                    break
            except asyncio.TimeoutError:
                break
        if not len(batch):
            continue

        uuid_str = str(uuid.uuid4())
        logging.info(f'{uuid_str}, vc dynamic batching size {len(batch)}')
        vc_batch_queue.put((uuid_str, batch))

async def encode_audio(y):
    if DYNAMIC_BATCHING:
        future = asyncio.Future()
        await vc_dynamic_batch_queue.put((future, y))
        tokens = await future
    else:
        tokens = batch_encode([y])
        tokens = tokens[0]
    return tokens

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

async def stream_speech(
    prompt,
    model,
    max_tokens,
    temperature,
    repetition_penalty,
    playback_speed,
    playback_overlap_speed,
    response_format,
    stream,
    request,
):
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json',
    }

    json_data = {
        'model': model,
        'prompt': prompt,
        'max_tokens': max_tokens,
        'temperature': temperature,
        'repetition_penalty': repetition_penalty,
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
                        logging.error(f"Backend error: {resp.status} - {error_text}")
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

    chunk_size = int(playback_speed * codebook_size)
    overlap = int(playback_overlap_speed * codebook_size)
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

    func = audio_stream()
    if stream:
        if response_format == 'wav':
            async def wav_stream():
                max_data = 0x7FFFFFFF - 36
                wav_header = struct.pack('<4sI4s', b'RIFF', max_data + 36, b'WAVE')
                wav_header += struct.pack('<4sIHHIIHH', b'fmt ', 16, 1, 1, sr, sr * 2, 2, 16)
                wav_header += struct.pack('<4sI', b'data', max_data)
                yield wav_header
                async for chunk in func:
                    yield chunk
            return StreamingResponse(wav_stream(), media_type="audio/wav")
        else:
            return StreamingResponse(func, media_type="audio/pcm")
    else:
        ys = []
        async for y_ in func:
            ys.append(y_)
        merged_bytes = b"".join(ys)

        if response_format == 'wav':
            bio = io.BytesIO()
            with wave.open(bio, 'wb') as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(sr)
                w.writeframes(merged_bytes)

            wav_bytes = bio.getvalue()
            resp_headers = {
                "Content-Type": "audio/wav",
                "Accept-Ranges": "bytes",
                "Content-Length": str(len(wav_bytes)),
            }
            return Response(content=wav_bytes, headers=resp_headers)

        else:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pcm")
            tmp.write(merged_bytes)
            tmp.close()
            return FileResponse(
                path=tmp.name,
                media_type="audio/L16; rate=24000; channels=1",
                filename="merged_audio.pcm"
            )

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

    return await stream_speech(
        prompt=prompt,
        model=data.model,
        max_tokens=data.max_tokens,
        temperature=data.temperature,
        repetition_penalty=data.repetition_penalty,
        playback_speed=data.playback_speed,
        playback_overlap_speed=data.playback_overlap_speed,
        response_format=data.response_format,
        stream=data.stream,
        request=request,
    )
    
def batch_encode(ys):
    with torch.no_grad():
        ys_pt = [codec._prepare_audio(torch.tensor(ys[i])[None, None])[0, 0] for i in range(len(ys))]
        features = codec.feature_extractor(ys_pt, sampling_rate=16_000, return_tensors="pt")
        semantic_features = features.input_features.to('cuda')
        padded = pad_sequence(ys_pt, batch_first=True)[:,None]
        acoustic_emb = codec.CodecEnc(padded.to('cuda'))
        acoustic_emb = acoustic_emb.transpose(1, 2)
        semantic_output = (
            codec.semantic_model(semantic_features).hidden_states[16].transpose(1, 2)
        )
        semantic_encoded = codec.SemanticEncoder_module(semantic_output)
        if acoustic_emb.shape[-1] != semantic_encoded.shape[-1]:
            min_len = min(acoustic_emb.shape[-1], semantic_encoded.shape[-1])
            acoustic_emb = acoustic_emb[:, :, :min_len]
            semantic_encoded = semantic_encoded[:, :, :min_len]        
        concat_emb = torch.cat([semantic_encoded, acoustic_emb], dim=1)
        concat_emb = codec.fc_prior(concat_emb.transpose(1, 2)).transpose(1, 2)

        _, fsq_codes, _ = codec.generator(concat_emb, vq=True)
        lens = features.attention_mask.sum(dim=1)
        fsq_codes = fsq_codes.cpu()

        tokens = []
        for i in range(lens.shape[0]):
            tokens.append(fsq_codes[i, 0, :lens[i]])
        
        return tokens


@app.post('/v1/audio/vc')
async def vc_stream(
    reference_audio: bytes = File(..., description="Reference audio"),
    reference_text: str = Form(..., description="Reference text"),
    generate_text: str = Form(..., description="Text to generate"),
    model: str = Form(default=MODEL_NAME, description="Model name"),
    response_format: str = Form(default="pcm", description="Response format: pcm or wav"),
    temperature: float = Form(default=DEFAULT_TEMPERATURE, description="Temperature"),
    repetition_penalty: float = Form(default=DEFAULT_REPETITION_PENALTY, description="Repetition penalty"),
    max_tokens: int = Form(default=DEFAULT_MAX_TOKENS, description="Max tokens"),
    stream: bool = Form(default=True, description="Stream response"),
    playback_speed: float = Form(default=DEFAULT_PLAYBACK_SPEED, description="Playback speed"),
    playback_overlap_speed: float = Form(default=DEFAULT_PLAYBACK_OVERLAP_SPEED, description="Playback overlap speed"),
    request: Request = None
):
    file_like = io.BytesIO(reference_audio)
    y, _ = librosa.load(file_like, sr=16000)

    codes = await encode_audio(y)

    tokens = ''.join([f'<|s_{i}|>' for i in codes])
    prompt = f"<|im_start|>{reference_text}<|speech_start|>{tokens}<|im_end|><|im_start|>{generate_text}<|speech_start|>"
    logging.debug(f'prompt: {prompt}')

    return await stream_speech(
        prompt=prompt,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        playback_speed=playback_speed,
        playback_overlap_speed=playback_overlap_speed,
        response_format=response_format,
        stream=stream,
        request=request,
    )

if len(SENTRY_DSN):
    @app.get("/sentry-debug")
    async def trigger_error():
        division_by_zero = 1 / 0

app.state.background_dynamic_batching = asyncio.create_task(dynamic_batching())
app.state.background_vc_dynamic_batching = asyncio.create_task(vc_dynamic_batching())

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
t3 = threading.Thread(
    target=vc_batch_thread_fn,
    args=(loop,),
    daemon=True
)
t3.start()
t4 = threading.Thread(
    target=vc_compute_thread_fn,
    args=(loop,),
    daemon=True
)
t4.start()
app.state.batch_thread = t1
app.state.compute_thread = t2
app.state.vc_batch_thread = t3
app.state.vc_compute_thread = t4