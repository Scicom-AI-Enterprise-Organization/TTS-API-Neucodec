# Interleaved generation (`interleave_id`)

How consecutive TTS requests are made to sound like one continuous utterance, how much is
retained, and how the history is shared between uvicorn workers. Code:
[`app/interleave.py`](app/interleave.py) (all the logic, no torch), the plumbing in
[`app/main.py`](app/main.py), tests in [`tests/test_interleave.py`](tests/test_interleave.py)
(44, no GPU) and `TestTTSInterleave` in
[`tests/test_tts_vc_api.py`](tests/test_tts_vc_api.py) (live API).

---

## 1. The problem

LiveKit's agent framework does not send the TTS one reply at a time. The `openai.TTS`
plugin is a *non-streaming* TTS, so `livekit-agents` wraps it in a `StreamAdapter` that
cuts the LLM's text into sentence-ish chunks (`min_sentence_len=20` chars, so often 5–6
words) and synthesizes each chunk with its **own** `/v1/audio/speech` request — strictly
one after the other (`StreamAdapterWrapper._synthesize` awaits the whole audio of chunk N
before asking for chunk N+1).

Each of those requests was an independent prompt:

```
request 1:  <|im_start|>husein: hello my name is husein,<|speech_start|>
request 2:  <|im_start|>husein: i like to eat chicken rice.<|speech_start|>
```

The LM at T+1 has no idea it continues T. It starts cold: sentence-initial pitch reset, a
fresh pace, a fresh energy level, sampled independently (temperature 0.6). Every join
sounds like a new speaker taking a breath and starting over — awkward — even though the
model renders the same paragraph naturally when it gets it in one prompt.

## 2. The idea: prompt the chunk the way the model was trained

This is not a trick bolted onto the LM; it is the shape it was **packed with**. The
GPUPlatform trainer (`gateway/gateway/training/tts/pack_stage1.py`, `--interleave_style
full`) writes the chunks of one recording as consecutive turns inside a single attention
document, so chunk 3 can attend to chunk 0's prosody:

```
<|im_start|>{spk}: {text_0}<|speech_start|>{audio_0}<|im_end|>
<|im_start|>{spk}: {text_1}<|speech_start|>{audio_1}<|im_end|> …      ← ONE document
```

> **This only works on a checkpoint packed that way, and that packing is ours alone.**
> `--interleave_style full` is an in-house GPUPlatform recipe — **no open-source TTS LM is trained
> on interleaved documents**, and the checkpoints that are are private model repos (the
> tm-h20 `HF_TOKEN` reads them; a laptop token gets 401). The benefit is therefore a property of
> the *training*, not of the prompt shape: measured on a private interleave-trained checkpoint it
> cuts the chunk N→N+1 register/level jump ~25% for free
> ([`bench/INTERLEAVE_AB.md`](bench/INTERLEAVE_AB.md)); on a model
> packed *without* it the same prompt bought nothing and collapsed 7.5% of chunks (§9). So
> **check what the served vLLM is actually loading before relying on this** — prod's `vllm-tts`
> serves `…-TMVoice-Synthetic`, which is not the interleave checkpoint — and set
> `INTERLEAVE_STORE=off` when it is a model without the packing.

That is a *prompt* shape as much as a training one. A request carrying an `interleave_id`
with history behind it is therefore prompted:

```
<|im_start|>husein: hello my name is husein,<|speech_start|><|s_1834|><|s_209|>…<|s_77|><|im_end|>
<|im_start|>husein: i like to eat chicken rice.<|speech_start|>
                                                              ↑ the LM generates from here
```

- History turns carry the **normalized text that was actually prompted** and the **speech
  tokens the LM actually produced** for it — not re-encoded audio, the real tokens, so the
  LM sees its own previous output.
- Only the new turn's tokens are generated, decoded by NeuCodec and streamed. The previous
  tokens are prefill only: ~750 tokens of prefill on a ~12.8k tok/s LM is tens of
  milliseconds, and the codec path (the actual bottleneck) is untouched.
- The LM continues in the prosodic state it left off in — same register, pace, energy —
  instead of restarting.
- An id with no history yet builds *exactly* the old single-turn prompt
  (`build_prompt([], voice, text)`), so the first chunk pays nothing and requests without
  an id are byte-for-byte unchanged.

Where the history comes from: when a turn's LM stream finishes cleanly, its
`(text, token_ids)` pair is appended to the id's history for the next chunk.

### 2b. The fallback guard (`INTERLEAVE_FALLBACK`)

With history in the prompt the LM can decide the utterance is *already over* and emit
end-of-speech after 0–8 tokens — the previous turn ends with the final fall and silence it
produced when it believed that chunk was the whole utterance. Measured on tm-h20 against
the pre-interleave model: 3/40 chunks; without history it never happened (0/40). So the LM
reader **holds the first tokens back** — `4 × words`, floored at 10 (0.2 s) and capped at
50 (1 s), always below the first decode window so nothing is delayed — and if the stream
ends before that many arrive, it discards them and re-issues the request with the plain
single-turn prompt: exactly what a request without an id would have produced. Nothing has
reached the stitcher at that point, so the client sees a normal chunk; the switch is
logged, an event on `lm.generate`, and `tts.interleave_fallback=true` on `tts.stream`. A
real rendering runs ~15–20 tokens per word, so the threshold only trips on the collapse.
Cost: one extra LM call on those chunks.

## 3. Request lifecycle

```
POST /v1/audio/speech {"input": "...", "voice": "husein", "interleave_id": "room-42"}
        │
        ▼
tts_stream()                                          app/main.py
   key = body.interleave_id  or  header X-Interleave-Id / X-Context-Id
   s   = normalize(input)                              (rule / spoken / llm, unchanged)
   il, max_tokens = load_interleave(key, voice, s, max_tokens, max_retain)
        │  history = store.get(key)                    ← file/memory store, shared by workers
        │  history = select_voice(history, voice)      ← trailing run in this voice only
        │  turns, max_tokens = fit_interleave(...)     ← retain cap + left trim + window clamp (§4)
   prompt = build_prompt(turns, voice, s)              ← the interleaved prompt above
        │
        ▼
stream_speech(prompt, ..., interleave=il)
   producer generate_audio_stream():   reads vLLM SSE
        for each delta:  queue.put(text)              ← stitcher decodes/streams as before
                         lm_text.append(text)          ← keep everything the LM said
                         finish_reason = delta.finish_reason
        on "[DONE]":     lm_done = True
        finally:
           if lm_done and finish_reason != 'length':
               ids = parse <|s_N|> from lm_text
               il.commit(ids)                          ← store.append(key, Turn(s, ids, voice, ts))
           queue.put_nowait(None)                      ← terminator, AFTER the commit
   consumer audio_stream_crossfade(): unchanged
   response headers: X-Interleave-Id / -Turns / -Tokens / -Max-Retain
```

Points worth noticing:

**The commit happens before the terminator, synchronously.** The stitcher cannot finish the
HTTP response until it has consumed the `None`, so by the time a client has received chunk
N in full, chunk N is already in the store. LiveKit requests chunk N+1 only after chunk N
is fully received, so N+1 always sees N — on whichever worker it lands. No await, no race,
no "eventually consistent". A store write is a few tens of microseconds (tmpfs), so doing
it inline on the event loop is cheaper than a thread hop.

**Only clean finishes are stored.** `finish_reason == 'length'` means `max_tokens` cut the
speech short: the text says one thing, the tokens stop halfway. Feeding that as history
teaches the LM "this speaker talks impossibly fast" — worse than nothing. Client
disconnects (LiveKit barge-in) cancel the producer before `[DONE]`, so they don't store
either; the earlier complete chunks remain valid history.

**Nothing in the decode pipeline changed.** `decode_speech_token` → dynamic batching →
CUDA graphs are untouched; the batch-queue tuples keep their arity. The feature lives
entirely in prompt construction and the LM reader.

## 4. Bounding the history

A call can run for minutes; the LM window is fixed at `--max-model-len 4096` tokens, and
speech tokens are 50/s, so the *entire* window is ~80 s of speech. Three limits apply, in
order (`fit_interleave` in `app/interleave.py`):

### 4a. `MAX_RETAIN_INTERLEAVE` — how many turns (default **5**)

The newest 5 turns, and only those, are retained per id and put in the prompt. Five
sentence-sized chunks is ~10–15 s of speech: enough for the LM to hold a register, far
short of the window. The cap is applied at both ends — the store never keeps more (so
files stay small however long the call runs), and a request may ask for fewer with the
`max_retain_interleave` field (asking for more gets what is there). `0` means no turn
limit, leaving §4b as the only bound.

### 4b. `INTERLEAVE_MAX_S` — how many seconds (default 20 s = 1000 tokens)

The turn cap does not bound *length*: one `say()` of a whole paragraph is a single turn.
`trim_turns(turns, budget)` therefore walks the history **from the newest turn backwards**:

1. A turn that fits whole is kept.
2. The first turn that does not fit whole is **cut to its tail** — its last `room` tokens —
   if at least `MIN_PARTIAL_TOKENS` (50 = 1 s) of room remain; then the walk stops.
   Everything older is dropped.

```
history:   [A: 600 tok] [B: 300 tok] [C: 300 tok]        budget 1000
kept:            [A tail: 400 tok] [B: 300] [C: 300]      = 1000
```

The partial turn's **text is shortened in the same proportion** (`trim_turn_tail`): last
`round(n_words × kept/total)` words, or characters for CJK text without spaces. It's a
heuristic alignment — speech rate inside one utterance is close to uniform, so the tail
text matches the tail audio to within a second or so — and it is what lets one long
previous `say()` still contribute its ending instead of being dropped whole. The most
recent seconds are what continuity needs anyway.

### 4c. The LM window — never let vLLM reject the request

vLLM does **not** truncate: `prompt_tokens + max_tokens > max_model_len` is a 400. The
request's default `max_tokens` is 3072, and 1000 history tokens + text + 3072 > 4096.

`fit_interleave` therefore:

1. Estimates the prompt size. Every `<|s_N|>` and special token is exactly one LM token;
   plain text is estimated at **one token per character** — right for CJK and a 3–4×
   overestimate for Latin script. Over-estimating only clamps `max_tokens` a bit harder
   (from 3072, which no sentence ever reaches); under-estimating would 400.
2. Sizes the history budget as
   `min(INTERLEAVE_MAX_S×50, max_model_len − new_text − min(max_tokens, INTERLEAVE_MIN_GEN_TOKENS) − margin)`,
   so history can never squeeze generation below `INTERLEAVE_MIN_GEN_TOKENS` (1000 = 20 s
   of speech) — if the window is tight, **history gives way, not generation**.
3. Clamps `max_tokens` to what remains: `max_model_len − prompt_estimate − 16`.

With the defaults, 5 sentence turns ≈ 750 speech tokens + ~250 chars of text ⇒
`max_tokens` clamps from 3072 to ≈2900 (≈58 s of speech) — irrelevant for a sentence
chunk. `LM_MAX_MODEL_LEN` must match vLLM's `--max-model-len`.

### 4d. Voice switches

`select_voice` keeps only the **trailing run** of turns in the requested voice. Priming
`idayu` with `husein`'s tokens is voice conversion, not continuity; a voice change on the
same id starts cold and the old voice's turns age out.

## 5. Sharing the history across workers

The API runs as `uvicorn --workers N` (+ NVIDIA MPS) to beat the GIL ceiling, and
consecutive chunks of one call land on arbitrary workers. The history therefore cannot
live in a process dict.

### Why not "worker 0 as leader"

A leader worker would have to run an extra server (socket/port) inside a uvicorn worker,
the others would need to discover it (uvicorn exposes no worker index, so: lock file or
port-bind election), and when uvicorn recycles that worker every id on the host is lost
and a new election has to happen. Redis solves it properly but is a new service to deploy
and a network round-trip on the hot path, for state that is only ever needed on this host.

### What is used instead: a shared directory in RAM

`FileInterleaveStore` (`INTERLEAVE_STORE=file`, the default) keeps **one small JSON file
per id** in `INTERLEAVE_STORE_DIR` — `/dev/shm/tts-interleave` by default, which is tmpfs,
so it's memory, not disk (docker's default 64 MB `/dev/shm` holds ~10k ids):

```
/dev/shm/tts-interleave/
  .lock                                   ← one directory-wide lock file, never deleted
  3f2a…c1.json  = {"updated": 1756280000.1,
                   "turns": [{"text": "hello my name is husein,", "tokens": [1834, 209, …],
                              "voice": "husein", "ts": 1756279998.4}, …]}
```

- **File name** = `sha1(id)` — any client string is a safe filename (tested with
  `../../etc/passwd`, slashes, emoji, 1000-char ids).
- **Atomic writes**: write to `<path>.<pid>.tmp`, then `os.replace` — a reader never sees
  a torn file (and a corrupt file, if one ever appeared, reads as empty and is overwritten).
- **No lost updates**: `append` is a read–modify–write, so it holds an exclusive `flock`
  on the single `.lock` file for the ~100 µs it takes. One lock for the whole directory
  rather than per key, because the data file's inode changes on every replace (a lock on
  it would not carry over) and a per-key lock file could be swept from under a holder.
  Verified with 8 concurrent writers × 25 appends on one key: all 200 turns present, in
  order, no `.tmp` leftovers, a concurrent reader never saw invalid JSON.
- **Reads take no lock** — `get` is one `open` + `json.load`.
- **TTL**: `updated` older than `INTERLEAVE_TTL_S` (600 s) reads as empty; a sweep deletes
  stale files, at most once per TTL/4 per process, skipping dot-files (the lock).
- **Order under overlapping requests**: turns are stored with the request's arrival time
  `ts` and kept sorted by it, so if a client ever overlaps two requests on one id and the
  shorter one finishes first, the prompt still lists them in text order.

Every worker on the host sees the same directory: nothing to elect, nothing to restart, no
new dependency. `MemoryInterleaveStore` is the same thing in a dict
(`INTERLEAVE_STORE=memory`, single worker / tests); `INTERLEAVE_STORE=off` ignores the id
entirely. The `InterleaveStore` base class is three methods (`get` / `append` / `delete`),
so a redis backend for multi-host deployments is a small addition — until then, chunks of
one call behind a multi-host load balancer must be pinned to one host, and
`X-Interleave-Turns: 0` on the response is the tell that they were not.

## 6. Getting the id from LiveKit

The livekit `openai.TTS` plugin drives the stock `openai` client and cannot add body
fields, but it accepts a pre-built client, and the client accepts default headers. So the
API also reads the id from the **`X-Interleave-Id` request header** (`X-Context-Id` is
accepted too; a body field wins over both), and the agent does:

```python
import openai as openai_sdk
from livekit.plugins import openai

tts_client = openai_sdk.AsyncClient(
    api_key="unused", base_url=TTS_BASE_URL,
    default_headers={"X-Interleave-Id": ctx.room.name},   # one id per room / session
)
session = AgentSession(tts=openai.TTS(model=..., voice=..., client=tts_client, response_format="pcm"))
```

One `AgentSession` = one TTS instance = one client = one room's id, so every sentence chunk
of that room is generated in the context of the ones before it, and rooms never share
history.

Over plain HTTP:

```bash
curl -s -X POST localhost:9091/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"input":"hello my name is husein,","voice":"husein","interleave_id":"room-42","response_format":"wav","stream":false}' -o c1.wav
curl -s -D - -X POST localhost:9091/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"input":"i like to eat chicken rice.","voice":"husein","interleave_id":"room-42","response_format":"wav","stream":false}' -o c2.wav
#   X-Interleave-Turns: 1        ← chunk 1 was in the prompt
#   X-Interleave-Tokens: 137     ← its speech tokens (2.7 s)
#   X-Interleave-Max-Retain: 5
curl -s localhost:9091/v1/audio/interleave/room-42            # retained turns (text + token counts)
curl -s -X DELETE localhost:9091/v1/audio/interleave/room-42  # forget now (else TTL)
```

## 7. Observability

- Response headers `X-Interleave-Id`, `-Turns`, `-Tokens`, `-Max-Retain` on every response
  shape (pcm, wav, SSE, buffered). Ids that are not latin-1 clean are percent-encoded in
  the header (Starlette would otherwise 500 encoding them).
- Log lines: `interleave <id>: 2/5 turns, 812 speech tokens in prompt, max_tokens 3072 ->
  2610` on the way in, `interleave <id>: +137 tokens -> 3 turns / 949 tokens retained` on
  commit, or `turn not stored (lm_done=False, finish_reason=None)` when a disconnect or a
  `length` finish skipped it. The `prompt:` log collapses token runs to
  `<812 speech tokens>` so it stays one line.
- Spans: `tts.interleave` (history size, prompt size, clamped `max_tokens`), and
  `tts.interleave_id/turns/tokens` + `lm.finish_reason` attributes on
  `tts.stream`/`lm.generate`.

## 8. Configuration

| Var | Default | Meaning |
|---|---|---|
| `INTERLEAVE_STORE` | `file` | `file` (shared by all workers on the host) / `memory` / `off` |
| `INTERLEAVE_STORE_DIR` | `/dev/shm/tts-interleave` | Directory for `file`; every worker must see the same one |
| `MAX_RETAIN_INTERLEAVE` | `5` | Previous turns retained per id and prompted (0 = seconds cap only) |
| `INTERLEAVE_MAX_S` | `20` | Seconds of previous speech tokens (×50) retained per id |
| `INTERLEAVE_TTL_S` | `600` | Idle seconds before an id is forgotten |
| `INTERLEAVE_MIN_GEN_TOKENS` | `1000` | Generation room the history may never squeeze below |
| `INTERLEAVE_FALLBACK` | `true` | Regenerate a chunk without history if the LM stops before `4 × words` tokens (§2b) |
| `LM_MAX_MODEL_LEN` | `4096` | vLLM's `--max-model-len`; prompt + `max_tokens` are sized against it |

Request fields: `interleave_id` (aliases `request_id`, `context_id`; header
`X-Interleave-Id` / `X-Context-Id`) and `max_retain_interleave` (default
`MAX_RETAIN_INTERLEAVE`).

## 9. Prior art in this repo, and what to listen for

An earlier attempt at the same idea lives on the unmerged branch `feat/speech-context`
(`SPEECH_CONTEXT.md`, `bench/context_ab*.py`, a listening page). Its tm-h20 A/B (2026-08-28,
40 chunks per condition) found this prompt shape gave **no measurable** reduction of the
pitch/level jump at the joins, and it also tried a second mode (`continue`: one turn, the
previous tokens as prefix) that scored better but made the LM emit end-of-speech
immediately on 13/40 chunks. That was against a model **not trained on interleaved
documents** — which is exactly what changed. Only the `full` turn format is implemented
here, because that is the one the packer writes; the fallback guard (§2b) survives from
that run.

**Does it actually help? Yes — measured 2026-09-07, [`bench/INTERLEAVE_AB.md`](bench/INTERLEAVE_AB.md).**
80 paragraphs (40 en + 40 ms) rendered three ways on the interleave-trained checkpoint: one
request for the whole paragraph, chunked with an `interleave_id`, and chunked cold. Over 438
matched consecutive-chunk pairs the interleaved prompt cuts the chunk N→N+1 discontinuity by
about a quarter — **−0.41 st of register step** [−0.56, −0.26] and **−0.45 dB of level step**
[−0.59, −0.32] — and on loudness cold chunking is *significantly worse than a one-shot rendering*
(+0.26 dB [+0.12, +0.41]) while interleaving is significantly better than one. It also recovers
~70% of the paragraph pitch declination cold chunking throws away (−0.137 vs −0.082 vs −0.161 st/s),
and at the seam itself cold joins step **up** +0.53 st (the register reset) where interleaved joins
step **down** −0.35 st. Costs: ~30 ms more silence per join (+1.7% duration), +399 prompt tokens
with **no latency change** (0.423 s vs 0.427 s), CER unchanged. Two things that new run settles
about this document:

- **§2b's fallback never fired: 0 collapses in 518 chunks** (against 3/40 on the pre-interleave
  model). Keep `INTERLEAVE_FALLBACK` as insurance for other checkpoints — it costs nothing unfired
  — but it is no longer a working part.
- **The old branch's null result was the model, not the idea.** Same prompt shape, same store, same
  bounds; only the checkpoint changed.

Still open:

1. **Speaker prefix on history turns.** `build_prompt` repeats `husein: ` on every turn,
   mirroring `pack_stage1.py --interleave_style full`. If a model was packed with
   `compact` instead (header once, later chunks bare), that is a one-line change.
2. **The extra 30 ms of silence at each join.** Trimming trailing-silence tokens from stored
   turns is the obvious lever.

Also worth knowing: vLLM's `repetition_penalty` (1.15) covers prompt tokens, so the
previous turns' codes are penalized — exactly as `/v1/audio/vc` has always done with its
reference tokens. If continuity is good but timbre drifts, try a lower penalty on
interleaved requests. And `STREAM_NORMALIZE` applies one loudness gain **per request**, so
part of any level jump at a join is the stitcher's, not the LM's; carrying the locked gain
through the store is the obvious follow-up.
