# Speech context across chunked requests (`request_id`)

How consecutive TTS requests are made to sound like one continuous utterance, how the
history is bounded, and how it is shared between uvicorn workers. Code:
[`app/context.py`](app/context.py) (all the logic, no torch), the `request_id` plumbing
in [`app/main.py`](app/main.py), tests in [`tests/test_context.py`](tests/test_context.py).

---

## 1. The problem

LiveKit's agent framework does not send the TTS one reply at a time. The `openai.TTS`
plugin is a *non-streaming* TTS, so `livekit-agents` wraps it in a `StreamAdapter` that
cuts the LLM's text into sentence-ish chunks (`min_sentence_len=20` chars, so often 5–6
words) and synthesizes each chunk with its **own** `/v1/audio/speech` request — strictly one
after the other (`StreamAdapterWrapper._synthesize` awaits the whole audio of chunk N before
asking for chunk N+1).

Each of those requests used to be an independent prompt:

```
request 1:  <|im_start|>husein: hello my name is husein,<|speech_start|>
request 2:  <|im_start|>husein: i like to eat chicken rice.<|speech_start|>
```

The LM has no idea request 2 continues request 1. It starts cold: sentence-initial pitch
reset, a fresh pace, a fresh energy level, sampled independently (temperature 0.6). Every
join therefore sounds like a new speaker taking a breath and starting over, even though the
model itself renders the same paragraph naturally when it gets it in one prompt. The model
was trained on long context — the chunking threw that away.

## 2. The idea: give the LM the previous chunk as in-context history

The model already has a multi-turn format. `/v1/audio/vc` uses it to prime a reference
voice:

```
<|im_start|>{reference_text}<|speech_start|>{reference speech tokens}<|im_end|><|im_start|>{text}<|speech_start|>
```

i.e. *"here is some text and the speech tokens that go with it; now continue with this text."*
That is exactly what a chunked conversation needs, with the previous chunk(s) as the
"reference". So a request that carries a `request_id` with history behind it is prompted as:

```
<|im_start|>husein: hello my name is husein,<|speech_start|><|s_1834|><|s_209|>…<|s_77|><|im_end|>
<|im_start|>husein: i like to eat chicken rice.<|speech_start|>
                                                              ↑ the LM generates from here
```

- The history turns carry the **normalized text that was actually prompted** and the
  **speech tokens the LM actually produced** for it — not re-encoded audio, the real
  tokens, so the LM sees its own previous output as context.
- Only the new turn's tokens are generated, decoded by NeuCodec and streamed. The
  previous tokens are prefill only: ~1000 tokens of prefill on a ~12.8k tok/s LM is tens
  of milliseconds, and the codec path (the actual bottleneck) is untouched.
- The LM continues in the prosodic state it left off in — same register, pace, energy —
  instead of restarting.
- A `request_id` with no history yet builds *exactly* the old single-turn prompt
  (`build_prompt([], voice, text)`), so the first chunk pays nothing and requests without
  an id are byte-for-byte unchanged.

Where the history comes from: when a turn's LM stream finishes cleanly, its
`(text, token_ids)` pair is appended to the id's history for the next chunk.

### 2b. Two ways to put the history in the prompt (`context_mode`)

The format above is `context_mode: "turns"` (default `CONTEXT_MODE=turns`). The first A/B on
tm-h20 (§9) showed its limit: the LM gets the previous chunks as *conditioning*, but a new
`<|im_start|>` turn is still a new utterance — the sentence-initial pitch/energy reset at each
join did not shrink. Hence a second mode, `"continue"`, which builds **one** turn:

```
<|im_start|>husein: hello my name is husein, i like to eat chicken rice.<|speech_start|><|s_1834|>…<|s_77|>
                                                                                         ↑ previous chunk's tokens, then the LM generates
```

All the text (previous chunks + new), and the previous chunks' speech tokens already after
`<|speech_start|>`. That is *exactly* the state the LM is in while generating a long text — it
simply resumes and emits the tokens for the remaining words. `join_texts()` strips the `.` the
rule normalizer appends to a chunk that ended in `,` (`"husein,."` → `"husein,"`) so the joined
text reads as one sentence; the new chunk keeps its final `.` so the LM terminates. Everything
else (store, trim, `fit_context`, commit, headers) is shared; `X-Context-Mode` says which mode
served a response. Trade-off to listen for in `continue`: the previous tokens end with whatever
final fall/pause the LM produced when it believed the chunk was the whole utterance, so the
resume happens *after* that — the join is a pause, not a cold restart, but it is still a pause.

### 2c. The fallback guard (`CONTEXT_FALLBACK`)

Both modes share one failure the first runs exposed (§9): with history in the prompt the LM
sometimes decides the utterance is *already over* and emits end-of-speech after 0–8 tokens —
in `continue` mode because the prefix ends with the previous chunk's terminal fall and
silence (the very tokens after which it emitted `<|im_end|>` last time), in `turns` mode
occasionally for the same reason one turn later. Without context this never happened
(0/40). So the LM reader **holds the first tokens back** — `4 × words`, floored at 10 (0.2 s)
and capped at 50 (1 s), always below the first decode window of 110 tokens so nothing is
delayed — and if the stream ends before that many have arrived, it discards them and
re-issues the request with the plain single-turn prompt: exactly what a request without
`request_id` would have produced. Nothing has reached the stitcher at that point, so the
client sees a normal chunk; the switch is logged, an event on `lm.generate`, and
`tts.context_fallback=true` on `tts.stream`. A real rendering runs ~15–20 tokens per word,
so the threshold only trips on the collapse. Cost: one extra LM call on those chunks.

## 3. Request lifecycle

```
POST /v1/audio/speech {"input": "...", "voice": "husein", "request_id": "room-42"}
        │
        ▼
tts_stream()                                          app/main.py
   key = body.request_id  or  header X-Context-Id
   s   = normalize(input)                              (rule / llm normalizer, unchanged)
   context, max_tokens = load_context(key, voice, s, max_tokens)
        │  history = store.get(key)                    ← file/memory store, shared by workers
        │  history = select_voice(history, voice)      ← trailing run in this voice only
        │  turns, max_tokens = fit_context(...)        ← left trim + LM-window clamp (§4)
   prompt = build_prompt(turns, voice, s)              ← the multi-turn prompt above
        │
        ▼
stream_speech(prompt, ..., context=context)
   producer generate_audio_stream():   reads vLLM SSE
        for each delta:  queue.put(text)              ← stitcher decodes/streams as before
                         lm_text.append(text)          ← NEW: keep everything the LM said
                         finish_reason = delta.finish_reason
        on "[DONE]":     lm_done = True
        finally:
           if lm_done and finish_reason != 'length':
               ids = parse <|s_N|> from lm_text
               context.commit(ids)                     ← store.append(key, Turn(s, ids, voice, ts))
           queue.put_nowait(None)                      ← terminator, AFTER the commit
   consumer audio_stream_crossfade(): unchanged
   response headers: X-Context-Id / X-Context-Turns / X-Context-Tokens
```

Points worth noticing:

**The commit happens before the terminator, synchronously.** The stitcher cannot finish the
HTTP response until it has consumed the `None`, so by the time a client has received chunk N
in full, chunk N is already in the store. LiveKit requests chunk N+1 only after chunk N is
fully received, so N+1 always sees N — on whichever worker it lands. No await, no race, no
"eventually consistent". A store write is a few tens of microseconds (tmpfs), so doing it
inline on the event loop is cheaper than a thread hop.

**Only clean finishes are stored.** `finish_reason == 'length'` means `max_tokens` cut the
speech short: the text says one thing, the tokens stop halfway. Feeding that as context
teaches the LM "this speaker talks impossibly fast" — worse than no context. Client
disconnects (LiveKit barge-in) cancel the producer before `[DONE]`, so they don't store
either; the earlier complete chunks remain valid context.

**Nothing in the decode pipeline changed.** `decode_speech_token` → dynamic batching →
CUDA graphs are untouched; the batch-queue tuples keep their arity. The feature lives
entirely in prompt construction and the LM reader.

## 4. Bounding the history (left trim)

A call can run for minutes; the LM window is fixed at `--max-model-len 4096` tokens, and
speech tokens are 50/s, so the *entire* window is ~80 s of speech. Two limits apply, in
order (`fit_context` in `app/context.py`):

### 4a. `CONTEXT_MAX_S` — how much history to keep (default 20 s = 1000 tokens)

`trim_turns(turns, budget)` walks the history **from the newest turn backwards**:

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

The store applies the same trim on every append, so files stay ~1–6 KB regardless of how
long the call runs.

### 4b. The LM window — never let vLLM reject the request

vLLM does **not** truncate: `prompt_tokens + max_tokens > max_model_len` is a 400. The
request's default `max_tokens` is 3072, and 1000 context tokens + text + 3072 > 4096.

`fit_context` therefore:

1. Estimates the prompt size. Every `<|s_N|>` and special token is exactly one LM token;
   plain text is estimated at **one token per character** — right for CJK and a 3–4×
   overestimate for Latin script. Over-estimating only clamps `max_tokens` a bit harder
   (from 3072, which no sentence ever reaches); under-estimating would 400.
2. Sizes the context budget as
   `min(CONTEXT_MAX_S×50, max_model_len − new_text − min(max_tokens, CONTEXT_MIN_GEN_TOKENS) − margin)`,
   so the context can never squeeze generation below `CONTEXT_MIN_GEN_TOKENS` (1000 = 20 s
   of speech) — if the window is tight, **context gives way, not generation**.
3. Clamps `max_tokens` to what remains: `max_model_len − prompt_estimate − 16`.

With defaults: 1000 context tokens + ~500 chars of history text + a sentence ≈ 1600
estimated ⇒ `max_tokens` clamps from 3072 to ≈2480 (≈50 s of speech) — irrelevant for a
sentence chunk. `LM_MAX_MODEL_LEN` must match vLLM's `--max-model-len`.

### 4c. Voice switches

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
port-bind election), and when uvicorn recycles that worker every context on the host is
lost and a new election has to happen. Redis solves it properly but is a new service to
deploy and a network round-trip on the hot path, for state that is only ever needed on
this host.

### What is used instead: a shared directory in RAM

`FileContextStore` (`CONTEXT_STORE=file`, the default) keeps **one small JSON file per
context id** in `CONTEXT_STORE_DIR` — `/dev/shm/tts-context` by default, which is tmpfs, so
it's memory, not disk (docker's default 64 MB `/dev/shm` holds ~10k ids):

```
/dev/shm/tts-context/
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
- **TTL**: `updated` older than `CONTEXT_TTL_S` (600 s) reads as empty; a sweep deletes
  stale files, at most once per TTL/4 per process, skipping dot-files (the lock).
- **Order under overlapping requests**: turns are stored with the request's arrival time
  `ts` and kept sorted by it, so if a client ever overlaps two requests on one id and the
  shorter one finishes first, the prompt still lists them in text order.

Every worker on the host sees the same directory: nothing to elect, nothing to restart,
no new dependency. `MemoryContextStore` is the same thing in a dict (`CONTEXT_STORE=memory`,
single worker / tests); `CONTEXT_STORE=off` ignores `request_id` entirely. The
`ContextStore` base class is three methods (`get` / `append` / `delete`), so a redis
backend for multi-host deployments is a small addition — until then, chunks of one call
behind a multi-host load balancer must be pinned to one host, and `X-Context-Turns: 0` on
the response is the tell that they were not.

## 6. Getting the id from LiveKit

The livekit `openai.TTS` plugin drives the stock `openai` client and cannot add body
fields, but it accepts a pre-built client, and the client accepts default headers. So the
API also reads the id from the **`X-Context-Id` request header** (body field wins if both
are present), and the agent does:

```python
import openai as openai_sdk
from livekit.plugins import openai

tts_client = openai_sdk.AsyncClient(
    api_key="unused", base_url=TTS_BASE_URL,
    default_headers={"X-Context-Id": ctx.room.name},   # one context per room / session
)
session = AgentSession(tts=openai.TTS(model=..., voice=..., client=tts_client, response_format="pcm"))
```

One `AgentSession` = one TTS instance = one client = one room's id, so every sentence chunk
of that room is generated in the context of the ones before it, and rooms never share
history. `bench/livekit/stress_agent.py` does this (`TTS_CONTEXT=0` disables it for A/B).

Over plain HTTP:

```bash
curl -s -X POST localhost:9091/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"input":"hello my name is husein,","voice":"husein","request_id":"room-42","response_format":"wav","stream":false}' -o c1.wav
curl -s -D - -X POST localhost:9091/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"input":"i like to eat chicken rice.","voice":"husein","request_id":"room-42","response_format":"wav","stream":false}' -o c2.wav
#   X-Context-Turns: 1        ← chunk 1 was in the prompt
#   X-Context-Tokens: 137     ← its speech tokens (2.7 s)
curl -s localhost:9091/v1/audio/context/room-42            # stored turns (text + token counts)
curl -s -X DELETE localhost:9091/v1/audio/context/room-42  # forget now (else TTL)
```

## 7. Observability

- Response headers `X-Context-Id`, `X-Context-Turns`, `X-Context-Tokens` on every response
  shape (pcm, wav, SSE, buffered). Ids that are not latin-1 clean are percent-encoded in the
  header (Starlette would otherwise 500 encoding them).
- Log lines: `context <id>: 2/5 turns, 812 speech tokens in prompt, max_tokens 3072 -> 2610`
  on the way in, `context <id>: +137 tokens -> 3 turns / 949 tokens stored` on commit, or
  `turn not stored (lm_done=False, finish_reason=None)` when a disconnect/`length` skipped it.
  The `prompt:` log collapses token runs to `<812 speech tokens>` so it stays one line.
- Spans: `tts.context` (history size, prompt size, clamped `max_tokens`), and
  `tts.context_id/turns/tokens` + `lm.finish_reason` attributes on `tts.stream`/`lm.generate`.

## 8. Configuration

| Var | Default | Meaning |
|---|---|---|
| `CONTEXT_STORE` | `file` | `file` (shared by all workers on the host) / `memory` / `off` |
| `CONTEXT_STORE_DIR` | `/dev/shm/tts-context` | Directory for `file`; every worker must see the same one |
| `CONTEXT_MAX_S` | `20` | Seconds of previous speech tokens (×50) kept per id and prepended to the prompt |
| `CONTEXT_TTL_S` | `600` | Idle seconds before an id is forgotten |
| `CONTEXT_MIN_GEN_TOKENS` | `1000` | Generation room the context may never squeeze below |
| `LM_MAX_MODEL_LEN` | `4096` | vLLM's `--max-model-len`; prompt + `max_tokens` are sized against it |
| `CONTEXT_MODE` | `turns` | `turns` (closed VC-style turns, new chunk = new turn) or `continue` (one turn, previous tokens as prefix); request field `context_mode` overrides |
| `CONTEXT_FALLBACK` | `true` | Regenerate a chunk without history if the LM stops before `4 × words` tokens (§2c) |

## 9. What the tm-h20 A/B showed (2026-08-28)

Setup: branch running as slurm job on GPU 2 of tm-h20 (2 uvicorn workers, file store, same
vLLM engine as prod, rule normalizer, temperature 0.6), `bench/context_ab.py`: 4 texts
(EN husein, MS idayu, EN TM_English_Normal, the chicken-rice pair) × 2 takes × conditions
A (no context) / B1 (`turns`) / B2 (`continue`) / C (one request), 40 chunks per condition.
`bench/context_ab_metrics.py` measures, at every chunk join, the pitch jump (median F0,
semitones) and level jump (RMS, dB) between the last voiced ~0.3 s before and the first
voiced ~0.3 s after. The listening page (`bench/context_ab_page.py`) is what to actually
judge by — the metric is crude (pyin on 300 ms windows, sampled speech, per-chunk loudness
normalization in the stitcher) and its noise floor turned out to be ~5 semitones.

| run | A no context | B1 turns | B2 continue | C one request | note |
|---|---|---|---|---|---|
| 1 (turns only) | 5.6 st / 5.8 dB | 6.1 st / 5.8 dB | — | 2.7 st / 3.1 dB | B1 no better than A at the joins → added `continue` |
| 3 (both modes, no fallback) | 5.2 st / 4.5 dB | 5.7 st / 7.1 dB | 5.2 st / 5.3 dB | 5.2 st / 3.6 dB | **B2: 13/40 chunks empty (0 tokens), B1: 3/40 near-empty, A: 0/40** → added the fallback |
| 4 (both modes + fallback) | 5.6 st / 4.5 dB | 6.2 st / 5.1 dB | **4.3 st** / 8.1 dB | 3.6 st / 3.6 dB | **0 empty chunks** in 240; B1 still overran twice (8.9 s and 6.6 s for ~3 s sentences); B2 none. Latency p90 (chunks 2+): A 1.15 s, B1 1.00 s, B2 0.51 s |

Findings that matter more than the medians:

1. **The LM can decide the utterance is over.** Given the previous chunk's tokens — which end
   in a sentence-final fall and silence because that chunk was generated as a complete
   utterance — the LM frequently emits end-of-speech at once for the next text in `continue`
   mode (and once it does, every following chunk of that take too, since each resumes from
   the same kind of prefix). `turns` mode does it less often but not never. This is why the
   fallback exists; with it, no chunk in the smoke run (32) or run 4 (240) came back empty.
   Run 4 is the one on the listening page: A / B1 / B2 / C, 8 takes, with the join metrics
   under each row. Reading it: B2 has the smallest pitch jump of the chunked conditions
   (4.3 st, vs 5.6 cold and 3.6 for one request) and no failures; B1 gives no measurable
   benefit and still overruns; the level-jump column is dominated by the per-request gain
   (point 4). The ears decide — that is what the page is for.
2. **`continue` chunks are shorter** than their cold twins (e.g. 1.62 vs 2.28 s, 1.60 vs 1.84 s),
   consistent with mid-utterance pace and no sentence-initial/final pauses — the desired
   effect — but the listening test has to confirm no words are dropped.
3. **`turns` mode occasionally overruns** (one 6.2 s rendering of a 2.4 s sentence) — the
   VC-style history seems to invite the LM to keep going. Not guarded yet.
4. The per-chunk loudness normalization (`STREAM_NORMALIZE`) applies one gain **per
   request**, so part of the level jump at every join in A/B1/B2 is the stitcher's, not the
   LM's; C gets one gain for the whole text. A context-aware gain (carry the locked gain of
   the previous chunk through the store) is an obvious follow-up if B is adopted.

## 10. Verification and open points

- `uv run --with pytest -- pytest tests/test_context.py` — 34 tests, no GPU: trim semantics,
  proportional tail cut (words and CJK), voice filtering, prompt format, `fit_context`
  never exceeding the window and preferring generation over context, both stores (sharing
  between instances, TTL + sweep, corrupt-file recovery, hostile keys, 8-writer concurrency),
  header safety, commit never raising.
- `tests/test_tts_vc_api.py::TestTTSContext` — 7 live-API tests: second chunk reports
  `X-Context-Turns: 1` with a token count matching chunk 1's duration, `GET`/`DELETE`,
  header and alias forms, voice switch starts cold, headers on pcm/SSE streams. Run these
  against the GPU box; the `main.py` glue has not been exercised on a live LM yet.
- Two things to **listen for** on the first real run:
  1. The speaker prefix (`husein:`) is repeated on every history turn, mirroring the
     single-turn training format. If the model was trained multi-turn without it on
     history turns, `build_prompt` is a one-line change.
  2. vLLM's `repetition_penalty` (1.15) also covers prompt tokens, so the previous turns'
     codes are penalized — exactly as `/v1/audio/vc` has always done with its reference
     tokens. If continuity is good but timbre drifts, try a lower penalty on context
     requests.
