# Verifying the silent-wedge fix

The bug: after long runs at high concurrency, `/v1/audio/speech` and `/v1/audio/vc` stop
responding with nothing in the logs. Three independent silent-failure paths caused it.
This document is how to prove each one is fixed, without waiting hours for it to happen
in production.

The key move in every test below is the **A/B**: run the test against the fixed code,
then `git stash` the fix and run it again. A test that passes on both is testing nothing.

## Test 1 — leaked spin-loops (the primary cause)

A vLLM non-200 (429/503/timeout, i.e. what concurrency produces) used to make the
producer return without a queue terminator, leaving the consumer spinning on
`get_nowait()` + `sleep(1e-9)` forever on the single-threaded event loop.

`bench/wedge_test.py` stands a fake LM in place of vLLM so the failure fires on demand.

```bash
# terminal 1 — fake LM (no GPU, no model)
python bench/wedge_test.py serve --port 9095

# terminal 2 — the app, pointed at the fake LM (GPU box; NeuCodec still loads)
TTS_API=http://localhost:9095 uvicorn app.main:app --host 0.0.0.0 --port 9091

# terminal 3
python bench/wedge_test.py run --app http://localhost:9091 --lm http://localhost:9095
```

Three phases: **baseline** (healthy, records TTFB) → **fault** (30 requests, LM returns
503) → **recovery** (healthy again). The assertion that matters is the third one: the
bug was never that faulty requests fail, it is that they poison every request after.

- **Fixed:** fault requests return an error promptly; recovery TTFB within 3× baseline.
- **Pre-fix:** fault requests never return (reported as `HUNG`), and recovery latency
  collapses or times out. Watch `top` — the app process pegs a core and stays there
  after the load stops. That stuck-at-100%-CPU-while-idle signature *is* the bug.

Worth running `--faults 100` too: the degradation is cumulative, one leaked spinner per
backend error.

## Test 2 — worker-thread death

`compute_thread_fn` / `batch_thread_fn` had no exception handling. One CUDA OOM killed
the thread; the process stayed up, the queue was never drained again, and every later
decode hung on an unresolved future. Reproducing this needs a fault injected into the
decode path, so it ships as a patch rather than production code:

```bash
git apply bench/fault_inject_decode.patch
FAULT_INJECT_DECODE_N=5 TTS_API=http://localhost:9095 \
  uvicorn app.main:app --host 0.0.0.0 --port 9091   # 5th decode raises

python bench/wedge_test.py run --app http://localhost:9091 --lm http://localhost:9095

git apply -R bench/fault_inject_decode.patch        # revert -- never deploy this
```

- **Fixed:** one request fails with a logged `compute_thread_fn failed` traceback;
  every request after it succeeds.
- **Pre-fix:** the request that trips the fault hangs, and so does *every subsequent
  request* — permanently, with no log line. Kill required.

## Test 3 — unbounded future waits

`BATCH_TIMEOUT` (default 120s, `0` disables) bounds how long a request waits on a
decode/encode future. Verify it independently of the above:

```bash
# make the timeout short enough to observe, then trip a worker fault
git apply bench/fault_inject_decode.patch
BATCH_TIMEOUT=5 FAULT_INJECT_DECODE_N=5 ... uvicorn app.main:app ...
```

The faulted request should return **503 within ~5s** rather than hanging. Note that
with Test 2's fix in place the future is failed directly, so the timeout is a backstop
for a *wedged* worker (one stuck inside a CUDA call), not a crashed one — to see it
fire, change the injected `raise` to `time.sleep(3600)`.

## What this does not cover

These tests run the streaming TTS path. `/v1/audio/vc` shares the same batching threads
and got the same guards, but the harness does not drive it — the VC path is exercised
by `tests/test_tts_vc_api.py` against a live instance.

Whisper CER is unaffected: none of these changes touch decode math. Re-run
`bench/cer_eval.py` only if you want the guardrail number refreshed.
