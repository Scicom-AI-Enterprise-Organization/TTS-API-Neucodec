"""Reproduce (and verify the fix for) the silent wedge under long-running load.

Symptom being tested: after a period of high concurrency the TTS/VC endpoints stop
responding, with no error logged anywhere. The trigger is a backend (vLLM) error
response -- 429/503/timeout, which is exactly what concurrency produces. Each one used
to strand a consumer coroutine in a `get_nowait()` + `sleep(1e-9)` spin that never
exits, permanently burning the single-threaded event loop. Enough of them and the app
answers nothing while looking perfectly healthy.

This harness stands in a fake LM in place of vLLM so the failure can be produced on
demand instead of waited for.

    # 1. fake LM (a stand-in for vLLM; no GPU, no model)
    python bench/wedge_test.py serve --port 9095

    # 2. the app, pointed at the fake LM (needs the GPU box for NeuCodec)
    TTS_API=http://localhost:9095 uvicorn app.main:app --host 0.0.0.0 --port 9091

    # 3. the test
    python bench/wedge_test.py run --app http://localhost:9091 --lm http://localhost:9095

To confirm the test actually detects the bug rather than passing vacuously, run it
against the pre-fix code: `git stash` the fix, restart the app, run it again. It should
fail at the FAULT phase (requests never return) or the RECOVERY phase (latency has
collapsed and never comes back).
"""

import argparse
import asyncio
import json
import random
import statistics
import sys
import time

import aiohttp
from aiohttp import web

# NeuCodec's codebook is 16384 entries; any id in range decodes to *some* audio. This
# test is about liveness, not audio quality, so random valid ids are fine.
CODEBOOK_SIZE = 16384
TOKENS_PER_REQUEST = 250          # ~5s of audio at 50 tokens/s -> several decode windows
TOKEN_DELAY = 0.004               # crude stand-in for LM decode speed


# --------------------------------------------------------------------------------
# fake LM
# --------------------------------------------------------------------------------

class FakeLM:
    """Serves /v1/completions like vLLM, and can be flipped to fail on command."""

    def __init__(self):
        self.mode = 'ok'          # 'ok' | 'fail503' | 'hang'
        self.requests = 0

    async def completions(self, request):
        self.requests += 1
        mode = self.mode

        if mode == 'fail503':
            # the failure that used to strand a consumer forever
            return web.Response(status=503, text='fake LM: service unavailable')

        if mode == 'hang':
            # backend accepts then stalls: exercises the client-side read path
            await asyncio.sleep(300)
            return web.Response(status=200, text='')

        resp = web.StreamResponse(
            status=200,
            headers={'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache'},
        )
        await resp.prepare(request)
        try:
            for _ in range(TOKENS_PER_REQUEST):
                tok = f'<|s_{random.randrange(CODEBOOK_SIZE)}|>'
                payload = {'choices': [{'text': tok}]}
                await resp.write(f'data: {json.dumps(payload)}\n\n'.encode())
                await asyncio.sleep(TOKEN_DELAY)
            await resp.write(b'data: [DONE]\n\n')
        except (ConnectionResetError, asyncio.CancelledError):
            pass
        return resp

    async def set_mode(self, request):
        self.mode = request.query.get('mode', 'ok')
        return web.json_response({'mode': self.mode, 'requests': self.requests})

    async def status(self, request):
        return web.json_response({'mode': self.mode, 'requests': self.requests})


def serve(port):
    lm = FakeLM()
    app = web.Application()
    app.router.add_post('/v1/completions', lm.completions)
    app.router.add_get('/mode', lm.set_mode)
    app.router.add_get('/status', lm.status)
    print(f'fake LM on :{port} -- point the app at TTS_API=http://localhost:{port}')
    web.run_app(app, port=port, print=None)


# --------------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------------

async def one_request(session, app_url, timeout_s):
    """Returns (ok, ttfb_or_None, note). A client-side timeout means the app hung."""
    body = {
        'input': 'Testing the streaming path under fault injection.',
        'voice': 'husein',
        'response_format': 'wav',
        'stream': True,
    }
    t0 = time.perf_counter()
    try:
        timeout = aiohttp.ClientTimeout(total=timeout_s)
        async with session.post(
            f'{app_url}/v1/audio/speech', json=body, timeout=timeout
        ) as resp:
            ttfb = None
            nbytes = 0
            async for chunk in resp.content.iter_any():
                if ttfb is None:
                    ttfb = time.perf_counter() - t0
                nbytes += len(chunk)
            if resp.status != 200:
                return (False, ttfb, f'http {resp.status}')
            if nbytes == 0:
                return (False, ttfb, 'empty body')
            return (True, ttfb, f'{nbytes}B')
    except asyncio.TimeoutError:
        return (False, None, f'HUNG (no response in {timeout_s}s)')
    except Exception as e:
        return (False, None, f'{type(e).__name__}: {e}')


async def phase(session, app_url, n, concurrency, timeout_s, label):
    print(f'\n--- {label}: {n} requests, concurrency {concurrency} ---')
    sem = asyncio.Semaphore(concurrency)

    async def guarded():
        async with sem:
            return await one_request(session, app_url, timeout_s)

    t0 = time.perf_counter()
    results = await asyncio.gather(*[guarded() for _ in range(n)])
    wall = time.perf_counter() - t0

    ok = [r for r in results if r[0]]
    hung = [r for r in results if r[2].startswith('HUNG')]
    ttfbs = [r[1] for r in results if r[1] is not None]
    notes = {}
    for r in results:
        key = r[2] if not r[0] else 'ok'
        notes[key] = notes.get(key, 0) + 1

    print(f'  wall {wall:.1f}s | ok {len(ok)}/{n} | hung {len(hung)}')
    print(f'  outcomes: {notes}')
    if ttfbs:
        print(f'  ttfb mean {statistics.mean(ttfbs):.2f}s  median {statistics.median(ttfbs):.2f}s')
    return {
        'n': n, 'ok': len(ok), 'hung': len(hung), 'wall': wall,
        'ttfb_mean': statistics.mean(ttfbs) if ttfbs else None,
    }


async def set_mode(session, lm_url, mode):
    async with session.get(f'{lm_url}/mode', params={'mode': mode}) as r:
        await r.json()
    print(f'\n[fake LM mode -> {mode}]')


async def run(app_url, lm_url, faults, timeout_s):
    failures = []
    async with aiohttp.ClientSession() as session:
        # 1. Baseline: how fast is a healthy request?
        await set_mode(session, lm_url, 'ok')
        base = await phase(session, app_url, 5, 1, timeout_s, 'BASELINE (healthy)')
        if base['ok'] == 0:
            print('\nFAIL: baseline requests do not work -- fix the setup before testing.')
            return 1

        # 2. Fault: every request hits a 503 from the LM. Pre-fix, each of these
        #    strands a spinning coroutine; the requests themselves never return.
        await set_mode(session, lm_url, 'fail503')
        fault = await phase(session, app_url, faults, 8, timeout_s, 'FAULT (LM returns 503)')
        if fault['hung']:
            failures.append(
                f"{fault['hung']}/{fault['n']} requests hung during the fault phase -- "
                'the producer is not terminating its queue (the original bug)'
            )

        # 3. Recovery: the LM is healthy again. This is the real assertion -- the bug
        #    is not that faulty requests fail, it is that they poison everything after.
        await set_mode(session, lm_url, 'ok')
        rec = await phase(session, app_url, 5, 1, timeout_s, 'RECOVERY (healthy again)')
        if rec['ok'] < 5:
            failures.append(
                f"only {rec['ok']}/5 requests succeeded after the fault burst -- "
                'the app did not recover'
            )
        elif base['ttfb_mean'] and rec['ttfb_mean']:
            ratio = rec['ttfb_mean'] / base['ttfb_mean']
            print(f'\n  recovery ttfb / baseline ttfb = {ratio:.2f}x')
            if ratio > 3.0:
                failures.append(
                    f'post-fault latency is {ratio:.1f}x baseline -- leaked spin-loops '
                    'are still starving the event loop'
                )

    print('\n' + '=' * 70)
    if failures:
        print('FAIL')
        for f in failures:
            print(f'  - {f}')
        return 1
    print('PASS: faulty requests failed cleanly and the app fully recovered.')
    print('=' * 70)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='cmd', required=True)

    s = sub.add_parser('serve', help='run the fake LM')
    s.add_argument('--port', type=int, default=9095)

    r = sub.add_parser('run', help='run the wedge test against a live app')
    r.add_argument('--app', default='http://localhost:9091')
    r.add_argument('--lm', default='http://localhost:9095')
    r.add_argument('--faults', type=int, default=30,
                   help='how many failing requests to fire (default 30)')
    r.add_argument('--timeout', type=float, default=30.0,
                   help='per-request client timeout; exceeding it counts as a hang')

    a = ap.parse_args()
    if a.cmd == 'serve':
        serve(a.port)
    else:
        sys.exit(asyncio.run(run(a.app, a.lm, a.faults, a.timeout)))


if __name__ == '__main__':
    main()
