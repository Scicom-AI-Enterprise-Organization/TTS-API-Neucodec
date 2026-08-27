"""Build the listening page for bench/context_ab.py output.

Reads <out>/results.json and the wavs next to it, encodes each concatenated/oneshot
take to mp3 (ffmpeg, 24 kHz mono 96 kbps) as a data URI, precomputes waveform peaks and
the chunk-join times, and writes one self-contained HTML file (no external requests, so
it can be published as an artifact).

    python bench/context_ab_page.py --out audio/context_ab --html audio/context_ab/listen.html
"""

import argparse
import base64
import json
import os
import subprocess
import wave

import numpy as np

SR = 24000
PEAKS = 700   # bins per waveform strip


def read_pcm(path):
    with wave.open(path) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0


def peaks(y, n=PEAKS):
    if len(y) == 0:
        return []
    edges = np.linspace(0, len(y), n + 1).astype(int)
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        seg = y[a:b] if b > a else y[a:a + 1]
        out.append(round(float(np.abs(seg).max()), 3))
    return out


def mp3_data_uri(wav_path):
    r = subprocess.run(
        ['ffmpeg', '-v', 'error', '-y', '-i', wav_path, '-ac', '1', '-ar', str(SR),
         '-codec:a', 'libmp3lame', '-b:a', '96k', '-f', 'mp3', 'pipe:1'],
        check=True, capture_output=True,
    )
    return 'data:audio/mpeg;base64,' + base64.b64encode(r.stdout).decode('ascii')


COND_META = {
    'nocontext': ('A', 'No context', 'every chunk generated cold — today’s behaviour'),
    'context':   ('B', 'With request_id', 'each chunk prompted with the previous chunks’ text + speech tokens'),
    'oneshot':   ('C', 'One request', 'the whole text in a single call — the reference'),
}


def build(out_dir):
    with open(os.path.join(out_dir, 'results.json')) as f:
        doc = json.load(f)
    cases = []
    for res in doc['results']:
        d = os.path.join(out_dir, f"{res['case']}_take{res['take']}")
        conds = []
        for key in ('nocontext', 'context', 'oneshot'):
            c = res['conditions'][key]
            wav = os.path.join(d, f'{key}.wav')
            y = read_pcm(wav)
            # join times: cumulative chunk durations (oneshot has none)
            joins, t = [], 0.0
            if key != 'oneshot':
                for ch in c['chunks'][:-1]:
                    t += ch['audio_s']
                    joins.append(round(t, 3))
            code, label, blurb = COND_META[key]
            conds.append({
                'key': key, 'code': code, 'label': label, 'blurb': blurb,
                'src': mp3_data_uri(wav),
                'peaks': peaks(y),
                'duration': round(len(y) / SR, 2),
                'joins': joins,
                'chunks': c['chunks'],
            })
        cases.append({
            'name': res['case'], 'voice': res['voice'], 'take': res['take'],
            'texts': [ch['text'] for ch in res['conditions']['context']['chunks']],
            'conds': conds,
        })
    return {'generated': doc['generated'], 'url': doc['url'], 'cases': cases}


HTML = r'''<title>Chunk Join Listening Test</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700&family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;1,400&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{
  --bg:#EEF0F3; --surface:#FFFFFF; --surface-2:#E3E7EC; --line:#C9D0D8; --line-strong:#98A3AF;
  --ink:#1B2229; --ink-2:#4B5661; --ink-3:#7A8590;
  --a:#6F7E8C; --b:#C9921F; --c:#2E7F8A;           /* A slate, B ochre (the feature), C teal */
  --b-soft:rgba(201,146,31,.14); --join:#B4261F; --play:#1B2229;
  --focus:#C9921F;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --bg:#14181D; --surface:#1B2128; --surface-2:#232B33; --line:#2F3941; --line-strong:#4A5661;
    --ink:#E7EBEF; --ink-2:#AEB8C2; --ink-3:#7E8993;
    --a:#8E9CAA; --b:#E0B04A; --c:#4FB0BC;
    --b-soft:rgba(224,176,74,.14); --join:#F0645B; --play:#E7EBEF; --focus:#E0B04A;
  }
}
:root[data-theme="dark"]{
  --bg:#14181D; --surface:#1B2128; --surface-2:#232B33; --line:#2F3941; --line-strong:#4A5661;
  --ink:#E7EBEF; --ink-2:#AEB8C2; --ink-3:#7E8993;
  --a:#8E9CAA; --b:#E0B04A; --c:#4FB0BC;
  --b-soft:rgba(224,176,74,.14); --join:#F0645B; --play:#E7EBEF; --focus:#E0B04A;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 "IBM Plex Sans",system-ui,-apple-system,Segoe UI,sans-serif;-webkit-font-smoothing:antialiased}
.wrap{max-width:1040px;margin:0 auto;padding:40px 24px 80px}
h1,h2,h3{font-family:"Barlow Condensed","Arial Narrow",sans-serif;text-wrap:balance;margin:0}
h1{font-size:44px;line-height:1;font-weight:700;letter-spacing:-.01em}
.eyebrow{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:12px;letter-spacing:.12em;text-transform:uppercase;color:var(--ink-3)}
.lede{max-width:66ch;color:var(--ink-2);margin:14px 0 0}
.lede b{color:var(--ink);font-weight:600}
header{display:grid;gap:10px;padding-bottom:28px;border-bottom:1px solid var(--line)}
.legend{display:flex;flex-wrap:wrap;gap:10px 22px;margin-top:18px;font-size:14px;color:var(--ink-2)}
.legend span::before{content:"";display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:8px;vertical-align:-1px;background:var(--sw)}
.legend .j::before{width:2px;height:12px;border-radius:0;background:var(--join)}
.toolbar{display:flex;flex-wrap:wrap;align-items:center;gap:14px;margin:26px 0 8px;font-size:14px;color:var(--ink-2)}
.toolbar label{display:inline-flex;align-items:center;gap:8px;cursor:pointer}
.toolbar input{accent-color:var(--b);width:16px;height:16px}
.meta{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:12px;color:var(--ink-3);margin-left:auto}

.case{margin-top:36px;background:var(--surface);border:1px solid var(--line);border-radius:6px;overflow:hidden}
.case-head{display:grid;grid-template-columns:1fr auto;gap:8px 24px;padding:18px 22px 14px;border-bottom:1px solid var(--line);align-items:end}
.case-head h2{font-size:26px;font-weight:600;line-height:1.05}
.voice{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:13px;color:var(--ink-2);text-align:right}
.voice b{color:var(--ink);font-weight:500}
.chunks{grid-column:1/-1;display:flex;flex-wrap:wrap;gap:6px 0;font-size:15px;line-height:1.5;color:var(--ink-2);margin-top:4px}
.chunks span{padding:2px 8px 2px 0;border-right:2px solid var(--join);margin-right:8px}
.chunks span:last-child{border-right:0}

.cond{display:grid;grid-template-columns:150px 1fr;gap:0 18px;padding:14px 22px;border-top:1px solid var(--line);align-items:start}
.cond:first-of-type{border-top:0}
.cond.playing{background:var(--surface-2)}
.tag{display:flex;flex-direction:column;gap:3px;padding-top:2px}
.tag .code{font-family:"Barlow Condensed",sans-serif;font-weight:700;font-size:28px;line-height:1;color:var(--sw)}
.tag .lbl{font-weight:600;font-size:14px;line-height:1.2}
.tag .blurb{font-size:12.5px;line-height:1.35;color:var(--ink-3);max-width:150px}
.body{display:grid;gap:8px;min-width:0}
.wave{position:relative;height:84px;border-radius:4px;background:var(--surface-2);cursor:pointer;overflow:hidden}
.wave canvas{display:block;width:100%;height:100%}
.wave:focus-visible{outline:2px solid var(--focus);outline-offset:2px}
.row{display:flex;align-items:center;gap:14px;flex-wrap:wrap}
audio{height:34px;flex:1 1 320px;min-width:240px}
.stats{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:12px;color:var(--ink-3);font-variant-numeric:tabular-nums;display:flex;gap:16px;flex-wrap:wrap}
.stats b{color:var(--ink-2);font-weight:500}
details{font-size:13px}
summary{cursor:pointer;color:var(--ink-3);font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:12px;letter-spacing:.04em}
table{border-collapse:collapse;margin-top:8px;font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:12px;font-variant-numeric:tabular-nums;width:100%}
td,th{padding:4px 10px 4px 0;text-align:left;border-bottom:1px solid var(--line);vertical-align:top;color:var(--ink-2)}
th{color:var(--ink-3);font-weight:500;letter-spacing:.06em;text-transform:uppercase;font-size:11px}
td.n{text-align:right;padding-right:16px}
td.hi{color:var(--b);font-weight:500}
.tbl-wrap{overflow-x:auto}
.blind .tag .lbl,.blind .tag .blurb,.blind .stats .ctx,.blind details{visibility:hidden}
.blind .tag .code{color:var(--ink-2)}
.blind .wave canvas{filter:grayscale(1)}
footer{margin-top:40px;color:var(--ink-3);font-size:13px;max-width:70ch}
footer code{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:12px;background:var(--surface-2);padding:1px 5px;border-radius:3px}
@media (max-width:720px){
  h1{font-size:36px}
  .cond{grid-template-columns:1fr}
  .tag{flex-direction:row;align-items:baseline;gap:12px}
  .tag .blurb{max-width:none}
  .case-head{grid-template-columns:1fr}
  .voice{text-align:left}
}
@media (prefers-reduced-motion: reduce){ .wave canvas{transition:none} }
</style>

<div class="wrap">
<header>
  <div class="eyebrow">TTS-API-Neucodec · feat/speech-context · box 1024 (tm-h20, GPU 2, 2 workers)</div>
  <h1>Does chunk two continue chunk one?</h1>
  <p class="lede">A LiveKit agent cuts each reply into short chunks and sends every chunk as its own TTS request,
  played back to back. Each row below is the <b>same chunks, same voice, same text</b>, joined with no gap.
  Listen across the red join marks: does pitch, pace and energy carry over, or does every chunk restart?
  <b>A</b> is what ships today. <b>B</b> is the branch: the same request with a <code>request_id</code>, so each chunk is
  generated in the context of the ones before it. <b>C</b> is the whole text in one request — the ceiling.</p>
  <div class="legend">
    <span style="--sw:var(--a)">A · no context</span>
    <span style="--sw:var(--b)">B · with request_id</span>
    <span style="--sw:var(--c)">C · one request</span>
    <span class="j">chunk join (where a new request started)</span>
  </div>
</header>

<div class="toolbar">
  <label><input type="checkbox" id="blind"> Blind mode — hide labels and shuffle A/B/C within each case</label>
  <span class="meta" id="meta"></span>
</div>

<div id="cases"></div>

<footer>Generated by <code>bench/context_ab.py</code> against the branch running on the box (rule normalizer,
temperature 0.6 — the prod default, so takes differ run to run; that variance is part of what you are hearing).
<code>turns</code>/<code>tokens</code> under B are the <code>X-Context-Turns</code> / <code>X-Context-Tokens</code> response
headers: how many previous chunks, and how many of their speech tokens, went into that chunk’s prompt. The count
climbing while requests round-robin over two uvicorn workers is the shared store working. Waveforms are peak
envelopes of the mp3 you hear; the join marks sit at the exact sample where one chunk’s audio ends and the next
request’s begins.</footer>
</div>

<script id="data" type="application/json">__DATA__</script>
<script>
(function(){
  const data = JSON.parse(document.getElementById('data').textContent);
  const root = document.getElementById('cases');
  document.getElementById('meta').textContent = data.generated + ' · ' + data.cases.length + ' takes';
  const COLORS = {nocontext:'var(--a)', context:'var(--b)', oneshot:'var(--c)'};
  const pretty = n => n.replace(/_/g,' ').replace(/\btake(\d+)/,'take $1');

  let rafs = [];
  function css(v){ return getComputedStyle(document.documentElement).getPropertyValue(v).trim(); }

  function drawWave(cv, cond, playT){
    const dpr = window.devicePixelRatio || 1;
    const W = cv.clientWidth, H = cv.clientHeight;
    if (cv.width !== W*dpr || cv.height !== H*dpr){ cv.width = W*dpr; cv.height = H*dpr; }
    const g = cv.getContext('2d'); g.setTransform(dpr,0,0,dpr,0,0); g.clearRect(0,0,W,H);
    const n = cond.peaks.length, mid = H/2, bw = W/n;
    const col = css(COLORS[cond.key].slice(4,-1));
    const playX = playT != null ? (playT / cond.duration) * W : -1;
    for (let i=0;i<n;i++){
      const h = Math.max(1, cond.peaks[i] * (H*0.92));
      const x = i*bw;
      g.fillStyle = col; g.globalAlpha = (playX >= 0 && x <= playX) ? 1 : 0.55;
      g.fillRect(x, mid - h/2, Math.max(1, bw*0.8), h);
    }
    g.globalAlpha = 1;
    g.fillStyle = css('--join');
    for (const t of cond.joins){ const x = Math.round((t/cond.duration)*W); g.fillRect(x-1, 0, 2, H); }
    if (playX >= 0){ g.fillStyle = css('--play'); g.fillRect(Math.round(playX), 0, 1.5, H); }
  }

  function fmt(s){ return s.toFixed(2)+' s'; }

  for (const c of data.cases){
    const sec = document.createElement('section'); sec.className='case';
    sec.innerHTML = `<div class="case-head">
        <h2>${pretty(c.name)}</h2>
        <div class="voice">voice <b>${c.voice}</b> · ${c.texts.length} chunks</div>
        <div class="chunks">${c.texts.map(t=>`<span>${t.replace(/</g,'&lt;')}</span>`).join('')}</div>
      </div><div class="conds"></div>`;
    const conds = sec.querySelector('.conds');
    for (const cond of c.conds){
      const row = document.createElement('div'); row.className='cond'; row.dataset.key = cond.key;
      row.style.setProperty('--sw', COLORS[cond.key]);
      const ctxTurns = cond.key==='context' ? cond.chunks.map(ch=>ch.context_turns).join('→') : '';
      const ctxTok = cond.key==='context' ? cond.chunks.map(ch=>ch.context_tokens).join('→') : '';
      row.innerHTML = `<div class="tag"><span class="code">${cond.code}</span><span class="lbl">${cond.label}</span><span class="blurb">${cond.blurb}</span></div>
        <div class="body">
          <div class="wave" tabindex="0" role="slider" aria-label="${cond.label} waveform, click to seek"><canvas></canvas></div>
          <div class="row"><audio controls preload="metadata" src="${cond.src}"></audio>
            <div class="stats"><span><b>${fmt(cond.duration)}</b> total</span>
              ${cond.key!=='oneshot' ? `<span><b>${cond.joins.length}</b> joins</span>` : ''}
              ${cond.key==='context' ? `<span class="ctx">turns <b>${ctxTurns}</b></span><span class="ctx">tokens <b>${ctxTok}</b></span>` : ''}
            </div></div>
          ${cond.key!=='oneshot' ? `<details><summary>per-chunk detail</summary><div class="tbl-wrap"><table><tr><th>#</th><th>chunk</th><th class="n">audio</th><th class="n">latency</th>${cond.key==='context'?'<th class="n">turns in prompt</th><th class="n">tokens in prompt</th>':''}</tr>
            ${cond.chunks.map((ch,i)=>`<tr><td>${i+1}</td><td>${ch.text.replace(/</g,'&lt;')}</td><td class="n">${ch.audio_s.toFixed(2)} s</td><td class="n">${ch.latency_s.toFixed(2)} s</td>${cond.key==='context'?`<td class="n hi">${ch.context_turns}</td><td class="n hi">${ch.context_tokens}</td>`:''}</tr>`).join('')}
          </table></div></details>` : ''}
        </div>`;
      conds.appendChild(row);
      const cv = row.querySelector('canvas'), au = row.querySelector('audio'), wv = row.querySelector('.wave');
      const redraw = () => drawWave(cv, cond, au.paused && au.currentTime===0 ? null : au.currentTime);
      let raf;
      const tick = () => { redraw(); if(!au.paused) raf = requestAnimationFrame(tick); };
      au.addEventListener('play', () => { document.querySelectorAll('audio').forEach(o=>{ if(o!==au) o.pause(); }); row.classList.add('playing'); tick(); });
      au.addEventListener('pause', () => { row.classList.remove('playing'); cancelAnimationFrame(raf); redraw(); });
      au.addEventListener('ended', () => { row.classList.remove('playing'); cancelAnimationFrame(raf); redraw(); });
      au.addEventListener('seeked', redraw);
      const seek = x => { const r = wv.getBoundingClientRect(); au.currentTime = Math.max(0, Math.min(cond.duration-0.01, (x - r.left)/r.width*cond.duration)); if(au.paused) au.play(); };
      wv.addEventListener('click', e => seek(e.clientX));
      wv.addEventListener('keydown', e => { if(e.key==='ArrowRight'){au.currentTime=Math.min(cond.duration,au.currentTime+1);} if(e.key==='ArrowLeft'){au.currentTime=Math.max(0,au.currentTime-1);} if(e.key===' '||e.key==='Enter'){e.preventDefault(); au.paused?au.play():au.pause();} });
      rafs.push(redraw);
    }
    root.appendChild(sec);
  }
  const redrawAll = () => rafs.forEach(f=>f());
  requestAnimationFrame(redrawAll);
  window.addEventListener('resize', redrawAll);
  if (window.matchMedia){ window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', redrawAll); }
  new MutationObserver(redrawAll).observe(document.documentElement, {attributes:true, attributeFilter:['data-theme']});

  // blind mode: hide labels, shuffle rows inside each case (stable per toggle), reveal on untoggle
  const blind = document.getElementById('blind');
  blind.addEventListener('change', () => {
    document.body.classList.toggle('blind', blind.checked);
    document.querySelectorAll('.conds').forEach(cs => {
      const rows = [...cs.children];
      if (blind.checked){ for (let i=rows.length-1;i>0;i--){ const j=Math.floor(Math.random()*(i+1)); [rows[i],rows[j]]=[rows[j],rows[i]]; }
        rows.forEach((r,i)=>{ r.querySelector('.code').textContent = String(i+1); cs.appendChild(r); }); }
      else { const order = {nocontext:0, context:1, oneshot:2}; rows.sort((a,b)=>order[a.dataset.key]-order[b.dataset.key]);
        rows.forEach(r=>{ r.querySelector('.code').textContent = {nocontext:'A',context:'B',oneshot:'C'}[r.dataset.key]; cs.appendChild(r); }); }
    });
    redrawAll();
  });
})();
</script>
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='audio/context_ab')
    ap.add_argument('--html', default=None)
    args = ap.parse_args()
    data = build(args.out)
    html = HTML.replace('__DATA__', json.dumps(data, ensure_ascii=False).replace('</', '<\\/'))
    path = args.html or os.path.join(args.out, 'listen.html')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'wrote {path} ({os.path.getsize(path)/1e6:.1f} MB, {len(data["cases"])} takes)')


if __name__ == '__main__':
    main()
