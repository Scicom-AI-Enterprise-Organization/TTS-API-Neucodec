"""Stage 1: text -> speech tokens with vLLM (offline), one model per process.

Prompt shape is the serving app's single-turn TTS prompt (app/interleave.py
build_prompt with no history):  <|im_start|>{speaker}: {text}<|speech_start|>
"""
import argparse, json, os, re, time

ap = argparse.ArgumentParser()
ap.add_argument('--model', required=True)
ap.add_argument('--sentences', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--speaker', default='TM_English_Normal')
ap.add_argument('--temperature', type=float, default=0.6)
ap.add_argument('--top-p', type=float, default=0.95)
ap.add_argument('--repetition-penalty', type=float, default=1.15)
ap.add_argument('--max-tokens', type=int, default=3072)
ap.add_argument('--seed', type=int, default=1234)
ap.add_argument('--max-model-len', type=int, default=4096)
ap.add_argument('--gpu-mem', type=float, default=0.30)
a = ap.parse_args()

from vllm import LLM, SamplingParams

texts = [l.strip() for l in open(a.sentences, encoding='utf-8') if l.strip()]
prompts = [f'<|im_start|>{a.speaker}: {t}<|speech_start|>' for t in texts]

llm = LLM(model=a.model, dtype='bfloat16', max_model_len=a.max_model_len,
          gpu_memory_utilization=a.gpu_mem, max_num_seqs=16, seed=a.seed,
          tensor_parallel_size=1, disable_log_stats=True)
sp = SamplingParams(temperature=a.temperature, top_p=a.top_p,
                    repetition_penalty=a.repetition_penalty,
                    max_tokens=a.max_tokens,
                    stop_token_ids=[151643, 151645], seed=a.seed)
t0 = time.time()
outs = llm.generate(prompts, sp)
dt = time.time() - t0

recs = []
for i, (t, o) in enumerate(zip(texts, outs)):
    txt = o.outputs[0].text
    ids = [int(x) for x in re.findall(r'<\|s_(\d+)\|>', txt)]
    recs.append({
        'index': i, 'text': t, 'speaker': a.speaker,
        'tokens': ids, 'n_tokens': len(ids),
        'audio_s': round(len(ids) / 50.0, 3),
        'finish_reason': o.outputs[0].finish_reason,
        'n_gen_tokens': len(o.outputs[0].token_ids),
        'non_speech_chars': len(re.sub(r'<\|s_\d+\|>', '', txt)),
    })

meta = {'model': a.model, 'speaker': a.speaker, 'temperature': a.temperature,
        'top_p': a.top_p, 'repetition_penalty': a.repetition_penalty,
        'max_tokens': a.max_tokens, 'seed': a.seed, 'dtype': 'bfloat16',
        'vllm': __import__('vllm').__version__, 'generate_wall_s': round(dt, 2),
        'n': len(recs)}
os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
with open(a.out, 'w') as f:
    json.dump({'meta': meta, 'records': recs}, f)
bad = [r['index'] for r in recs if r['finish_reason'] != 'stop' or r['n_tokens'] < 10]
print('WROTE', a.out, meta)
print('suspect rows (finish!=stop or <10 tokens):', bad)
print('GEN_DONE')
