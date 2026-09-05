"""Tiny async client for the normalizer's OpenAI-compatible LLM (OPENAI_* in .env): JSON-schema
constrained chat completions with the same fallback as app.llm_normalizer (retry without
response_format on 400), plus a file-backed cache so reruns are free."""
import asyncio
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from app.env import OPENAI_BASE_URL, OPENAI_API_KEY, OPENAI_MODEL_NAME, OPENAI_TIMEOUT  # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'multilingual_normalizer')
os.makedirs(RESULTS, exist_ok=True)


def _extract_json(text):
    text = re.sub(r'^```(?:json)?\s*|\s*```$', '', text.strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r'\{.*\}', text, re.S)
        return json.loads(m.group(0)) if m else None


async def chat_json(session, messages, schema_name, schema, sem, max_tokens=1500, temperature=0.7, retries=2):
    """Returns the parsed JSON object or None."""
    import aiohttp
    url = f'{OPENAI_BASE_URL.rstrip("/")}/chat/completions'
    headers = {'Authorization': f'Bearer {OPENAI_API_KEY}'}
    body = {'model': OPENAI_MODEL_NAME, 'messages': messages, 'temperature': temperature, 'max_tokens': max_tokens,
            'response_format': {'type': 'json_schema', 'json_schema': {'name': schema_name, 'strict': True, 'schema': schema}}}
    for attempt in range(retries + 1):
        try:
            async with sem:
                async with session.post(url, json=body, headers=headers, timeout=aiohttp.ClientTimeout(total=OPENAI_TIMEOUT or 120)) as r:
                    if r.status == 400 and 'response_format' in body:
                        body.pop('response_format')
                        continue
                    data = await r.json()
            content = data['choices'][0]['message']['content']
            obj = _extract_json(content)
            if obj is not None:
                return obj
        except Exception as e:                       # noqa: BLE001 - network / parse; retry
            if attempt == retries:
                print(f'   llm error: {type(e).__name__}: {e}', file=sys.stderr)
            await asyncio.sleep(1.5 * (attempt + 1))
    return None


class JsonlCache:
    """Append-only jsonl keyed by a caller-chosen key."""

    def __init__(self, path):
        self.path = path
        self.rows = {}
        if os.path.exists(path):
            for line in open(path):
                if line.strip():
                    r = json.loads(line)
                    self.rows[r['_key']] = r

    def __contains__(self, key):
        return key in self.rows

    def get(self, key):
        return self.rows.get(key)

    def add(self, key, row):
        row = dict(row, _key=key)
        self.rows[key] = row
        with open(self.path, 'a') as f:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')

    def values(self):
        return list(self.rows.values())


SCRIPT_RE = {'ar': re.compile(r'[؀-ۿ]'), 'zh': re.compile(r'[一-鿿]'), 'ta': re.compile(r'[஀-௿]'),
             'ta-LK': re.compile(r'[஀-௿]'), 'si': re.compile(r'[඀-෿]')}
LATIN_RE = re.compile(r'[A-Za-zÀ-ɏ]')


def right_script(text, lang):
    """The text is (mostly) in the locale's script."""
    if lang in SCRIPT_RE:
        return len(SCRIPT_RE[lang].findall(text)) >= 3
    other = sum(len(r.findall(text)) for r in SCRIPT_RE.values())
    return len(LATIN_RE.findall(text)) >= 5 and other == 0
