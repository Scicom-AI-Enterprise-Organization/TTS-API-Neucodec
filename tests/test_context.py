"""
Unit tests for the cross-request speech context (app/context.py): the left trim, the
LM-window fit, the multi-turn prompt, and the memory/file stores (including the
cross-process file store under concurrent appends).

No GPU / torch / network required:

    uv run --with pytest -- pytest tests/test_context.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import threading
import time

import pytest

from app.context import (
    Turn,
    MIN_PARTIAL_TOKENS,
    LM_WINDOW_MARGIN,
    seconds_to_tokens,
    total_tokens,
    trim_turn_tail,
    trim_turns,
    select_voice,
    estimate_prompt_tokens,
    fit_context,
    build_prompt,
    tokens_to_str,
    MemoryContextStore,
    FileContextStore,
    make_store,
    default_store_dir,
    RequestContext,
)


def turn(n, text='one two three four five six seven eight nine ten', voice='husein', ts=0.0, base=0):
    return Turn(text=text, tokens=list(range(base, base + n)), voice=voice, ts=ts)


class TestTrim:
    def test_seconds_to_tokens(self):
        assert seconds_to_tokens(20) == 1000
        assert seconds_to_tokens(0.5) == 25
        assert seconds_to_tokens(-1) == 0

    def test_everything_fits(self):
        turns = [turn(100), turn(200), turn(300)]
        assert trim_turns(turns, 600) == turns
        assert trim_turns(turns, 10_000) == turns

    def test_drops_oldest_whole_turns(self):
        turns = [turn(100, ts=1), turn(200, ts=2), turn(300, ts=3)]
        kept = trim_turns(turns, 500)
        assert [t.ts for t in kept] == [2, 3]
        assert total_tokens(kept) == 500

    def test_oldest_kept_turn_is_left_cut_to_fill_budget(self):
        turns = [turn(600, ts=1, base=1000), turn(300, ts=2), turn(300, ts=3)]
        kept = trim_turns(turns, 1000)
        assert [t.ts for t in kept] == [1, 2, 3]
        assert total_tokens(kept) == 1000
        # the tail of the oldest turn, not its head
        assert kept[0].tokens == list(range(1000 + 200, 1000 + 600))

    def test_single_oversize_turn_keeps_its_tail(self):
        t = turn(3000, base=0)
        kept = trim_turns([t], 1000)
        assert len(kept) == 1
        assert kept[0].tokens == list(range(2000, 3000))

    def test_tiny_leftover_room_is_not_used(self):
        turns = [turn(600, ts=1), turn(300, ts=2), turn(300, ts=3)]
        kept = trim_turns(turns, 600 + MIN_PARTIAL_TOKENS - 1)
        assert [t.ts for t in kept] == [2, 3]

    def test_empty_turns_dropped_and_zero_budget(self):
        turns = [turn(0), turn(10), turn(0)]
        assert [len(t.tokens) for t in trim_turns(turns, 100)] == [10]
        assert trim_turns(turns, 0) == []
        assert trim_turns([], 100) == []

    def test_tail_text_is_cut_in_proportion_on_words(self):
        t = Turn(text='a b c d e f g h i j', tokens=list(range(100)), voice='v')
        cut = trim_turn_tail(t, 30)
        assert cut.tokens == list(range(70, 100))
        assert cut.text == 'h i j'
        assert cut.voice == 'v'

    def test_tail_text_cjk_by_character(self):
        t = Turn(text='我喜欢吃鸡饭和面条', tokens=list(range(90)), voice='v')
        cut = trim_turn_tail(t, 30)
        assert cut.text == '和面条'

    def test_tail_keeps_whole_turn_when_it_fits(self):
        t = turn(50)
        assert trim_turn_tail(t, 50) is t
        assert trim_turn_tail(t, 500) is t


class TestVoiceAndPrompt:
    def test_select_voice_trailing_run(self):
        turns = [turn(1, voice='idayu', ts=1), turn(1, voice='husein', ts=2),
                 turn(1, voice='idayu', ts=3), turn(1, voice='husein', ts=4),
                 turn(1, voice='husein', ts=5)]
        assert [t.ts for t in select_voice(turns, 'husein')] == [4, 5]
        assert select_voice(turns, 'idayu') == []
        assert select_voice([], 'husein') == []

    def test_plain_prompt_without_context(self):
        assert build_prompt([], 'husein', 'hello there.') == '<|im_start|>husein: hello there.<|speech_start|>'

    def test_multi_turn_prompt_format(self):
        prev = Turn(text='hello my name is husein,', tokens=[1, 2, 3], voice='husein')
        p = build_prompt([prev], 'husein', 'i like to eat chicken rice.')
        assert p == (
            '<|im_start|>husein: hello my name is husein,<|speech_start|>'
            '<|s_1|><|s_2|><|s_3|><|im_end|>'
            '<|im_start|>husein: i like to eat chicken rice.<|speech_start|>'
        )
        assert tokens_to_str([7, 42]) == '<|s_7|><|s_42|>'

    def test_turn_order_preserved(self):
        p = build_prompt([Turn('first', [1], 'v'), Turn('second', [2], 'v')], 'v', 'third')
        assert p.index('first') < p.index('second') < p.index('third')


class TestFit:
    MAX_MODEL_LEN = 4096

    def test_no_context_only_clamps_when_needed(self):
        turns, mt = fit_context([], 'hello', 3072, self.MAX_MODEL_LEN, 1000, 1000)
        assert turns == [] and mt == 3072
        turns, mt = fit_context([], 'x' * 3000, 3072, self.MAX_MODEL_LEN, 1000, 1000)
        assert mt == self.MAX_MODEL_LEN - estimate_prompt_tokens([], 'x' * 3000) - LM_WINDOW_MARGIN

    def test_prompt_plus_max_tokens_never_exceeds_window(self):
        history = [turn(400, text='w ' * 100, ts=i) for i in range(10)]
        for text_len in (10, 200, 2000):
            turns, mt = fit_context(history, 'y' * text_len, 3072, self.MAX_MODEL_LEN, 1000, 1000)
            assert estimate_prompt_tokens(turns, 'y' * text_len) + mt <= self.MAX_MODEL_LEN
            assert total_tokens(turns) <= 1000

    def test_context_gives_way_before_generation_floor(self):
        # window too small for both: context shrinks, generation floor survives
        history = [turn(400, ts=i) for i in range(10)]
        turns, mt = fit_context(history, 'hi', 3072, 1500, 1000, 1000)
        assert mt >= 1000
        assert estimate_prompt_tokens(turns, 'hi') + mt <= 1500
        assert total_tokens(turns) > 0            # some context still fits

    def test_generation_floor_is_min_of_requested(self):
        history = [turn(400, ts=i) for i in range(10)]
        turns, mt = fit_context(history, 'hi', 200, self.MAX_MODEL_LEN, 1000, 1000)
        assert mt == 200                            # a small request is never inflated
        assert total_tokens(turns) == 1000          # and full context fits alongside it

    def test_context_budget_respected(self):
        history = [turn(400, ts=i) for i in range(10)]
        turns, _ = fit_context(history, 'hi', 3072, self.MAX_MODEL_LEN, 250, 1000)
        assert total_tokens(turns) == 250


class TestMemoryStore:
    def test_append_get_trim(self):
        st = MemoryContextStore(ttl_s=60, max_tokens=500)
        assert st.get('a') == []
        st.append('a', turn(300, ts=1))
        st.append('a', turn(300, ts=2))
        got = st.get('a')
        assert [t.ts for t in got] == [1, 2]
        assert total_tokens(got) == 500            # trimmed on append
        assert st.get('b') == []

    def test_keys_independent_and_delete(self):
        st = MemoryContextStore(ttl_s=60, max_tokens=500)
        st.append('a', turn(10))
        st.append('b', turn(20))
        assert len(st.get('a')[0].tokens) == 10
        assert st.delete('a') is True
        assert st.delete('a') is False
        assert st.get('a') == [] and len(st.get('b')) == 1

    def test_ttl(self):
        st = MemoryContextStore(ttl_s=0.05, max_tokens=500)
        st.append('a', turn(10))
        assert len(st.get('a')) == 1
        time.sleep(0.08)
        assert st.get('a') == []

    def test_overlapping_requests_sorted_by_arrival(self):
        st = MemoryContextStore(ttl_s=60, max_tokens=500)
        st.append('a', Turn('second', [2], 'v', ts=2.0))   # finished first
        st.append('a', Turn('first', [1], 'v', ts=1.0))
        assert [t.text for t in st.get('a')] == ['first', 'second']


class TestFileStore:
    def test_shared_between_instances(self, tmp_path):
        # two instances = two worker processes on the same directory
        w0 = FileContextStore(str(tmp_path), ttl_s=60, max_tokens=500)
        w1 = FileContextStore(str(tmp_path), ttl_s=60, max_tokens=500)
        w0.append('room-1', Turn('hello my name is husein,', [1, 2, 3], 'husein', ts=1))
        got = w1.get('room-1')
        assert len(got) == 1 and got[0].text == 'hello my name is husein,' and got[0].tokens == [1, 2, 3]
        w1.append('room-1', Turn('i like to eat chicken rice.', [4, 5], 'husein', ts=2))
        assert [t.text for t in w0.get('room-1')] == ['hello my name is husein,', 'i like to eat chicken rice.']
        assert w0.get('room-2') == []

    def test_trim_on_append_and_delete(self, tmp_path):
        st = FileContextStore(str(tmp_path), ttl_s=60, max_tokens=500)
        for i in range(5):
            st.append('k', turn(300, ts=i))
        got = st.get('k')
        assert total_tokens(got) == 500 and got[-1].ts == 4
        assert st.delete('k') is True
        assert st.delete('k') is False
        assert st.get('k') == []

    def test_any_key_string_is_safe(self, tmp_path):
        st = FileContextStore(str(tmp_path), ttl_s=60, max_tokens=500)
        for key in ['../../etc/passwd', 'room/with/slashes', 'ünïcödé 🎤', 'x' * 1000]:
            st.append(key, turn(5))
            assert len(st.get(key)) == 1
        files = [f for f in os.listdir(tmp_path) if not f.startswith('.')]
        assert len(files) == 4 and all(f.endswith('.json') for f in files)

    def test_ttl_and_sweep(self, tmp_path):
        st = FileContextStore(str(tmp_path), ttl_s=0.05, max_tokens=500)
        st.append('old', turn(5))
        time.sleep(0.08)
        assert st.get('old') == []                  # expired on read
        st._last_sweep = 0.0
        st.append('new', turn(5))                   # triggers a sweep
        names = [f for f in os.listdir(tmp_path) if f.endswith('.json')]
        assert len(names) == 1                      # 'old' swept, 'new' kept
        assert len(st.get('new')) == 1

    def test_corrupt_file_reads_as_empty(self, tmp_path):
        st = FileContextStore(str(tmp_path), ttl_s=60, max_tokens=500)
        st.append('k', turn(5))
        with open(st._path('k'), 'w') as f:
            f.write('{not json')
        assert st.get('k') == []
        st.append('k', turn(6))                     # recovers
        assert len(st.get('k')[0].tokens) == 6

    def test_no_torn_reads_no_lost_updates_under_concurrency(self, tmp_path):
        """Many writers on one key (the overlapping-requests case): every append
        survives (the flock serializes read-modify-write) and readers never see a
        half-written file (atomic replace)."""
        N_WRITERS, N_EACH = 8, 25
        st = FileContextStore(str(tmp_path), ttl_s=60, max_tokens=10 ** 9)
        errors = []
        stop = threading.Event()

        def writer(w):
            own = FileContextStore(str(tmp_path), ttl_s=60, max_tokens=10 ** 9)
            for i in range(N_EACH):
                own.append('k', Turn(f'{w}-{i}', [w * 1000 + i], 'v', ts=w * 1000 + i))

        def reader():
            own = FileContextStore(str(tmp_path), ttl_s=60, max_tokens=10 ** 9)
            while not stop.is_set():
                try:
                    with open(own._path('k')) as f:
                        json.load(f)                # a torn file would raise here
                except FileNotFoundError:
                    pass
                except Exception as e:            # pragma: no cover
                    errors.append(e)

        threads = [threading.Thread(target=writer, args=(w,)) for w in range(N_WRITERS)]
        rd = threading.Thread(target=reader)
        rd.start()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        stop.set()
        rd.join()
        assert not errors
        got = st.get('k')
        assert len(got) == N_WRITERS * N_EACH
        assert [t.ts for t in got] == sorted(t.ts for t in got)
        assert not [f for f in os.listdir(tmp_path) if f.endswith('.tmp')]

    def test_lock_file_survives_sweep(self, tmp_path):
        st = FileContextStore(str(tmp_path), ttl_s=0.01, max_tokens=500)
        st.append('k', turn(5))
        time.sleep(0.03)
        st._last_sweep = 0.0
        st.append('k2', turn(5))
        assert os.path.exists(os.path.join(tmp_path, FileContextStore.LOCK_NAME))


class TestFactoryAndRequestContext:
    def test_make_store_kinds(self, tmp_path):
        assert make_store('off', '', 60, 100) is None
        assert make_store('none', '', 60, 100) is None
        assert isinstance(make_store('memory', '', 60, 100), MemoryContextStore)
        fs = make_store('file', str(tmp_path / 'ctx'), 60, 100)
        assert isinstance(fs, FileContextStore) and os.path.isdir(tmp_path / 'ctx')
        with pytest.raises(ValueError):
            make_store('redis', '', 60, 100)
        assert default_store_dir().endswith('tts-context')

    def test_request_context_commit_and_headers(self):
        st = MemoryContextStore(ttl_s=60, max_tokens=500)
        st.append('room', Turn('hello my name is husein,', [1, 2, 3], 'husein', ts=1))
        ctx = RequestContext(store=st, key='room', voice='husein', text='i like to eat chicken rice.',
                             turns=st.get('room'))
        assert ctx.tokens == 3
        assert ctx.headers() == {'X-Context-Id': 'room', 'X-Context-Turns': '1', 'X-Context-Tokens': '3'}
        turns = ctx.commit([4, 5, 6, 7])
        assert [t.text for t in turns] == ['hello my name is husein,', 'i like to eat chicken rice.']
        assert st.get('room')[-1].tokens == [4, 5, 6, 7]
        assert ctx.commit([]) is None               # nothing generated -> nothing stored

    def test_headers_are_latin1_safe(self):
        ctx = RequestContext(store=MemoryContextStore(60, 100), key='ruang 🎤', voice='v', text='t')
        h = ctx.headers()
        h['X-Context-Id'].encode('latin-1')
        assert h['X-Context-Id'] == 'ruang%20%F0%9F%8E%A4'

    def test_commit_never_raises(self):
        class Broken(MemoryContextStore):
            def append(self, key, turn):
                raise OSError('disk on fire')
        ctx = RequestContext(store=Broken(60, 100), key='k', voice='v', text='t')
        assert ctx.commit([1, 2]) is None


class TestContinueMode:
    def test_join_texts_drops_normalizer_stop_after_punctuation(self):
        from app.context import join_texts
        assert join_texts(['hello my name is husein,.', 'I like to eat chicken rice.']) == \
            'hello my name is husein, I like to eat chicken rice.'
        assert join_texts(['Is it you?.', 'Yes!.', 'Sure.']) == 'Is it you? Yes! Sure.'
        assert join_texts(['a.', '', '  ', 'b.']) == 'a. b.'
        assert join_texts(['version 2.0.']) == 'version 2.0.'        # a real decimal survives

    def test_continue_prompt_is_one_turn_with_prefix_tokens(self):
        prev = Turn(text='hello my name is husein,.', tokens=[1, 2, 3], voice='husein')
        p = build_prompt([prev], 'husein', 'I like to eat chicken rice.', mode='continue')
        assert p == ('<|im_start|>husein: hello my name is husein, I like to eat chicken rice.'
                     '<|speech_start|><|s_1|><|s_2|><|s_3|>')
        assert '<|im_end|>' not in p

    def test_continue_prompt_multiple_turns_in_order(self):
        turns = [Turn('one.', [1], 'v', ts=1), Turn('two.', [2, 3], 'v', ts=2)]
        p = build_prompt(turns, 'v', 'three.', mode='continue')
        assert p == '<|im_start|>v: one. two. three.<|speech_start|><|s_1|><|s_2|><|s_3|>'

    def test_no_turns_identical_in_both_modes(self):
        assert build_prompt([], 'v', 'hi.', mode='continue') == build_prompt([], 'v', 'hi.', mode='turns') \
            == '<|im_start|>v: hi.<|speech_start|>'

    def test_bad_mode_rejected(self):
        with pytest.raises(ValueError):
            build_prompt([], 'v', 'hi.', mode='sideways')

    def test_headers_carry_mode(self):
        ctx = RequestContext(store=MemoryContextStore(60, 100), key='k', voice='v', text='t', mode='continue')
        assert ctx.headers()['X-Context-Mode'] == 'continue'
        assert RequestContext(store=MemoryContextStore(60, 100), key='k', voice='v', text='t').headers()['X-Context-Mode'] == 'turns'
