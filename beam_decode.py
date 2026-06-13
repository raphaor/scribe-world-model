"""
CTC beam search decoder with character n-gram language model.

Self-contained: no external LM library (KenLM, pyctcdecode) needed.
The n-gram model uses stupid-backoff (Brants et al. 2007) which is
simple, fast, and well-suited for character-level LMs.

Usage in recognize.py / visualize.py:
    from beam_decode import CharNgramLM, ctc_beam_search_decode

    lm = CharNgramLM("char_8gram.pkl")
    decoded = ctc_beam_search_decode(
        ctc_logits.cpu(), input_lengths, idx_to_char, lm=lm,
        beam_width=20, lm_weight=0.3,
    )
"""

import math
import pickle

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Character n-gram language model (stupid backoff)
# ---------------------------------------------------------------------------

class CharNgramLM:
    """
    Character-level n-gram LM with stupid backoff.
    """

    def __init__(self, pkl_path, discount=0.4):
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        self.counts = {k: dict(v) for k, v in data["counts"].items()}
        self.total_chars = data["total_chars"]
        self.vocab_size = data["vocab_size"]
        self._order = data["order"]
        self.discount = discount
        self._log_discount = math.log(discount)

        # Precompute log unigram probs
        self._log_unigram = {}
        uni = self.counts.get(1, {})
        log_total = math.log(self.total_chars) if self.total_chars > 0 else 0.0
        for char, cnt in uni.items():
            self._log_unigram[char] = math.log(cnt) - log_total

        self._log_oov = math.log(
            0.5 / (self.total_chars + 0.5 * self.vocab_size)
        )

        # Cache
        self._cache = {}
        self._cache_hits = 0
        self._cache_misses = 0

    @property
    def order(self):
        return self._order

    def log_prob(self, context, char):
        """Stupid-backoff log-probability of char given context."""
        key = (context, char)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache_hits += 1
            return cached

        self._cache_misses += 1

        max_k = min(len(context), self._order - 1)
        result = self._compute_lp(context, char, max_k)

        if len(self._cache) < 2_000_000:
            self._cache[key] = result

        return result

    def _compute_lp(self, context, char, max_k):
        for k in range(max_k, -1, -1):
            ctx = context[len(context) - k:] if k > 0 else ()
            order = k + 1

            if k == 0:
                return self._log_unigram.get(
                    (char,), self._log_oov
                ) + max_k * self._log_discount

            ngram = ctx + (char,)
            order_counts = self.counts.get(order)
            if order_counts:
                c_ngram = order_counts.get(ngram, 0)
                if c_ngram > 0:
                    c_ctx = self.counts.get(k, {}).get(ctx, 0)
                    if c_ctx > 0:
                        backoffs = max_k - k
                        return (
                            math.log(c_ngram) - math.log(c_ctx)
                            + backoffs * self._log_discount
                        )

        return self._log_oov + max_k * self._log_discount

    def clear_cache(self):
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0


# ---------------------------------------------------------------------------
# Log-space helpers
# ---------------------------------------------------------------------------

_NEG_INF = float("-inf")


def _logsumexp(a, b):
    """Numerically stable log(exp(a) + exp(b))."""
    if a == _NEG_INF:
        return b
    if b == _NEG_INF:
        return a
    if a > b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


def _beam_score(lp_b, lp_nb):
    """Total log-prob of a beam (blank, non-blank)."""
    return _logsumexp(lp_b, lp_nb)


# ---------------------------------------------------------------------------
# CTC prefix beam search
# ---------------------------------------------------------------------------

def ctc_beam_search_decode(
    logits,
    lengths,
    idx_to_char,
    lm=None,
    beam_width=20,
    lm_weight=0.3,
    prune_topk=15,
    blank_id=0,
    is_log_probs=True,
    num_workers=0,
):
    """
    CTC prefix beam search with optional character n-gram LM.
    """
    if logits.dim() != 3:
        raise ValueError(f"Expected (B, T, C) tensor, got {logits.shape}")

    if is_log_probs:
        log_probs_np = logits.float().numpy()
    else:
        log_probs_np = torch.log_softmax(logits.float(), dim=-1).numpy()

    n_chars = log_probs_np.shape[-1]
    chars = [idx_to_char.get(i, "") for i in range(n_chars)]

    n_samples = log_probs_np.shape[0]

    if num_workers > 1 and n_samples >= 4:
        return _decode_parallel(
            log_probs_np, lengths, chars, lm,
            beam_width, lm_weight, prune_topk, blank_id, num_workers,
        )

    results = []
    for b in range(n_samples):
        L = int(lengths[b])
        seq = log_probs_np[b, :L, :]
        text = _beam_search_single(
            seq, chars, lm, beam_width, lm_weight, prune_topk, blank_id
        )
        results.append(text)

    return results


# ---------------------------------------------------------------------------
# Multiprocessing workers
# ---------------------------------------------------------------------------

_w_lm = None
_w_chars = None
_w_params = None


def _init_worker(lm, chars, beam_width, lm_weight, prune_topk, blank_id):
    global _w_lm, _w_chars, _w_params
    _w_lm = lm
    _w_chars = chars
    _w_params = (beam_width, lm_weight, prune_topk, blank_id)


def _worker_single(args):
    """Decode one sample — takes (seq_array, length)."""
    seq, length = args
    bw, lmw, ptk, bid = _w_params
    return _beam_search_single(
        seq[:length], _w_chars, _w_lm, bw, lmw, ptk, bid
    )


def _worker_decode(args):
    """Decode one sample — takes (seq_array, raw_text), returns string."""
    seq, _text = args
    bw, lmw, ptk, bid = _w_params
    return _beam_search_single(
        seq, _w_chars, _w_lm, bw, lmw, ptk, bid
    )


def _decode_parallel(
    log_probs_np, lengths, chars, lm,
    beam_width, lm_weight, prune_topk, blank_id, num_workers,
):
    import multiprocessing as mp

    n_samples = log_probs_np.shape[0]
    n_procs = min(num_workers, n_samples)

    tasks = [
        (log_probs_np[b], int(lengths[b]))
        for b in range(n_samples)
    ]

    chunksize = max(1, n_samples // (n_procs * 4))

    with mp.Pool(
        n_procs,
        initializer=_init_worker,
        initargs=(lm, chars, beam_width, lm_weight, prune_topk, blank_id),
    ) as pool:
        results = pool.map(_worker_single, tasks, chunksize=chunksize)

    return results


# ---------------------------------------------------------------------------
# Core beam search (correct CTC prefix beam search, Hannun et al. 2014)
# ---------------------------------------------------------------------------

def _beam_search_single(
    log_probs, chars, lm, beam_width, lm_weight, prune_topk, blank_id
):
    """
    CTC prefix beam search for a single sequence.

    Implements the algorithm from Hannun et al. (2014).
    Each beam tracks:
      - prefix: tuple of label indices (no blanks)
      - lp_blank: log-prob of best path ending in blank
      - lp_non_blank: log-prob of best path ending in non-blank
    """
    T, C = log_probs.shape
    lm_ctx_len = (lm.order - 1) if lm else 0

    beams = {(): (0.0, _NEG_INF)}  # prefix -> (lp_blank, lp_non_blank)

    for t in range(T):
        new_beams = {}

        # Top-k candidates for this frame
        topk = min(prune_topk, C)
        top_indices = np.argpartition(log_probs[t], -topk)[-topk:]
        if blank_id not in top_indices:
            top_indices = np.append(top_indices, blank_id)

        frame_lp = log_probs[t]

        # Fast path: blank-dominated frame
        if frame_lp[blank_id] > -0.05:
            lp_blank = float(frame_lp[blank_id])
            for prefix, (lp_b, lp_nb) in beams.items():
                total = _logsumexp(lp_b, lp_nb) + lp_blank
                cur = new_beams.get(prefix)
                if cur is None:
                    new_beams[prefix] = (total, _NEG_INF)
                else:
                    new_beams[prefix] = (_logsumexp(cur[0], total), cur[1])
            beams = new_beams
            continue

        for prefix, (lp_b, lp_nb) in beams.items():
            # LM context (truncated to order-1)
            if lm and prefix:
                ctx_slice = prefix if len(prefix) <= lm_ctx_len else prefix[-lm_ctx_len:]
                lm_ctx = tuple(chars[i] for i in ctx_slice)
            elif lm:
                lm_ctx = ()
            else:
                lm_ctx = None

            for c in top_indices:
                c = int(c)
                lp_c = float(frame_lp[c])

                if c == blank_id:
                    # Blank: prefix unchanged, accumulate to blank prob
                    total = _logsumexp(lp_b, lp_nb) + lp_c
                    cur = new_beams.get(prefix, (_NEG_INF, _NEG_INF))
                    new_beams[prefix] = (_logsumexp(cur[0], total), cur[1])

                elif len(prefix) > 0 and prefix[-1] == c:
                    # Same label as last in prefix

                    # (a) Collapse: repeat without intervening blank
                    #     Prefix unchanged, accumulate to non-blank
                    val = lp_nb + lp_c  # only non-blank-ending path can collapse
                    cur = new_beams.get(prefix, (_NEG_INF, _NEG_INF))
                    new_beams[prefix] = (cur[0], _logsumexp(cur[1], val))

                    # (b) Extend: blank-separated repeat -> new prefix
                    new_prefix = prefix + (c,)
                    score = lp_b + lp_c  # only blank-ending path can extend repeat
                    if lm and lm_ctx is not None:
                        score += lm_weight * lm.log_prob(lm_ctx, chars[c])
                    cur2 = new_beams.get(new_prefix, (_NEG_INF, _NEG_INF))
                    new_beams[new_prefix] = (cur2[0], _logsumexp(cur2[1], score))

                else:
                    # Different label: always extend prefix
                    new_prefix = prefix + (c,)
                    total = _logsumexp(lp_b, lp_nb) + lp_c
                    if lm and lm_ctx is not None:
                        total += lm_weight * lm.log_prob(lm_ctx, chars[c])
                    cur = new_beams.get(new_prefix, (_NEG_INF, _NEG_INF))
                    new_beams[new_prefix] = (cur[0], _logsumexp(cur[1], total))

        # Prune: keep top beam_width by total log-prob
        scored = sorted(
            new_beams.items(),
            key=lambda x: _beam_score(x[1][0], x[1][1]),
            reverse=True,
        )
        beams = dict(scored[:beam_width])

    # Select best beam
    best_prefix = max(beams, key=lambda p: _beam_score(beams[p][0], beams[p][1]))
    return "".join(chars[c] for c in best_prefix)
