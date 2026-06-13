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
from collections import defaultdict

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Character n-gram language model (stupid backoff)
# ---------------------------------------------------------------------------

class CharNgramLM:
    """
    Character-level n-gram LM with stupid backoff.

    Stored as a pickle with:
        counts: {order: {ngram_tuple: count}}
        total_chars: int
        vocab_size: int
        order: int
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

        # Precompute log counts for unigrams (most common lookup path)
        self._log_unigram = {}
        uni = self.counts.get(1, {})
        log_total = math.log(self.total_chars) if self.total_chars > 0 else 0.0
        for char, cnt in uni.items():
            self._log_unigram[char] = math.log(cnt) - log_total

        # Smoothing fallback for OOV unigrams
        self._log_oov = math.log(
            0.5 / (self.total_chars + 0.5 * self.vocab_size)
        )

        # Cache: (context_tuple, char) -> log_prob
        self._cache = {}
        self._cache_hits = 0
        self._cache_misses = 0

    @property
    def order(self):
        return self._order

    def log_prob(self, context, char):
        """
        Stupid-backoff log-probability of *char* given *context*.

        Args:
            context: tuple of character strings (history, most recent last).
                     Only the last (order-1) chars are used.
            char: candidate character string.
        Returns:
            Natural-log probability (unnormalised — fine for ranking).
        """
        # --- Cache check ---
        key = (context, char)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache_hits += 1
            return cached

        self._cache_misses += 1

        # Truncate context to order-1
        max_k = min(len(context), self._order - 1)

        result = self._compute_lp(context, char, max_k)

        # Store in cache (limit size to avoid memory blowup)
        if len(self._cache) < 2_000_000:
            self._cache[key] = result

        return result

    def _compute_lp(self, context, char, max_k):
        """Core stupid-backoff computation (no cache)."""
        for k in range(max_k, -1, -1):
            ctx = context[len(context) - k:] if k > 0 else ()
            order = k + 1

            if k == 0:
                # Unigram with smoothing
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
# Helpers (log-space arithmetic)
# ---------------------------------------------------------------------------

_NEG_INF = float("-inf")
_LOG0P5 = math.log(0.5)


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

    Args:
        logits: (B, T, C) tensor — CTC log-probabilities or raw logits.
        lengths: (B,) tensor — valid frame count per sample.
        idx_to_char: dict mapping CTC class index → character string.
        lm: optional CharNgramLM. If None, pure CTC beam search.
        beam_width: number of beams to keep after each frame.
        lm_weight: LM interpolation weight.
        prune_topk: only consider the top-k characters per frame.
        blank_id: CTC blank index (default 0).
        is_log_probs: if True, *logits* are already log-softmax'd.
        num_workers: if > 0, use that many processes (multiprocessing).
                     0 = sequential (default).

    Returns:
        list[str], one decoded string per batch element.
    """
    if logits.dim() != 3:
        raise ValueError(f"Expected (B, T, C) tensor, got {logits.shape}")

    if is_log_probs:
        log_probs_np = logits.float().numpy()
    else:
        log_probs_np = torch.log_softmax(logits.float(), dim=-1).numpy()

    # Precompute char strings for all indices (avoid dict lookups in hot loop)
    n_chars = log_probs_np.shape[-1]
    chars = [idx_to_char.get(i, "") for i in range(n_chars)]

    n_samples = log_probs_np.shape[0]

    # --- Multiprocessing path ---
    if num_workers > 1 and n_samples >= 4:
        return _decode_parallel(
            log_probs_np, lengths, chars, lm,
            beam_width, lm_weight, prune_topk, blank_id, num_workers,
        )

    # --- Sequential path ---
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
# Multiprocessing workers (module-level for Windows 'spawn')
# ---------------------------------------------------------------------------

# Globals set by _init_worker — one LM copy per process, not per task.
_w_lm = None
_w_chars = None
_w_params = None


def _init_worker(lm, chars, beam_width, lm_weight, prune_topk, blank_id):
    global _w_lm, _w_chars, _w_params
    _w_lm = lm
    _w_chars = chars
    _w_params = (beam_width, lm_weight, prune_topk, blank_id)


def _worker_single(args):
    """Decode one sample — runs in a worker process."""
    seq, length = args
    bw, lmw, ptk, bid = _w_params
    return _beam_search_single(
        seq[:length], _w_chars, _w_lm, bw, lmw, ptk, bid
    )


def _worker_decode(args):
    """Decode one sample — takes (seq_array, raw_text), returns decoded string."""
    seq, _text = args
    bw, lmw, ptk, bid = _w_params
    return _beam_search_single(
        seq, _w_chars, _w_lm, bw, lmw, ptk, bid
    )


def _decode_parallel(
    log_probs_np, lengths, chars, lm,
    beam_width, lm_weight, prune_topk, blank_id, num_workers,
):
    """Run beam search across samples using a process pool."""
    import multiprocessing as mp

    n_samples = log_probs_np.shape[0]
    n_procs = min(num_workers, n_samples)

    # Pack args: each worker gets (seq_array, length)
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


def _beam_search_single(
    log_probs, chars, lm, beam_width, lm_weight, prune_topk, blank_id
):
    """
    Beam search for a single sequence (optimized).

    log_probs: (T, C) numpy array of log-probabilities.
    chars: list[str] — precomputed index→char mapping.
    Returns decoded text string.
    """
    T, C = log_probs.shape
    lm_ctx_len = (lm.order - 1) if lm else 0
    _log_d = lm._log_discount if lm else 0.0

    # Each beam: prefix_tuple → (lp_blank, lp_non_blank)
    beams = {(): (0.0, _NEG_INF)}

    for t in range(T):
        new_beams = {}

        # Top-k candidates for this frame
        topk = min(prune_topk, C)
        top_indices = np.argpartition(log_probs[t], -topk)[-topk:]

        frame_lp = log_probs[t]  # (C,) — avoid repeated indexing

        # Fast path: if blank dominates (>0.95), just boost all blank probs
        # without exploring extensions — saves the full beam × topk loop.
        if frame_lp[blank_id] > -0.05:  # exp(-0.05) ≈ 0.95
            lp_c = float(frame_lp[blank_id])
            for prefix, (lp_b, lp_nb) in beams.items():
                merged = _beam_score((lp_b, lp_nb)) + lp_c
                cur = new_beams.get(prefix)
                if cur is None:
                    new_beams[prefix] = (merged, _NEG_INF)
                else:
                    new_beams[prefix] = (
                        merged if cur[0] == _NEG_INF
                        else merged + math.log1p(math.exp(cur[0] - merged))
                        if merged > cur[0]
                        else cur[0] + math.log1p(math.exp(merged - cur[0])),
                        cur[1],
                    )
            beams = new_beams
            continue

        if blank_id not in top_indices:
            top_indices = np.append(top_indices, blank_id)

        for prefix, (lp_b, lp_nb) in beams.items():
            # Precompute LM context for this prefix (truncated to order-1)
            if lm and prefix:
                # Only last (order-1) chars matter
                if len(prefix) <= lm_ctx_len:
                    lm_ctx = tuple(chars[i] for i in prefix)
                else:
                    lm_ctx = tuple(chars[i] for i in prefix[-lm_ctx_len:])
            elif lm:
                lm_ctx = ()
            else:
                lm_ctx = None

            for c in top_indices:
                lp_c = float(frame_lp[c])

                if c == blank_id:
                    # Blank: prefix unchanged
                    cur = new_beams.get(prefix)
                    if cur is None:
                        cur = (_NEG_INF, _NEG_INF)
                    merged = (lp_b if lp_b > lp_nb
                              else lp_nb + math.log1p(math.exp(lp_b - lp_nb))) + lp_c
                    new_b = (cur[0] if cur[0] > merged
                             else merged + math.log1p(math.exp(cur[0] - merged)))
                    new_beams[prefix] = (new_b, cur[1])

                elif len(prefix) > 0 and prefix[-1] == c:
                    # Same label repeat
                    # (a) Collapse: stays same prefix
                    cur = new_beams.get(prefix)
                    if cur is None:
                        cur = (_NEG_INF, _NEG_INF)
                    val = lp_nb + lp_c
                    new_nb = (cur[1] if cur[1] > val
                              else val + math.log1p(math.exp(cur[1] - val)))
                    new_beams[prefix] = (cur[0], new_nb)

                    # (b) Extend: blank-separated repeat
                    new_prefix = prefix + (c,)
                    score = lp_b + lp_c
                    if lm and lm_ctx is not None:
                        ch = chars[c]
                        score += lm_weight * lm.log_prob(lm_ctx, ch)
                    cur2 = new_beams.get(new_prefix)
                    if cur2 is None:
                        cur2 = (_NEG_INF, _NEG_INF)
                    new_nb2 = (cur2[1] if cur2[1] > score
                               else score + math.log1p(math.exp(cur2[1] - score)))
                    new_beams[new_prefix] = (cur2[0], new_nb2)

                else:
                    # Different label: extend
                    new_prefix = prefix + (c,)
                    if lp_b > lp_nb:
                        total = lp_b
                    elif lp_nb == _NEG_INF:
                        total = lp_b
                    else:
                        total = lp_nb + math.log1p(math.exp(lp_b - lp_nb))
                    score = total + lp_c
                    if lm and lm_ctx is not None:
                        ch = chars[c]
                        score += lm_weight * lm.log_prob(lm_ctx, ch)
                    cur = new_beams.get(new_prefix)
                    if cur is None:
                        cur = (_NEG_INF, _NEG_INF)
                    new_nb = (cur[1] if cur[1] > score
                              else score + math.log1p(math.exp(cur[1] - score)))
                    new_beams[new_prefix] = (cur[0], new_nb)

        # Prune: keep top beam_width beams by total log-prob
        scored = [
            (pfx, lb if lb > lnb
             else (lnb + math.log1p(math.exp(lb - lnb)) if lb != _NEG_INF else lnb))
            for pfx, (lb, lnb) in new_beams.items()
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        beams = {pfx: new_beams[pfx] for pfx, _ in scored[:beam_width]}

    # Select best beam
    best_prefix = max(beams, key=lambda p: _beam_score(beams[p]))

    return "".join(chars[c] for c in best_prefix)


def _beam_score(pair):
    """Total log-prob of a beam (lp_blank, lp_non_blank)."""
    lb, lnb = pair
    if lb == _NEG_INF:
        return lnb
    if lnb == _NEG_INF:
        return lb
    if lb > lnb:
        return lb + math.log1p(math.exp(lnb - lb))
    return lnb + math.log1p(math.exp(lb - lnb))
