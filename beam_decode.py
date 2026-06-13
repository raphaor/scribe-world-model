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

    Scoring: P(w | context) uses stupid backoff with discount 0.4.
    Probabilities are NOT normalised (stupid backoff), but this is fine
    for ranking in beam search.
    """

    def __init__(self, pkl_path, discount=0.4):
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        self.counts = data["counts"]          # {1: {(c,): n, ...}, 2: {...}}
        self.total_chars = data["total_chars"]
        self.vocab_size = data["vocab_size"]
        self.order = data["order"]
        self.discount = discount
        self._log_discount = math.log(discount)

    def log_prob(self, context, char):
        """
        Stupid-backoff log-probability of *char* given *context*.

        Args:
            context: list of character strings (history, oldest first,
                     most recent last).
            char: candidate character string.
        Returns:
            Natural-log probability (unnormalised — fine for ranking).
        """
        max_k = min(len(context), self.order - 1)

        for k in range(max_k, -1, -1):
            ctx = tuple(context[-k:]) if k > 0 else ()
            ngram = ctx + (char,)
            order = k + 1

            c_ngram = self.counts.get(order, {}).get(ngram, 0)
            if c_ngram > 0:
                if k > 0:
                    c_ctx = self.counts.get(k, {}).get(ctx, 0)
                else:
                    c_ctx = self.total_chars
                if c_ctx > 0:
                    backoffs = max_k - k
                    return math.log(c_ngram / c_ctx) + backoffs * self._log_discount

            # k == 0 and unigram not found → smoothed fallback
            if k == 0:
                c_uni = self.counts.get(1, {}).get((char,), 0)
                backoffs = max_k
                return (
                    math.log((c_uni + 0.5) / (self.total_chars + 0.5 * self.vocab_size))
                    + backoffs * self._log_discount
                )

        return math.log(1.0 / self.vocab_size)  # unreachable safety net

    @property
    def order(self):
        return self._order

    @order.setter
    def order(self, v):
        self._order = v


# ---------------------------------------------------------------------------
# Helpers (log-space arithmetic)
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
):
    """
    CTC prefix beam search with optional character n-gram LM.

    Implements the algorithm from Hannun et al. (2014), "First-Pass
    Large Vocabulary Continuous Speech Recognition using Bi-Directional
    Recurrent DNNs." The LM score is added incrementally when a new
    character label is appended to a beam prefix.

    Args:
        logits: (B, T, C) tensor — CTC log-probabilities or raw logits.
        lengths: (B,) tensor — valid frame count per sample.
        idx_to_char: dict mapping CTC class index → character string.
        lm: optional CharNgramLM. If None, pure CTC beam search (no LM).
        beam_width: number of beams to keep after each frame.
        lm_weight: LM interpolation weight (0 = no LM influence).
        prune_topk: only consider the top-k characters per frame
                    (drastically reduces computation; 15 is enough for
                    a 107-class alphabet).
        blank_id: CTC blank index (default 0).
        is_log_probs: if True, *logits* are already log-softmax'd
                      (model CTC head output). If False, log_softmax
                      is applied internally.

    Returns:
        list[str], one decoded string per batch element.
    """
    import numpy as np

    if logits.dim() != 3:
        raise ValueError(f"Expected (B, T, C) tensor, got {logits.shape}")

    if is_log_probs:
        log_probs_np = logits.float().numpy()
    else:
        log_probs_np = torch.log_softmax(logits.float(), dim=-1).numpy()

    results = []
    for b in range(log_probs_np.shape[0]):
        L = int(lengths[b])
        seq = log_probs_np[b, :L, :]  # (L, C)
        text = _beam_search_single(
            seq, idx_to_char, lm, beam_width, lm_weight, prune_topk, blank_id
        )
        results.append(text)

    return results


def _beam_search_single(
    log_probs, idx_to_char, lm, beam_width, lm_weight, prune_topk, blank_id
):
    """
    Beam search for a single sequence.

    log_probs: (T, C) numpy array of log-probabilities.
    Returns decoded text string.
    """
    import numpy as np

    T, C = log_probs.shape

    # Each beam: (prefix_tuple, lp_blank, lp_non_blank)
    # prefix_tuple: tuple of CTC class indices (the label sequence)
    # lp_blank:     log-prob that the best path for this prefix ends with blank
    # lp_non_blank: log-prob that the best path ends with a non-blank label
    #
    # We use dicts keyed by prefix_tuple → (lp_blank, lp_non_blank).

    beams = {(): (0.0, _NEG_INF)}  # start: all mass in blank

    for t in range(T):
        new_beams = defaultdict(lambda: (_NEG_INF, _NEG_INF))

        # Pre-select top-k character candidates for this frame
        topk = min(prune_topk, C)
        # Always include blank in candidates
        top_indices = set(np.argpartition(log_probs[t], -topk)[-topk:].tolist())
        top_indices.add(blank_id)

        for prefix, (lp_b, lp_nb) in beams.items():
            for c in top_indices:
                lp_c = float(log_probs[t, c])

                if c == blank_id:
                    # Blank: prefix unchanged, accumulate to blank prob
                    cur_b, cur_nb = new_beams[prefix]
                    merged = _logsumexp(lp_b, lp_nb) + lp_c
                    new_beams[prefix] = (_logsumexp(cur_b, merged), cur_nb)

                elif len(prefix) > 0 and prefix[-1] == c:
                    # Same label as last in prefix — two sub-cases:

                    # (a) Collapse: repeat without intervening blank.
                    #     Prefix stays the same, accumulate to non-blank.
                    cur_b, cur_nb = new_beams[prefix]
                    new_beams[prefix] = (cur_b, _logsumexp(cur_nb, lp_nb + lp_c))

                    # (b) Extend: blank-separated repeat → new label.
                    new_prefix = prefix + (c,)
                    cur_b2, cur_nb2 = new_beams[new_prefix]
                    score = lp_b + lp_c
                    if lm:
                        ctx_chars = [idx_to_char.get(i, "") for i in prefix]
                        ch = idx_to_char.get(c, "")
                        score += lm_weight * lm.log_prob(ctx_chars, ch)
                    new_beams[new_prefix] = (cur_b2, _logsumexp(cur_nb2, score))

                else:
                    # Different label: always extend prefix.
                    new_prefix = prefix + (c,)
                    cur_b, cur_nb = new_beams[new_prefix]
                    score = _logsumexp(lp_b, lp_nb) + lp_c
                    if lm:
                        ctx_chars = [idx_to_char.get(i, "") for i in prefix]
                        ch = idx_to_char.get(c, "")
                        score += lm_weight * lm.log_prob(ctx_chars, ch)
                    new_beams[new_prefix] = (cur_b, _logsumexp(cur_nb, score))

        # Prune: keep top beam_width beams by total log-prob
        scored = [
            (pfx, _logsumexp(lb, lnb)) for pfx, (lb, lnb) in new_beams.items()
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        beams = {pfx: new_beams[pfx] for pfx, _ in scored[:beam_width]}

    # Select best beam
    best_prefix = max(
        beams, key=lambda p: _logsumexp(*beams[p])
    )

    return "".join(idx_to_char.get(c, "?") for c in best_prefix)
