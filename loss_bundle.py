"""Composable, weighted aggregation of named loss terms.

A ``LossTerm`` packages a name, a scalar weight and a compute function.
``LossBundle`` is an ``nn.Module`` that registers a list of terms,
runs each on a shared context dict and returns ``(total, metrics)``:

    total   = sum(term.weight * term.fn(ctx)[0]   for active terms)
    metrics = {term.name: raw_value, "<extras>": ..., "total": total}

Inactivity protocol
-------------------
A term's compute function returns ``None`` to mean *"can't run this step"*
(e.g. CTC with no labels, JEPA with no masked frames). The bundle silently
skips inactive terms; ``total`` stays well-defined. This is how the same
bundle drives both supervised steps and self-supervised ``adapt`` steps
without any branching in the training loop.

Adding a new loss
-----------------
1. Write ``compute_my_loss(ctx) -> (Tensor, dict) | None`` that pulls what
   it needs out of ``ctx`` (keys are whatever the model's
   ``compute_loss`` puts there).
2. Append a ``LossTerm("my_loss", weight=X, fn=compute_my_loss)`` to the
   bundle. That's the entire diff -- the training loop and the metrics
   plumbing both pick it up automatically.

See ``make_v12_bundle`` for a complete example: a four-term bundle
equivalent to the historical ``V12Loss`` aggregator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

from loss import InfoNCELoss, SIGRegEppsPulleyLoss, SupConLoss


LossOutput = Optional[Tuple[torch.Tensor, Dict[str, float]]]
LossFn = Callable[[dict], LossOutput]


@dataclass
class LossTerm:
    """One named, weighted loss term."""

    name: str
    weight: float
    fn: LossFn


class LossBundle(nn.Module):
    """Weighted sum of ``LossTerm``s with per-term gating.

    The bundle behaves like a single loss module: ``bundle(**ctx)`` returns
    ``(total_scalar, metrics_dict)``. Terms with weight 0 or whose ``fn``
    returns ``None`` are skipped silently, keeping the aggregation valid
    even when some inputs aren't available.

    The bundle inherits from ``nn.Module`` so sub-losses with their own
    parameters (e.g. learnable temperatures) are registered correctly.
    Register parameter-carrying losses on the owning model and reference
    them from ``fn`` via closure -- LossBundle deliberately doesn't
    register the ``fn`` callables themselves.
    """

    def __init__(self, terms: Iterable[LossTerm]):
        super().__init__()
        self._terms: List[LossTerm] = list(terms)
        names = [t.name for t in self._terms]
        if len(names) != len(set(names)):
            dups = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(f"Duplicate LossTerm names: {dups}")

    # -- introspection --------------------------------------------------

    def __len__(self) -> int:
        return len(self._terms)

    def __iter__(self):
        return iter(self._terms)

    def names(self) -> List[str]:
        return [t.name for t in self._terms]

    # -- main entry point -----------------------------------------------

    def forward(self, **ctx) -> Tuple[torch.Tensor, Dict[str, float]]:
        metrics: Dict[str, float] = {}
        total: Optional[torch.Tensor] = None

        for term in self._terms:
            if term.weight == 0.0:
                continue
            out = term.fn(ctx)
            if out is None:
                continue
            value, extra = out
            metrics[term.name] = value.detach().item()
            for k, v in extra.items():
                metrics[k] = v.item() if torch.is_tensor(v) else float(v)
            contribution = term.weight * value
            total = contribution if total is None else total + contribution

        if total is None:
            total = torch.zeros((), device=_ctx_device(ctx))

        metrics["total"] = total.detach().item()
        return total, metrics

    # -- composition ----------------------------------------------------

    def subset(self, names: Iterable[str]) -> "LossBundle":
        """Return a new bundle containing only the named terms.

        Useful for self-supervised steps: ``bundle.subset({"jepa","sigreg"})``
        skips CTC without rewriting the wire-up. Term identity (weight,
        function, parameters) is preserved.
        """
        keep = set(names)
        unknown = keep - set(self.names())
        if unknown:
            raise ValueError(f"Unknown LossTerm name(s): {sorted(unknown)}")
        return LossBundle(t for t in self._terms if t.name in keep)


def _ctx_device(ctx: dict) -> torch.device:
    for v in ctx.values():
        if torch.is_tensor(v):
            return v.device
    return torch.device("cpu")


# =============================================================================
# Stock bundle: faithful port of ``V12Loss`` (CTC + InfoNCE + SIGReg + SupCon).
# =============================================================================


def make_v12_bundle(
    lambda_ctc: float = 1.0,
    lambda_jepa: float = 0.5,
    lambda_sigreg: float = 0.1,
    lambda_wc: float = 0.2,
    infonce_temp: float = 0.1,
    supcon_temp: float = 0.1,
    sigreg_projections: int = 256,
    sigreg_knots: int = 17,
) -> LossBundle:
    """Drop-in replacement for ``V12Loss`` as a ``LossBundle``.

    Behaviour is bit-for-bit equivalent: same gating predicates, same
    weighted-sum formula, same metric keys (``jepa``, ``jepa_acc``,
    ``sigreg``, ``wc``, ``wc_cov``, ``ctc``, ``total``).

    The contained sub-losses (``InfoNCELoss``, ``SIGRegEppsPulleyLoss``,
    ``SupConLoss``, ``nn.CTCLoss``) are held in module attributes via a
    private ``nn.Module`` carrier so their parameters / buffers are
    registered with the owning model.
    """
    carrier = _V12Carrier(
        infonce_temp=infonce_temp,
        supcon_temp=supcon_temp,
        sigreg_projections=sigreg_projections,
        sigreg_knots=sigreg_knots,
    )

    def _jepa(ctx):
        z_pred = ctx.get("z_pred")
        z_target = ctx.get("z_target")
        if z_pred is None or z_target is None or z_pred.shape[0] < 2:
            return None
        jepa, acc = carrier.infonce(z_pred, z_target)
        return jepa, {"jepa_acc": acc}

    def _sigreg(ctx):
        z_seq = ctx.get("z_seq")
        if z_seq is None:
            return None
        return carrier.sigreg(z_seq, valid_mask=ctx.get("valid_mask")), {}

    def _wc(ctx):
        line_vec = ctx.get("line_vec")
        writer_id = ctx.get("writer_id")
        if line_vec is None or writer_id is None:
            return None
        wc, cov = carrier.supcon(line_vec, writer_id)
        return wc, {"wc_cov": cov}

    def _ctc(ctx):
        ctc_logits = ctx.get("ctc_logits")
        targets = ctx.get("targets")
        if ctc_logits is None or targets is None:
            return None
        ctc = carrier.ctc(
            ctc_logits.permute(1, 0, 2),
            targets,
            ctx.get("input_lengths"),
            ctx.get("target_lengths"),
        )
        return ctc, {}

    bundle = LossBundle([
        LossTerm("jepa", lambda_jepa, _jepa),
        LossTerm("sigreg", lambda_sigreg, _sigreg),
        LossTerm("wc", lambda_wc, _wc),
        LossTerm("ctc", lambda_ctc, _ctc),
    ])
    bundle._carrier = carrier  # parameter / buffer ownership
    return bundle


class _V12Carrier(nn.Module):
    """Holds the stateful sub-loss modules for ``make_v12_bundle``."""

    def __init__(
        self,
        infonce_temp: float,
        supcon_temp: float,
        sigreg_projections: int,
        sigreg_knots: int,
    ):
        super().__init__()
        self.infonce = InfoNCELoss(temperature=infonce_temp)
        self.sigreg = SIGRegEppsPulleyLoss(
            num_projections=sigreg_projections, num_knots=sigreg_knots
        )
        self.supcon = SupConLoss(temperature=supcon_temp)
        self.ctc = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)


# =============================================================================
# v19 bundle : LeVJEPA transpose aux lignes manuscrites.
#   L = lambda_inv    * L_inv      (invariance MSE vue globale <- vues locales)
#     + lambda_sigreg * SIGReg     (Epps-Pulley sur les z projetes du batch)
#     + lambda_ctc    * CTC        (reconnaissance supervisee)
# =============================================================================


def make_v19_bundle(
    lambda_inv: float = 1.0,
    lambda_sigreg: float = 0.1,
    lambda_ctc: float = 1.0,
    sigreg_projections: int = 256,
    sigreg_knots: int = 17,
) -> LossBundle:
    """Bundle de pertes v19 (LeVJEPA, arXiv:2608.27395).

    ``L_inv = 1/(V+1) * somme_v ||z0 - zv||^2`` : MSE entre la projection
    h_phi([cls]) de la vue globale (z0) et celle de chaque vue locale
    (zv). Gradients dans les DEUX branches — pas de stop-gradient, pas de
    predicteur, pas d'EMA : le MSE a un minimum trivial (tous les [cls]
    egaux), c'est SIGReg qui l'empeche en poussant la distribution des
    [cls] du batch vers N(0, I) (cf. le papier LeWorldModel ; meme
    regularisateur que v12-v18). lambda_sigreg est le SEUL hyperparametre
    SSL.

    Cles du contexte :
      - ``z_global``  : (B, K_proj) projection de la vue globale.
      - ``z_locals``  : liste de V tensors (B, K_proj).
      - ``z_all``     : concat(en dim0) de z_global et des z_locals ->
        ((V+1)*B, K_proj), entree du SIGReg (le MEME espace projete que
        l'invariance).
      - ``ctc_logits`` / ``targets`` / ``input_lengths`` (en TOKENS) /
        ``target_lengths`` : chemin supervise, saute sans labels.
    """
    carrier = _V19Carrier(sigreg_projections, sigreg_knots)

    def _inv(ctx):
        z_global = ctx.get("z_global")
        z_locals = ctx.get("z_locals")
        if z_global is None or not z_locals:
            return None
        # Chaque vue locale est tiree vers la vue globale ; le terme MSE
        # laisse passer les gradients des deux cotes (symetrique).
        per_view = [(z_global - zv).pow(2).mean() for zv in z_locals]
        inv = sum(per_view) / (len(z_locals) + 1)
        return inv, {}

    def _sigreg(ctx):
        z_global = ctx.get("z_global")
        z_locals = ctx.get("z_locals")
        if z_global is None or not z_locals:
            return None
        # SIGReg doit porter sur le MEME espace projete que l'invariance
        # (z = h_phi([cls]) de toutes les vues concatenees) — c'est l'espace
        # ou l'invariance peut s'effondrer, et proj_head lui donne une
        # echelle plus favorable a l'Epps-Pulley. Fidele a LeVJEPA
        # (invariance + SIGReg co-localises en espace K). Corrige le
        # split-space (SIGReg sur [cls] bruts) qui rendait le regularisateur
        # inerte (plateau v17 ~0.70) pendant que z s'effondrait vers ~0.
        z_all = torch.cat([z_global] + z_locals, dim=0)  # (V+1)*B, K_proj
        if z_all.shape[0] < 2:
            return None
        return carrier.sigreg(z_all), {}

    def _ctc(ctx):
        ctc_logits = ctx.get("ctc_logits")
        targets = ctx.get("targets")
        if ctc_logits is None or targets is None:
            return None
        ctc = carrier.ctc(
            ctc_logits.permute(1, 0, 2),
            targets,
            ctx.get("input_lengths"),
            ctx.get("target_lengths"),
        )
        return ctc, {}

    bundle = LossBundle([
        LossTerm("inv", lambda_inv, _inv),
        LossTerm("sigreg", lambda_sigreg, _sigreg),
        LossTerm("ctc", lambda_ctc, _ctc),
    ])
    bundle._carrier = carrier  # possession des parametres / buffers
    return bundle


class _V19Carrier(nn.Module):
    """Modules de perte avec etat pour ``make_v19_bundle``."""

    def __init__(self, sigreg_projections: int, sigreg_knots: int):
        super().__init__()
        self.sigreg = SIGRegEppsPulleyLoss(
            num_projections=sigreg_projections, num_knots=sigreg_knots
        )
        self.ctc = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
