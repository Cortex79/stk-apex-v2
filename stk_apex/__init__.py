"""
stk_apex v2 — STK-2 + Multi-Scale SGS + Fine-grained MoE + Shared Expert
            + GQA + KV-cache + RMSNorm + SwiGLU + HyperResidual
            + Calibrated Abstention (Abstain-R1) + TIAR RLVR + DGM
            + magyar-angol BPE tokenizáló (24 576, HU 1,96 tok/szó)

A modell NINCS betanítva — a súlyok véletlenszerűek. Ld. README.md.
"""
from .config      import STKApexConfig, apex_60m, apex_mini
from .core        import (STKApexTrunk, SGSBlock, RoPEAttnBlock,
                          MoEFeedForward, SwiGLU, RMSNorm, HyperResidual)
from .model       import STKApex, TemporalBrain, ReasonCompiler, RealityAnchor
from .abstention  import (abstention_reward, rlvr_reward, is_abstention,
                          wrong_penalty, verify,
                          AbstentionKind, AbstentionOutcome)
from .rag         import RAGRetriever
from .tokenizer   import STKTokenizer
from .multimodal  import (STKApexMultimodal, BytePatchEncoder, encode_stream,
                          VOCAB, TEXT_TOKEN, IMAGE_TOKEN, AUDIO_TOKEN)
from .evolution   import MAPElites, Individual
from .dgm         import DarwinGodelMachine, DGMAgent
from .train_utils import (combined_loss, tiar_rlvr_loss,
                          build_optimizer, cosine_schedule_with_warmup,
                          save_checkpoint, load_checkpoint)

__version__ = "2.0.0"
__all__ = [
    "STKApexConfig", "apex_60m", "apex_mini", "build_apex",
    "STKApex", "STKApexTrunk",
    "SGSBlock", "RoPEAttnBlock", "MoEFeedForward",
    "SwiGLU", "RMSNorm", "HyperResidual",
    "TemporalBrain", "ReasonCompiler", "RealityAnchor",
    "abstention_reward", "rlvr_reward", "is_abstention",
    "wrong_penalty", "verify", "AbstentionKind", "AbstentionOutcome",
    "RAGRetriever", "STKTokenizer",
    "STKApexMultimodal", "BytePatchEncoder", "encode_stream",
    "MAPElites", "Individual",
    "DarwinGodelMachine", "DGMAgent",
    "combined_loss", "tiar_rlvr_loss",
    "build_optimizer", "cosine_schedule_with_warmup",
    "save_checkpoint", "load_checkpoint",
]


def build_apex(size: str = "60m", **overrides) -> STKApex:
    """
    Gyári függvény.
      build_apex()              → 60M konfig, minden v2 fejlesztéssel
      build_apex("mini")        → ~15M aktív, gyors kísérletezés
      build_apex("60m", dropout=0.1, use_rag=False)
    """
    if size in ("60m", "full"):
        cfg = apex_60m()
    elif size == "mini":
        cfg = apex_mini()
    else:
        raise ValueError(f"Ismeretlen méret: '{size}'. Válasszon: '60m', 'mini'.")
    if overrides:
        from dataclasses import replace
        cfg = replace(cfg, **overrides)
        cfg.__post_init__()
    return STKApex(cfg)
