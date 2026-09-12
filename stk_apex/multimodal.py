"""
STK-4 multimodális bájtszintű ingestion — STKApex-be integrálva.

Forrás: SUPERBRAIN-STK2-274M/architecture/stk4_multiscale.py

Architektúra: hierarchikus bájt-folt modell
    bájt stream → lokális encoder [B, P*K, mm_local_d]
                → folt-pooling [B, P, d_model]
                → STKApex trunk folytatja

VOCAB=261: 256 bájt + TEXT/IMAGE/AUDIO/SEP/PAD speciális tokenek.
Empirikusan: +0.35-0.78 BPB rosszabb mint tokenizátor azonos méretben.
Aktiválása csak képes/hangos adathoz ajánlott.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union

BYTE_VOCAB = 256
TEXT_TOKEN  = 256
IMAGE_TOKEN = 257
AUDIO_TOKEN = 258
SEP_TOKEN   = 259
PAD_TOKEN   = 260
VOCAB       = 261


def encode_stream(*parts: tuple) -> torch.Tensor:
    """
    Multimodális bájt-stream összeállítása.
    parts: [(modality_str, bytes_or_tensor), ...]
    modality: 'text', 'image', 'audio'

    Visszaad: 1D LongTensor
    """
    _modal = {"text": TEXT_TOKEN, "image": IMAGE_TOKEN, "audio": AUDIO_TOKEN}
    tokens = []
    for modality, data in parts:
        tokens.append(_modal.get(modality, TEXT_TOKEN))
        if isinstance(data, (bytes, bytearray)):
            tokens.extend(list(data))
        elif isinstance(data, torch.Tensor):
            tokens.extend(data.view(-1).tolist())
        else:
            raise TypeError(f"Ismeretlen adat-típus: {type(data)}")
        tokens.append(SEP_TOKEN)
    return torch.tensor(tokens, dtype=torch.long)


class BytePatchEncoder(nn.Module):
    """
    Hierarchikus bájt-encoder: lokális SGS foltok, majd patch-pooling.
    Kimenet: [B, n_patches, d_model] — ugyanaz mint a STKApexTrunk bemenete.
    """

    def __init__(self, d_model: int, local_d: int = 128,
                 patch_size: int = 8, dropout: float = 0.05):
        super().__init__()
        self.patch_size = patch_size
        self.byte_embed = nn.Embedding(VOCAB, local_d)

        # lokális konvolúciós encoder (O(P²) ahol P = folt mérete)
        self.local_conv = nn.Sequential(
            nn.Conv1d(local_d, local_d, kernel_size=3, padding=1, groups=local_d),
            nn.Conv1d(local_d, local_d, 1),
            nn.SiLU(),
            nn.LayerNorm([local_d]),  # note: applied after transpose
        )
        # folt-összesítés → d_model
        self.patch_proj = nn.Linear(local_d * patch_size, d_model)
        self.drop = nn.Dropout(dropout)

        nn.init.normal_(self.byte_embed.weight, std=0.02)
        nn.init.normal_(self.patch_proj.weight, std=0.02)
        nn.init.zeros_(self.patch_proj.bias)

    def forward(self, byte_ids: torch.Tensor) -> torch.Tensor:
        """
        byte_ids: [B, L] — L-et kerekíti patch_size többszörösére.
        Kimenet: [B, n_patches, d_model]
        """
        B, L = byte_ids.shape
        K = self.patch_size
        pad = (-L) % K
        if pad:
            byte_ids = F.pad(byte_ids, (0, pad), value=PAD_TOKEN)
        x = self.byte_embed(byte_ids)   # [B, L+pad, local_d]

        # lokális konvolúció
        h = x.transpose(1, 2)           # [B, local_d, L+pad]
        h = self.local_conv[0](h)
        h = self.local_conv[1](h)
        h = self.local_conv[2](h)       # SiLU
        h = h.transpose(1, 2)           # [B, L+pad, local_d]
        h = self.local_conv[3](h)       # LayerNorm

        # folt-összesítés
        n_patches = h.shape[1] // K
        h = h.view(B, n_patches, K * h.shape[-1])
        return self.drop(self.patch_proj(h))   # [B, n_patches, d_model]


class STKApexMultimodal(nn.Module):
    """
    Multimodális belépési réteg: bájt-stream → d_model patch-beágyazások.
    Egy STKApexTrunk.forward_embeds()-be injektálható.
    """

    def __init__(self, d_model: int, local_d: int = 128,
                 patch_size: int = 8, dropout: float = 0.05):
        super().__init__()
        self.encoder = BytePatchEncoder(d_model, local_d, patch_size, dropout)

    def forward(self, byte_ids: torch.Tensor) -> torch.Tensor:
        return self.encoder(byte_ids)
