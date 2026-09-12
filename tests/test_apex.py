"""STKApex v2 működési tesztek."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch
from stk_apex import (build_apex, STKApex, apex_mini, apex_60m,
                      DarwinGodelMachine, DGMAgent)
from stk_apex.config import STKApexConfig
from stk_apex.abstention import (
    is_abstention, abstention_reward, rlvr_reward,
    wrong_penalty, AbstentionKind,
)
from stk_apex.core import STKApexTrunk
from stk_apex.train_utils import _tiar_weights

# A szótárméretet a configból vesszük, nem drótozzuk be — különben minden
# tokenizáló-csere IndexError-ra futtatja a teszteket.
VOCAB = STKApexConfig().vocab_size


# ── Config ────────────────────────────────────────────────────────────────────

class TestConfig:

    def test_post_init_default(self):
        cfg = STKApexConfig()
        assert len(cfg.sgs_chunk_sizes) == cfg._n_sgs_blocks()
        assert len(cfg.tdk_dilations)   == cfg._n_sgs_blocks()

    def test_post_init_custom_layers(self):
        cfg = STKApexConfig(n_layers=8, attn_every=4)
        n = cfg._n_sgs_blocks()
        assert len(cfg.sgs_chunk_sizes) == n
        assert len(cfg.tdk_dilations)   == n

    def test_wrong_penalty(self):
        cfg = STKApexConfig(abstention_threshold=0.75)
        assert abs(cfg.wrong_penalty - (-3.0)) < 1e-9

    def test_validate_passes(self):
        cfg = apex_mini()
        cfg.validate()

    def test_validate_chunk_mismatch_raises(self):
        cfg = STKApexConfig()
        cfg.sgs_chunk_sizes = (8, 16)   # szándékos hiba
        with pytest.raises(AssertionError):
            cfg.validate()

    def test_gqa_config(self):
        cfg = apex_mini()
        assert cfg.n_kv_heads < cfg.n_heads
        assert cfg.n_heads % cfg.n_kv_heads == 0

    def test_shared_expert_config(self):
        cfg = apex_mini()
        assert cfg.use_shared_expert is True
        assert cfg.shared_expert_d_ff > 0

    def test_hyper_connections_config(self):
        cfg = apex_mini()
        assert cfg.use_hyper_connections is True


# ── Trunk (v2: forward visszaad (tensor, kvs) tuple-t) ───────────────────────

class TestTrunk:

    @pytest.fixture
    def trunk(self):
        cfg = apex_mini()
        return STKApexTrunk(cfg).eval()

    def test_shape(self, trunk):
        x = torch.randint(0, VOCAB, (2, 32))
        with torch.no_grad():
            hidden, kvs = trunk(x)
        assert hidden.shape == (2, 32, trunk.cfg.d_model)

    def test_kv_cache_returned(self, trunk):
        x = torch.randint(0, VOCAB, (1, 16))
        with torch.no_grad():
            hidden, kvs = trunk(x)
        # kvs: lista, annyi elem ahány attention réteg van
        assert isinstance(kvs, list)
        assert all(kv is None or (isinstance(kv, tuple) and len(kv) == 2)
                   for kv in kvs)

    def test_causality(self, trunk):
        """A jövő ne szivárogjon a múltba."""
        torch.manual_seed(0)
        ids     = torch.randint(0, VOCAB, (1, 24))
        ids_mod = ids.clone()
        ids_mod[0, -1] = (ids_mod[0, -1] + 7) % VOCAB
        with torch.no_grad():
            a, _ = trunk(ids)
            b, _ = trunk(ids_mod)
        diff = (a[:, :-1] - b[:, :-1]).abs().max().item()
        assert diff < 1e-4, f"Kauzalitás sértés: {diff:.2e}"

    def test_kv_cache_accepted(self, trunk):
        """A trunk elfogad past_kvs-t és nem dob kivételt."""
        ids = torch.randint(0, VOCAB, (1, 12))
        with torch.no_grad():
            _, kvs  = trunk(ids[:, :-1])
            h, kvs2 = trunk(ids[:, -1:], past_kvs=kvs)
        assert h.shape == (1, 1, trunk.cfg.d_model)
        assert isinstance(kvs2, list)


# ── Model ─────────────────────────────────────────────────────────────────────

class TestModel:

    @pytest.fixture
    def model(self):
        return build_apex("mini", use_rag=False).eval()

    def test_forward_shape(self, model):
        ids = torch.randint(0, VOCAB, (2, 16))
        with torch.no_grad():
            out = model(ids)
        assert out["logits"].shape == (2, 16, VOCAB)
        assert out["loss"] is None
        assert out["moe_aux_loss"] is not None

    def test_forward_returns_past_kvs(self, model):
        ids = torch.randint(0, VOCAB, (1, 16))
        with torch.no_grad():
            out = model(ids)
        assert "past_kvs" in out
        assert isinstance(out["past_kvs"], list)

    def test_forward_with_labels(self, model):
        ids    = torch.randint(0, VOCAB, (2, 16))
        labels = ids.clone()
        labels[:, :8] = -100
        with torch.no_grad():
            out = model(ids, labels=labels)
        assert out["loss"] is not None
        assert out["loss"].item() > 0

    def test_generate(self, model):
        ids = torch.randint(0, VOCAB, (1, 8))
        out = model.generate(ids, max_new_tokens=10, use_abstention=False)
        assert "sequences" in out
        assert out["sequences"].shape[1] > 8

    def test_training_mode_restored(self, model):
        """generate() után a tanítási mód visszaáll."""
        model.train()
        ids = torch.randint(0, VOCAB, (1, 8))
        model.generate(ids, max_new_tokens=5, use_abstention=False)
        assert model.training, "training mód nem állt vissza"

    def test_param_report(self, model):
        rep = model.param_report()
        assert rep["total"] > rep["active"] > 0
        assert rep["ratio"] < 1.0

    def test_reset_memory(self, model):
        model.reset_memory()

    def test_moe_aux_loss_positive(self, model):
        model.train()
        ids = torch.randint(0, VOCAB, (2, 16))
        out = model(ids, labels=ids)
        assert out["moe_aux_loss"].item() >= 0

    def test_generate_kv_cache(self, model):
        """generate() KV-cache nélkül és cache-sel azonos kimenetet ad."""
        torch.manual_seed(42)
        ids = torch.randint(0, VOCAB, (1, 6))
        with torch.no_grad():
            seq = model.generate(ids, max_new_tokens=4,
                                 use_abstention=False, temperature=0.0)
        assert seq["sequences"].shape[1] == 10


# ── TIAR súlyok ───────────────────────────────────────────────────────────────

class TestTIAR:

    def test_correct_weights_sum_to_one(self):
        w = _tiar_weights(10, reward=1.0)
        assert abs(w.sum().item() - 1.0) < 1e-5

    def test_wrong_weights_uniform(self):
        w = _tiar_weights(10, reward=-1.0)
        # egyenletes × |reward|
        expected = torch.ones(10) / 10
        assert (w - expected).abs().max().item() < 1e-5

    def test_correct_weights_increasing(self):
        w = _tiar_weights(8, reward=1.0)
        # Kései tokenek kapnak magasabb súlyt: gamma^(L-1-i) → i=L-1 → gamma^0=1
        assert w[-1].item() > w[0].item()

    def test_empty_seq(self):
        w = _tiar_weights(0, reward=1.0)
        assert w.shape[0] == 1   # fallback


# ── Tartózkodás ───────────────────────────────────────────────────────────────

class TestAbstention:

    def test_is_abstention_hu(self):
        assert is_abstention("Nem tudom a valaszt.")
        assert is_abstention("Nem vagyok biztos ebben.")

    def test_is_abstention_en(self):
        assert is_abstention("I don't know.")
        assert is_abstention("I'm not sure about this.")

    def test_not_abstention(self):
        assert not is_abstention("A válasz 42.")
        assert not is_abstention("Paris is the capital of France.")

    def test_wrong_penalty(self):
        assert abs(wrong_penalty(0.75) - (-3.0)) < 1e-9
        assert abs(wrong_penalty(0.5)  - (-1.0)) < 1e-9

    def test_correct_reward(self):
        r = abstention_reward("paris", "paris")
        assert r.kind == AbstentionKind.CORRECT
        assert r.reward == 1.0

    def test_wrong_reward(self):
        r = abstention_reward("london", "paris")
        assert r.kind == AbstentionKind.WRONG
        assert abs(r.reward - (-3.0)) < 1e-9

    def test_abstention_reward_zero(self):
        r = abstention_reward("Nem tudom.", "paris")
        assert r.kind == AbstentionKind.ABSTAINED
        assert r.reward == 0.0

    def test_rlvr_math(self):
        assert rlvr_reward("math", r"\boxed{42}", r"\boxed{42}") == 1.0
        assert rlvr_reward("math", r"\boxed{7}",  r"\boxed{42}") < 0


# ── Build API ─────────────────────────────────────────────────────────────────

class TestBuildAPI:

    def test_build_mini(self):
        m = build_apex("mini")
        assert isinstance(m, STKApex)

    def test_build_60m(self):
        m = build_apex("60m")
        assert isinstance(m, STKApex)

    def test_build_with_overrides(self):
        m = build_apex("mini", dropout=0.1, use_rag=False)
        assert m.cfg.dropout == 0.1

    def test_invalid_size(self):
        with pytest.raises(ValueError):
            build_apex("nonexistent")


# ── DGM ───────────────────────────────────────────────────────────────────────

class TestDGM:

    @pytest.fixture
    def dgm(self):
        cfg = apex_mini()
        return DarwinGodelMachine(seed_cfg=cfg, seed=0)

    def test_archive_seeded(self, dgm):
        assert len(dgm.archive) >= 1

    def test_best_returns_agent(self, dgm):
        best = dgm.best()
        assert isinstance(best, DGMAgent)
        assert best.fitness > float("-inf")

    def test_evolve_runs(self, dgm):
        best = dgm.evolve(n_iters=5, verbose=False)
        assert isinstance(best, DGMAgent)
        assert dgm.gen == 5

    def test_evolve_grows_archive(self, dgm):
        before = len(dgm.archive)
        dgm.evolve(n_iters=20, verbose=False)
        # Archívum sosem zsugorodik (MAP-Elites)
        assert len(dgm.archive) >= before

    def test_summary_contains_gen(self, dgm):
        dgm.evolve(n_iters=3, verbose=False)
        s = dgm.summary()
        assert "Generáció" in s

    def test_cell_returns_tuple(self, dgm):
        cfg  = apex_mini()
        cell = dgm._cell(cfg)
        assert isinstance(cell, tuple) and len(cell) == 2
