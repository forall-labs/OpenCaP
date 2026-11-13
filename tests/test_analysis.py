import math
import unittest

from opencap.analysis import (CONTRASTS, McNemar, contrast, contrast_trend,
                            host_effect, inference_flops, mcnemar,
                            pareto_front, scaling_trend, wilson)


class TestWilson(unittest.TestCase):
    def test_matches_published_value(self):
        # Gate 0's actual result: 28/30. Published Wilson 95% CI ~ [0.788, 0.982].
        iv = wilson(28, 30)
        self.assertAlmostEqual(iv.point, 28 / 30, places=6)
        self.assertAlmostEqual(iv.lo, 0.788, delta=0.003)
        self.assertAlmostEqual(iv.hi, 0.982, delta=0.003)

    def test_stays_inside_unit_interval_at_the_floor(self):
        # The Wald interval goes negative here; this is why Wilson is used --
        # the bottom of the ladder is expected to sit at or near zero.
        iv = wilson(0, 30)
        self.assertEqual(iv.point, 0.0)
        self.assertGreaterEqual(iv.lo, 0.0)
        self.assertLess(iv.hi, 0.2)

    def test_ceiling(self):
        iv = wilson(30, 30)
        self.assertLessEqual(iv.hi, 1.0)
        self.assertGreater(iv.lo, 0.8)

    def test_empty(self):
        self.assertEqual(wilson(0, 0).point, 0.0)


class TestMcNemar(unittest.TestCase):
    def test_exact_known_value(self):
        # (b, c) = (1, 9): exact two-sided p = 2 * (C(10,0)+C(10,1)) / 2^10
        a = {i: True for i in range(1)} | {i: False for i in range(1, 10)}
        b = {i: False for i in range(1)} | {i: True for i in range(1, 10)}
        r = mcnemar(a, b)
        self.assertEqual((r.b, r.c), (1, 9))
        self.assertAlmostEqual(r.p_value, 2 * 11 / 1024, places=6)

    def test_no_discordant_pairs_is_uninformative(self):
        a = {i: i % 2 == 0 for i in range(20)}
        r = mcnemar(a, dict(a))
        self.assertEqual(r.discordant, 0)
        self.assertEqual(r.p_value, 1.0)

    def test_only_shared_states_count(self):
        a = {1: True, 2: False, 99: True}
        b = {1: False, 2: False}
        r = mcnemar(a, b)
        self.assertEqual((r.b, r.c), (1, 0))

    def test_pairing_beats_unpaired_intervals(self):
        # The methodological claim in Sec. 05: a consistent shift that unpaired
        # intervals cannot resolve is easily significant when paired.
        n = 150
        a = {i: i < 100 for i in range(n)}
        b = {i: i < 108 for i in range(n)}
        iv_a, iv_b = wilson(100, n), wilson(108, n)
        self.assertLess(iv_a.hi, 1.0)
        self.assertTrue(iv_a.hi > iv_b.lo, "unpaired intervals should overlap")
        self.assertLess(mcnemar(a, b).p_value, 0.01, "paired test should resolve it")


class TestScalingTrend(unittest.TestCase):
    def _ladder(self, fn):
        params = {"m0.6": 0.6, "m1.7": 1.7, "m4": 4.0,
                  "m8": 8.0, "m14": 14.0, "m32": 32.0}
        per_model = {}
        for name, p in params.items():
            rate = fn(p)
            k = round(rate * 100)
            per_model[name] = {s: s < k for s in range(100)}
        return per_model, params

    def test_recovers_a_known_positive_slope(self):
        per_model, params = self._ladder(lambda p: 0.2 + 0.15 * math.log10(p))
        t = scaling_trend(per_model, params, n_boot=400)
        self.assertAlmostEqual(t.slope, 0.15, delta=0.02)
        self.assertTrue(t.significant)
        self.assertEqual(t.n_rungs, 6)

    def test_flat_ladder_spans_zero(self):
        per_model, params = self._ladder(lambda p: 0.5)
        t = scaling_trend(per_model, params, n_boot=400)
        self.assertAlmostEqual(t.slope, 0.0, delta=1e-9)
        self.assertFalse(t.significant)

    def test_needs_two_rungs(self):
        t = scaling_trend({"a": {0: True}}, {"a": 1.0}, n_boot=10)
        self.assertEqual(t.n_rungs, 1)


class TestContrasts(unittest.TestCase):
    def _cells(self, gap_fn):
        params = {"m1": 1.0, "m4": 4.0, "m16": 16.0, "m32": 32.0}
        cells = {}
        for name, p in params.items():
            lo_rate, gap = 0.3, gap_fn(p)
            cells[(name, "S4")] = {s: s < 30 for s in range(100)}
            cells[(name, "S1")] = {s: s < round((lo_rate + gap) * 100)
                                   for s in range(100)}
        return cells, params

    def test_contrast_value(self):
        cells, _ = self._cells(lambda p: 0.2)
        self.assertAlmostEqual(contrast(cells, "m4", "abstraction_gap"), 0.2, places=6)

    def test_missing_tier_returns_none(self):
        self.assertIsNone(contrast({}, "m4", "abstraction_gap"))

    def test_shrinking_gap_gives_negative_slope_supporting_h1(self):
        # Gap narrows as models grow => scaffolding substitutes for scale.
        cells, params = self._cells(lambda p: 0.4 - 0.15 * math.log10(p))
        t = contrast_trend(cells, params, "abstraction_gap", n_boot=400)
        self.assertLess(t.slope, 0)
        self.assertTrue(t.significant)

    def test_widening_gain_gives_positive_slope_supporting_h2(self):
        params = {"m1": 1.0, "m4": 4.0, "m16": 16.0, "m32": 32.0}
        cells = {}
        for name, p in params.items():
            cells[(name, "S3")] = {s: s < 30 for s in range(100)}
            cells[(name, "M1")] = {s: s < round((0.3 + 0.1 * math.log10(p)) * 100)
                                   for s in range(100)}
        t = contrast_trend(cells, params, "interaction_gain", n_boot=400)
        self.assertGreater(t.slope, 0)
        self.assertTrue(t.significant)

    def test_contrast_definitions_match_roadmap(self):
        self.assertEqual(CONTRASTS["abstraction_gap"][:2], ("S1", "S4"))
        self.assertEqual(CONTRASTS["interaction_gain"][:2], ("M1", "S3"))
        self.assertEqual(CONTRASTS["perception_subst"][:2], ("M3", "M2"))


class TestFrontier(unittest.TestCase):
    def test_pareto_keeps_only_improvements(self):
        pts = [(10.0, 0.2, "cheap-weak"), (20.0, 0.5, "mid"),
               (30.0, 0.4, "dominated"), (40.0, 0.7, "expensive-strong")]
        front = [p[2] for p in pareto_front(pts)]
        self.assertEqual(front, ["cheap-weak", "mid", "expensive-strong"])

    def test_ties_prefer_cheaper(self):
        front = pareto_front([(10.0, 0.5, "cheap"), (99.0, 0.5, "dear")])
        self.assertEqual([p[2] for p in front], ["cheap"])

    def test_flops_proxy_scales_linearly(self):
        self.assertAlmostEqual(inference_flops(8.0, 1000),
                               2 * inference_flops(4.0, 1000), places=3)


class TestHostEffect(unittest.TestCase):
    def _rows(self, host, successes, n=60):
        return [{"model": "qwen3-8b", "tier": "S1", "task": "cube_stack",
                 "state_id": i, "success": i < successes, "host_kind": host}
                for i in range(n)]

    def test_none_without_an_overlap_rung(self):
        # Host is perfectly correlated with size unless a rung runs on both.
        self.assertIsNone(host_effect(self._rows("colab-l4", 40), "qwen3-8b", "S1"))

    def test_identical_hosts_are_indistinguishable(self):
        rows = self._rows("colab-l4", 40) + self._rows("h100", 40)
        r = host_effect(rows, "qwen3-8b", "S1")
        self.assertEqual(r.discordant, 0)
        self.assertEqual(r.p_value, 1.0)

    def test_large_host_shift_is_detected(self):
        rows = self._rows("colab-l4", 20) + self._rows("h100", 50)
        r = host_effect(rows, "qwen3-8b", "S1")
        self.assertLess(r.p_value, 0.01)


if __name__ == "__main__":
    unittest.main(verbosity=2)
