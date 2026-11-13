"""Minimal-repair verification and hand tagging for S3/S4.

None of these launch a GPU run: run_draw is replaced with a stub in every test
that would reach it (RESUME.md trap 4 -- a test once did, and its child
outlived the subprocess timeout and competed with a real verification).
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"_{name}", ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _draw(fails: set[int], n: int = 30) -> dict:
    out = {s: s not in fails for s in range(1, n + 1)}
    return {"label": "stub", "score": n - len(fails), "outcomes": out,
            "fails": sorted(fails), "crashed": []}


class TestDrawsStopOnlyWhenTheVerdictCannotChange(unittest.TestCase):

    def setUp(self):
        self.m = _load("verify_reduced_repair")
        self.tmp = Path(tempfile.mkdtemp())
        self.m.ORACLE = self.tmp / "oracle"     # never pick up a real run
        self.code = self.tmp / "fix.py"
        self.code.write_text("pass\n")

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def _run(self, seq, task="cube_lifting", k_max=5):
        it = iter(seq)
        self.m.run_draw = lambda *a, **k: next(it)
        return self.m.verify(task, "S3", "pid", self.code, 30, k_max)

    def test_accepts_as_soon_as_every_scene_has_passed_somewhere(self):
        r = self._run([_draw({29, 4}), _draw({7})])
        self.assertTrue(r["accepted"])
        self.assertEqual(r["k_used"], 2)
        self.assertEqual(r["stopped"], "accepted")

    def test_a_scene_failing_every_draw_is_rejected_at_k_max(self):
        r = self._run([_draw({29, i}) for i in range(1, 6)])
        self.assertFalse(r["accepted"])
        self.assertEqual(r["always_fail"], [29])
        self.assertEqual(r["k_used"], 5)
        self.assertEqual(r["stopped"], "k_max")

    def test_a_low_rate_is_flagged_not_rejected_early(self):
        # The bar is about scenes. A first version stopped after 3 draws on a
        # low pooled RATE and rejected a repair the bar would have accepted.
        r = self._run([_draw(set(range(1, 16))), _draw(set(range(16, 31)))])
        self.assertTrue(r["accepted"], "every scene passes once: the bar accepts")
        self.assertTrue(r["below_reference"], "15/30 vs ~87% is still flagged")
        bad = set(range(1, 26))
        r = self._run([_draw(bad)] * 5)
        self.assertEqual(r["k_used"], 5, "no early stop on rate")
        self.assertEqual(r["stopped"], "k_max")

    def test_the_result_is_json_serialisable(self):
        # below_reference came back as numpy.bool_ and json.dumps raised AFTER
        # the draws had run, so a whole verification was lost per queue item.
        import json
        r = self._run([_draw(set(range(1, 16))), _draw(set(range(16, 31)))])
        json.dumps(r)
        self.assertIs(type(r["below_reference"]), bool)

    def test_a_completed_draw_on_disk_is_reused(self):
        label = "min_x_S3_pid_abc_d1"
        rd = self.m.ORACLE / label
        rd.mkdir(parents=True)
        (rd / "aaa_done_flag").write_text("")
        for s in range(1, 31):
            ok = 0 if s == 7 else 1
            (rd / f"trial_{s:02d}_sandboxrc_0_reward_1.000_taskcompleted_{ok}").mkdir()
        d = self.m.reuse_draw(label, 30)
        self.assertEqual(d["score"], 29)
        self.assertEqual(d["fails"], [7])
        self.assertIsNone(self.m.reuse_draw("min_x_S3_pid_abc_d2", 30))
        (rd / "aaa_done_flag").unlink()
        self.assertIsNone(self.m.reuse_draw(label, 30), "an unfinished run is not reused")

    def test_no_early_rejection_without_a_reference(self):
        bad = set(range(1, 26))
        r = self._run([_draw(bad)] * 8, task="nut_assembly", k_max=8)
        self.assertEqual(r["k_used"], 8)

    def test_a_draw_that_did_not_run_is_not_counted_as_a_draw(self):
        self.m.run_draw = lambda *a, **k: {"label": "x", "score": None, "error": "boom"}
        r = self.m.verify("cube_lifting", "S3", "pid", self.code, 30, 5)
        self.assertFalse(r["ok"])

    def test_reference_rejection_chance(self):
        self.assertEqual(self.m.p_reference_rejected({1: 0.0, 2: 0.0}, 3), 0.0)
        self.assertAlmostEqual(self.m.p_reference_rejected({1: 0.5}, 3), 0.125)
        self.assertAlmostEqual(self.m.p_reference_rejected({1: 0.5, 2: 0.5}, 1), 0.75)
        self.assertLess(self.m.p_reference_rejected({1: 0.7}, 12), self.m.p_reference_rejected({1: 0.7}, 8))

    def test_reference_fail_rates_read_completed_draws_only(self):
        for i, fails in ((1, {3}), (2, {3, 4})):
            rd = self.m.ORACLE / f"ref_x_d{i}"
            rd.mkdir(parents=True)
            (rd / "aaa_done_flag").write_text("")
            for s in range(1, 31):
                (rd / f"trial_{s:02d}_sandboxrc_0_reward_1.000_taskcompleted_{0 if s in fails else 1}").mkdir()
        (self.m.ORACLE / "ref_x_d3").mkdir()           # unfinished: ignored
        n, fr = self.m.reference_fail_rates("ref_x")
        self.assertEqual(n, 2)
        self.assertEqual((fr[3], fr[4], fr[5]), (1.0, 0.5, 0.0))

    def test_cube_stack_k_follows_the_programs_perception_route(self):
        kit = "def _red_only(m):\n    segment_sam3_text_prompt(x, 'green cube')\n# ---- end repair kit ----\n"
        self.assertEqual(self.m.k_for("cube_stack", kit + "m = segment_sam3_text_prompt(rgb, 'red cube')\n"), 16)
        self.assertEqual(self.m.k_for("cube_stack", kit + "p = point_prompt_molmo(rgb, 'red cube')\n"
                                      "m = segment_sam3_point_prompt(rgb, p['red cube'])\n"), 12,
                         "the kit's own text prompt must not make a point-prompt program text-route")
        self.assertEqual(self.m.k_for("cube_stack", kit + "p = point_prompt_molmo(rgb, 'red cube')\n"
                                      "m = segment_sam3_text_prompt(rgb, 'red cube')\n"), 16,
                         "molmo used only as an existence check: the masks are still text-prompted")
        self.assertEqual(self.m.k_for("cube_lifting", "segment_sam3_text_prompt(rgb, 'x')\n"), 5)

    def test_k_max_is_measured_not_the_homogeneous_default(self):
        self.assertEqual(self.m.K_MAX["spill_wipe"], 1)
        self.assertEqual(self.m.K_MAX["cube_lifting"], 5)
        self.assertEqual(self.m.K_MAX["cube_stack"], 12)


class TestAcceptedRepairsAreFiledWithTheirTier(unittest.TestCase):

    def setUp(self):
        self.m = _load("verify_reduced_repair")
        self.tmp = Path(tempfile.mkdtemp())
        (self.tmp / "scripts").mkdir()
        shutil.copy(ROOT / "scripts" / "save_repair.py", self.tmp / "scripts")
        self.r = {"k_used": 2, "k_max": 5, "scores": [27, 28], "sha": "abc"}

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_minimal_repair_lands_in_minimal_with_tier(self):
        self.m.save_accepted("cube_lifting", "S4", "p1", "x = 1\n", self.r, False, "",
                             root=self.tmp)
        d = json.loads((self.tmp / "results/minimal/cube_lifting.json").read_text())
        self.assertEqual(d["p1"]["tier"], "S4")
        self.assertEqual(d["p1"]["code"], "x = 1\n")
        self.assertIn("MINIMAL", d["p1"]["note"])

    def test_a_passing_original_is_stored_as_no_defect(self):
        self.m.save_accepted("cube_lifting", "S3", "p2", "orig\n", self.r, True, "",
                             root=self.tmp)
        d = json.loads((self.tmp / "results/minimal/cube_lifting.json").read_text())
        self.assertIsNone(d["p2"]["code"])
        self.assertEqual(d["p2"]["level"], "L4_scene_limited")

    def test_attempts_are_appended_not_overwritten(self):
        log = self.tmp / "v.json"
        self.m.log_attempt("p", {"sha": "1"}, log)
        self.m.log_attempt("p", {"sha": "2"}, log)
        self.assertEqual([e["sha"] for e in json.loads(log.read_text())["p"]], ["1", "2"])


class TestHandTagsStayInTheS1Taxonomy(unittest.TestCase):

    def setUp(self):
        self.m = _load("tag_reduced")
        self.tmp = Path(tempfile.mkdtemp())
        (self.tmp / "results" / "minimal").mkdir(parents=True)
        (self.tmp / "scripts").mkdir()
        shutil.copy(ROOT / "scripts" / "spill_wipe_batch.py", self.tmp / "scripts")
        orig = "a = 1\nb = 2\nmove_to_joints(a)\n"
        (self.tmp / "results/failures_s3.json").write_text(json.dumps([
            {"pid": "p1", "task": "cube_lifting", "models": ["m"], "n_episodes": 30,
             "code": orig},
            {"pid": "p2", "task": "cube_lifting", "models": ["m"], "n_episodes": 30,
             "code": "x = (\n"}]))
        (self.tmp / "results/minimal/cube_lifting.json").write_text(json.dumps(
            {"p1": {"code": "a = 1\nb = 3\nmove_to_joints(a)\n", "tier": "S3"}}))

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_tags_outside_the_taxonomy_are_refused(self):
        with self.assertRaises(SystemExit):
            self.m.build_record("S3", "p1", {"missing_world_transform": "x"}, root=self.tmp)

    def test_a_tag_record_carries_family_evidence_and_distance(self):
        r = self.m.build_record("S3", "p1", {"wrong_formula": "b off by one"}, root=self.tmp)
        self.assertEqual(r["families"], ["SEMANTIC"])
        self.assertEqual(r["evidence"]["wrong_formula"], "b off by one")
        self.assertGreater(r["distance"], 0)
        self.assertEqual(r["tagged_by"], "hand")

    def test_no_repair_means_no_tags_except_syntax(self):
        with self.assertRaises(SystemExit):
            self.m.build_record("S3", "p2", {"wrong_formula": "x"}, root=self.tmp)
        r = self.m.build_record("S3", "p2", {"syntax": "ends mid-call"}, root=self.tmp)
        self.assertTrue(r["truncated"])
        self.assertIsNone(r["distance"])

    def test_substitutions_are_recorded_but_are_not_tags(self):
        r = self.m.build_record("S3", "p1", {"wrong_formula": "x"}, root=self.tmp,
                                substitutions={"grasp": "plan_grasp cannot grasp this"})
        self.assertEqual(r["substitutions"], {"grasp": "plan_grasp cannot grasp this"})
        self.assertEqual(r["tags"], ["wrong_formula"])

    def test_a_repair_filed_for_another_tier_is_refused(self):
        (self.tmp / "results/minimal/cube_lifting.json").write_text(json.dumps(
            {"p1": {"code": "y\n", "tier": "S1"}}))
        with self.assertRaises(SystemExit):
            self.m.build_record("S3", "p1", {"wrong_formula": "x"}, root=self.tmp)


class TestCrossTierShares(unittest.TestCase):

    def setUp(self):
        self.m = _load("cross_tier_report")

    def test_a_program_counts_once_per_family_and_reduced_only_tags_are_kept_out(self):
        recs = [{"tags": ["wrong_target", "wrong_formula", "over_dense_path"], "evidence": {}},
                {"tags": ["approach_param"], "evidence": {}}]
        n, fam, tag = self.m.shares(recs)
        self.assertEqual(n, 2)
        self.assertEqual(fam["SEMANTIC"], 1, "two SEMANTIC tags, one program")
        self.assertEqual(fam["TRAJECTORY"], 1, "over_dense_path is not in the S1 mix")
        self.assertNotIn("over_dense_path", tag)

    def test_prompted_tags_can_be_dropped(self):
        recs = [{"tags": ["object_name"], "evidence": {"object_name": "brown square block (prompted)"}}]
        self.assertEqual(self.m.shares(recs)[1]["INTERFACE"], 1)
        n, fam, _ = self.m.shares(recs, drop_prompted=True)
        self.assertEqual((n, fam["INTERFACE"]), (1, 0), "the program stays in the denominator")


if __name__ == "__main__":
    unittest.main()


class TestPresenceViewIsCountFree(unittest.TestCase):
    """The headline must not rest on the quantity the reliability check calls
    the least robust.

    Phase 9's strongest claim -- S1's scale advantage largely vanishes at
    S3/S4 -- is measured in tags per program, and the blind re-tag found 2.93
    causes per S3/S4 program against 3.43. A reviewer will put those two
    sentences side by side. `--presence` restates the same result in a form a
    rater can barely disagree about: did the repair have to fix three or more
    distinct things, yes or no.
    """

    ROOT = Path(__file__).resolve().parents[1]

    def _mod(self):
        import importlib.util
        import sys
        spec = importlib.util.spec_from_file_location(
            "cross_tier_report", self.ROOT / "scripts" / "cross_tier_report.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["cross_tier_report"] = mod
        spec.loader.exec_module(mod)
        return mod

    def test_it_bands_models_the_same_way_as_by_band(self):
        src = (self.ROOT / "scripts" / "cross_tier_report.py").read_text()
        # Two views of one result must not disagree about which band a model
        # is in. Both use `p <= 2 / <= 4 / < 12`.
        self.assertEqual(src.count('"<=2B" if p <= 2 else "3-4B" if p <= 4 '
                                   'else "7-9B" if p < 12 else ">=12B"'), 2)

    def test_it_counts_only_taxonomy_tags(self):
        src = (self.ROOT / "scripts" / "cross_tier_report.py").read_text()
        # over_dense_path, degenerate_repetition and truncated are not causes
        # in the S1 taxonomy; counting them would inflate S3/S4 by fiat.
        self.assertIn('len([t for t in r.get("tags", []) if t in FAMILY])', src)

    def test_the_presence_view_renders(self):
        m = self._mod()
        g = m.load(task=None, s1_source="hand")
        g = {k: [r for r in v
                 if not (r.get("truncated") and set(r["tags"]) <= {"syntax"})]
             for k, v in g.items()}
        out = m.presence(g)
        for want in (">= 1 FRAME tag", ">= 1 INTERFACE tag",
                     ">= 3 distinct causes", "S3/S4"):
            self.assertIn(want, out)
        # and it must carry the caveat: this is an additional view, not a
        # replacement for the per-band means
        self.assertIn("alongside", out)
