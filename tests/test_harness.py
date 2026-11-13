import re
import os
import json
import tempfile
import unittest
from pathlib import Path

from opencap import config
from opencap.capx_bridge import LaunchPlan, config_path, oracle_config_path
from opencap.cells import (ALL_TIERS, CORE_TASKS, Cell, WAVE1_TIERS, cell_exists,
                         grid, is_runnable, sweep_order)
from opencap.models import (BY_KEY, HOSTS, LADDER, OVERLAP_RUNG,
                          needs_h100, runnable_on, spine)
from opencap.schema import Episode, RESULT_COLUMNS
from opencap.store import LocalStore, deserialise, serialise


class TestContextPolicy(unittest.TestCase):
    def test_budget_arithmetic(self):
        self.assertEqual(config.PROMPT_BUDGET,
                         config.CONTEXT_WINDOW - config.GENERATION_RESERVE)
        self.assertEqual(config.PROMPT_BUDGET, 30_720)

    def test_window_is_the_common_floor(self):
        # 32K because Qwen3 below 8B and Gemma3-1B ship 32K. Raising this would
        # let large models win on window size rather than capability.
        self.assertEqual(config.CONTEXT_WINDOW, 32_768)

    def test_greedy_decoding(self):
        # The DEFAULT is frozen. The sampled arm opts in per process via
        # OPENCAP_TEMPERATURE rather than editing this file, because editing it
        # would change the condition every future greedy cell is collected
        # under, silently, for every arm.
        self.assertEqual(config.DEFAULT_TEMPERATURE, 0.0)
        self.assertEqual(config.TEMPERATURE, 0.0)
        self.assertFalse(config.is_sampled())

    def test_the_decode_condition_is_recorded_in_every_trace(self):
        # A run must be able to prove which condition it ran under. A gate that
        # leaves no trace is how two decode conditions would end up in one
        # corpus with no way to separate them.
        man = config.context_manifest()
        for key in ("temperature", "seed", "sample_index", "sampled"):
            self.assertIn(key, man)

    def test_noise_floor_is_recorded_not_assumed(self):
        # Greedy decoding is NOT bit-reproducible here: matched pairs differed
        # by 1 trial even with batching, prefix caching and CUDA graphs all
        # disabled. Effects below the floor are not measurements.
        self.assertEqual(config.BATCH_MAX, 1)
        self.assertGreater(config.GENERATION_NOISE_FRACTION, 0.0)
        self.assertLess(config.GENERATION_NOISE_FRACTION, 0.05)

    def test_prefix_caching_must_be_disabled(self):
        # Cross-cell contamination: same cell 27/30 fresh vs 0/30 reused with
        # prefix caching ON; 28/30 both ways with it OFF. This flag is
        # load-bearing -- without it model-major sweeps are invalid.
        self.assertTrue(config.DISABLE_PREFIX_CACHING)
        self.assertFalse(config.FRESH_ENGINE_PER_CELL)


class TestLadder(unittest.TestCase):
    def test_spine_sizes(self):
        self.assertEqual(len(spine("A")), 6)
        self.assertEqual(len(spine("B")), 4)
        self.assertEqual(len(spine("C")), 3)
        self.assertEqual(len(spine("D")), 5)
        self.assertEqual(len(spine("anchor")), 3)
        # Spine E (Qwen2.5-Coder) added after Coder-7B solved spill_wipe 29/30
        # -- a task every Spine A rung failed 0/30, 14B included -- by writing
        # a depth-2 nested loop none of them produce. The rungs locate where
        # that capability appears within a code-trained family.
        self.assertEqual(len(spine("E")), 2)
        # FP8 duplicates live in "control", never in a spine -- two entries at
        # one size would corrupt the curve.
        self.assertEqual(len(spine("control")), 1)
        self.assertEqual(len(LADDER), 24)

    def test_controls_are_not_swept(self):
        from opencap.cells import grid
        swept = {c.model for c in grid(1)} | {c.model for c in grid(2)}
        self.assertNotIn("qwen3-8b-fp8", swept)

    def test_no_duplicate_sizes_within_a_spine(self):
        for sp in ("A", "B", "C", "D"):
            sizes = [m.params_b for m in spine(sp)]
            self.assertEqual(len(sizes), len(set(sizes)),
                             f"spine {sp} has two rungs at the same size")

    def test_only_exact_generation_match_is_4b(self):
        a = {m.params_b for m in spine("A")}
        d = {m.params_b for m in spine("D")}
        self.assertEqual(a & d, {4.0},
                         "Spine A/D overlap must be 4B alone; compare curves, not points")

    def test_keys_unique(self):
        self.assertEqual(len(BY_KEY), len(LADDER))

    def test_host_split(self):
        # Corrected after qwen3-8b failed to serve on an L4: the first VRAM
        # model ignored KV cache, which at 32K context can exceed the weights.
        deferred = {m.key for m in needs_h100()}
        self.assertEqual(deferred, {"qwen3-32b", "qwen3vl-32b", "gemma3-27b",
                                    "qwen35-27b", "gpt-oss-120b"})

    def test_kv_uses_measured_value_when_known(self):
        # KV scales with layers x kv_heads, not parameters. vLLM reports
        # 4.5GiB for 8B and 5.0GiB for 14B; the per-parameter estimate says
        # 7.8GiB for 14B and would wrongly exile it to an H100.
        self.assertEqual(BY_KEY["qwen3-14b"].kv_gb, 5.0)
        self.assertTrue(BY_KEY["qwen3-14b"].fits(HOSTS["colab-a100"]))
        # unmeasured models keep the conservative per-parameter upper bound
        self.assertAlmostEqual(BY_KEY["qwen3-32b"].kv_gb, 32.0 * 0.56, places=3)

    def test_host_budget_follows_serving_config(self):
        # The budget is nameplate x gpu_memory_utilization, not the card.
        from opencap.models import GPU_MEMORY_UTILIZATION
        self.assertAlmostEqual(HOSTS["colab-a100"], 40.0 * GPU_MEMORY_UTILIZATION)

    def test_kv_cache_counted_not_treated_as_headroom(self):
        # The 32K cap is a control, so context cannot be shrunk to make a model
        # fit. qwen3-8b: 16GB weights + 4.5GB KV -> does not fit a 22GB L4.
        m = BY_KEY["qwen3-8b"]
        self.assertAlmostEqual(m.kv_gb, 4.5, delta=0.2)
        self.assertFalse(m.fits(HOSTS["colab-l4"]))
        self.assertTrue(m.fits(HOSTS["colab-a100"]))

    def test_anchor_20b_needs_a100(self):
        self.assertFalse(BY_KEY["gpt-oss-20b"].fits(HOSTS["colab-l4"]))
        self.assertTrue(BY_KEY["gpt-oss-20b"].fits(HOSTS["colab-a100"]))

    def test_overlap_rung_runs_on_both_hosts(self):
        # Host is confounded with size unless one rung runs on both.
        m = BY_KEY[OVERLAP_RUNG]
        self.assertTrue(m.fits(HOSTS["colab-a100"]))
        self.assertTrue(m.fits(HOSTS["h100"]))

    def test_anchor_hosts(self):
        # 120B needs more than one H100 even before KV -- hence TP=2.
        self.assertFalse(BY_KEY["gpt-oss-120b"].fits(HOSTS["h100"]))
        self.assertEqual(BY_KEY["gpt-oss-120b"].tensor_parallel, 2)

    def test_120b_uses_tensor_parallel(self):
        # MXFP4 weights leave ~13GB KV cache on one 80GB card at 32K context.
        self.assertEqual(BY_KEY["gpt-oss-120b"].tensor_parallel, 2)
        self.assertEqual(BY_KEY["qwen3-32b"].tensor_parallel, 1)

    def test_vllm_args_carry_the_context_cap(self):
        args = BY_KEY["qwen3-32b"].vllm_args()
        self.assertIn("--max-model-len", args)
        self.assertEqual(args[args.index("--max-model-len") + 1], "32768")

    def test_large_rungs_are_unquantised(self):
        # Was: assertIn("fp8", args), because bf16 starves the KV cache on an
        # 80GB H100. On a 141GB H200 the same rung fits at bf16 (84GB), so the
        # curve stays one precision and needs no fp8 control. Flipping this
        # back for 80GB hardware means reinstating that control too.
        for key in ("qwen3-32b", "qwen3vl-32b", "gemma3-27b", "qwen35-27b"):
            self.assertNotIn("--quantization", BY_KEY[key].vllm_args(), key)


class TestGrid(unittest.TestCase):
    def test_m2_excludes_text_only_models(self):
        self.assertFalse(is_runnable(BY_KEY["qwen3-8b"], "M2"))
        self.assertTrue(is_runnable(BY_KEY["qwen3vl-8b"], "M2"))

    def test_text_models_still_reach_m3(self):
        # M3's visual differencing module returns text, so text-only models run it.
        self.assertTrue(is_runnable(BY_KEY["qwen3-8b"], "M3"))

    def test_spine_d_is_wave1_only(self):
        self.assertTrue(is_runnable(BY_KEY["qwen35-9b"], "S1"))
        self.assertFalse(is_runnable(BY_KEY["qwen35-9b"], "M3"))

    def test_wave1_cell_count(self):
        # Every model except the single "control" rung, x 4 tiers x 7 tasks,
        # minus cube_restack's 3 excluded single-turn Wave 1 tiers.
        # Derived from LADDER rather than hard-coded, so adding a spine
        # updates the expectation instead of failing a stale constant.
        sweepable = len(LADDER) - len(spine("control"))
        expected = sweepable * (len(WAVE1_TIERS) * len(CORE_TASKS) - 3)
        self.assertEqual(len(list(grid(1))), expected)

    def test_seven_task_families(self):
        # A full listing of env_configs/ gives 7, matching the paper's
        # "across 7 tasks". An earlier pass that enumerated only guessed
        # directory names missed cube_lifting and cube_restack.
        self.assertEqual(len(CORE_TASKS), 7)
        for t in ("cube_lifting", "cube_restack", "nut_assembly"):
            self.assertIn(t, CORE_TASKS)

    def test_handover_has_no_m4(self):
        self.assertFalse(cell_exists("two_arm_handover", "M4"))
        self.assertTrue(cell_exists("two_arm_lift", "M4"))
        m4 = [c for c in grid(2) if c.tier == "M4"]
        self.assertTrue(all(c.task != "two_arm_handover" for c in m4))

    def test_sweep_is_model_major(self):
        order = sweep_order(1)
        models = [c.model for c in order]
        # each model appears in one contiguous block: load once, unload once
        self.assertEqual(len(set(models)), len({m for m in models}))
        first_seen = {}
        for i, m in enumerate(models):
            first_seen.setdefault(m, i)
        for m, start in first_seen.items():
            block = [i for i, x in enumerate(models) if x == m]
            self.assertEqual(block, list(range(start, start + len(block))))


class TestCellPaths(unittest.TestCase):
    def test_path_roundtrip(self):
        for tier in ALL_TIERS:
            c = Cell("qwen3-8b", tier, "peg_insertion")
            self.assertEqual(Cell.from_path(c.path()), c)

    def test_wave_encoded_in_path(self):
        self.assertIn("wave=1", Cell("qwen3-8b", "S1", "cube_lift").path())
        self.assertIn("wave=2", Cell("qwen3vl-8b", "M2", "cube_lift").path())

    def test_unrelated_paths_ignored(self):
        for junk in ("results/episodes.parquet", "README.md",
                     "traces/wave=1/model=x/tier=S1/task=y/notes.txt"):
            self.assertIsNone(Cell.from_path(junk))

    def test_rejects_unsafe_slugs(self):
        with self.assertRaises(ValueError):
            Cell("../escape", "S1", "cube_lift")
        with self.assertRaises(ValueError):
            Cell("qwen3-8b", "S9", "cube_lift")


class TestStoreResume(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = LocalStore(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _episodes(self, n=30):
        return [Episode(f"e{i}", "qwen3-8b", "S1", "cube_lift", i, 1).to_trace()
                for i in range(n)]

    def test_roundtrip(self):
        c = Cell("qwen3-8b", "S1", "cube_lift")
        self.store.write_cell(c, self._episodes())
        self.assertEqual(len(self.store.read_cell(c)), 30)

    def test_completed_derived_from_file_tree(self):
        # The tree IS the manifest: no separate index to drift.
        done = [Cell("qwen3-8b", "S1", t) for t in CORE_TASKS[:3]]
        for c in done:
            self.store.write_cell(c, self._episodes())
        self.assertEqual(self.store.completed(), set(done))

    def test_remaining_skips_completed(self):
        planned = sweep_order(1)
        self.store.write_cell(planned[0], self._episodes())
        remaining = self.store.remaining(planned)
        self.assertEqual(len(remaining), len(planned) - 1)
        self.assertNotIn(planned[0], remaining)

    def test_empty_cell_refused(self):
        # An unreadable-or-empty shard is worse than a missing one: completed()
        # would count it as done and the cell would never re-run.
        with self.assertRaises(ValueError):
            self.store.write_cell(Cell("qwen3-8b", "S1", "cube_lift"), [])

    def test_short_cell_is_a_data_integrity_hazard(self):
        # A cell killed mid-run committed 5 episodes instead of 30. Both the
        # parse and CaP-X's summary agreed on 5, so cross_check passed -- yet
        # completed() would mark it done and a resumed sweep would skip it.
        # The count guard lives in run_cell; this pins the invariant the
        # corpus relies on.
        from opencap.config import STATES_PER_TASK
        c = Cell("qwen3-8b", "S1", "cube_lifting")
        self.store.write_cell(c, self._episodes(5))
        self.assertIn(c, self.store.completed())
        self.assertNotEqual(len(self.store.read_cell(c)), STATES_PER_TASK)

    def test_no_partial_files_left_behind(self):
        self.store.write_cell(Cell("qwen3-8b", "S1", "cube_lift"), self._episodes())
        self.assertEqual(list(Path(self.tmp.name).rglob("*.partial")), [])


class TestScratchIsolation(unittest.TestCase):
    def test_scratch_is_env_overridable(self):
        # Output dirs key on model/tier/task only, so two conditions of the
        # same cell collide unless the scratch root differs. The A/B arms hit
        # exactly this: 60 trial dirs in one place, parse read a mixture.
        import subprocess, sys as _s
        from pathlib import Path
        src = (Path(__file__).resolve().parent.parent / "scripts" / "run_cell.py").read_text()
        self.assertIn('os.environ.get("OPENCAP_SCRATCH"', src)


class TestHostKind(unittest.TestCase):
    """host_kind must reflect the machine, not whatever a wrapper hard-coded.

    A hard-coded OPENCAP_HOST_KIND=colab-l4 in pilot.sh stamped real A100 cells
    as L4. Harmless to a success rate, fatal to analysis.host_effect(), which
    exists precisely to tell machines apart.
    """

    def _kind(self, gpu, env=None):
        import importlib.util, os, sys
        from pathlib import Path
        spec = importlib.util.spec_from_file_location(
            "rc", Path(__file__).resolve().parent.parent / "scripts" / "run_cell.py")
        m = importlib.util.module_from_spec(spec)
        sys.modules["rc"] = m
        spec.loader.exec_module(m)
        m._gpu_name = lambda: gpu
        old = os.environ.pop("OPENCAP_HOST_KIND", None)
        if env:
            os.environ["OPENCAP_HOST_KIND"] = env
        try:
            return m._host_kind()
        finally:
            os.environ.pop("OPENCAP_HOST_KIND", None)
            if old is not None:
                os.environ["OPENCAP_HOST_KIND"] = old

    def test_derives_from_gpu(self):
        self.assertEqual(self._kind("NVIDIA A100-SXM4-40GB"), "colab-a100")
        self.assertEqual(self._kind("NVIDIA L4"), "colab-l4")
        self.assertEqual(self._kind("Tesla T4"), "colab-t4")
        self.assertEqual(self._kind("NVIDIA H100 80GB HBM3"), "h100")
        self.assertEqual(self._kind("NVIDIA H200"), "h200")

    def test_explicit_env_still_wins(self):
        self.assertEqual(self._kind("NVIDIA L4", env="h100"), "h100")


class TestFailureTaxonomy(unittest.TestCase):
    def _t(self, success, rc, reward):
        from opencap.results import TrialResult
        return TrialResult(trial=1, reward=reward, success=success,
                           sandbox_rc=rc, attempts=1, dir_name="d")

    def test_code_that_raised_is_not_a_physical_failure(self):
        from opencap.results import classify_failure
        # This is the Qwen3 <think>-block case: SyntaxError, rc=1.
        self.assertEqual(classify_failure(self._t(False, 1, 0.001)), "api_misuse")

    def test_clean_run_no_movement(self):
        from opencap.results import classify_failure
        self.assertEqual(classify_failure(self._t(False, 0, 0.0)), "no_op")

    def test_clean_run_wrong_outcome(self):
        from opencap.results import classify_failure
        self.assertEqual(classify_failure(self._t(False, 0, 0.4)),
                         "physical_failure")

    def test_success(self):
        from opencap.results import classify_failure
        self.assertEqual(classify_failure(self._t(True, 0, 1.0)), "success")


class TestSchema(unittest.TestCase):
    def test_reward_is_recorded(self):
        # Binary success is uninformative at the floor -- four rungs all read
        # 0/30 -- but reward distinguishes "no_op" from "nearly succeeded".
        ep = Episode("e0", "qwen3-8b", "S1", "cube_lifting", 0, 1)
        ep.reward = 0.42
        self.assertIn("reward", ep.to_row())
        self.assertEqual(ep.to_row()["reward"], 0.42)

    def test_row_has_exactly_the_result_columns(self):
        row = Episode("e0", "qwen3-8b", "S1", "cube_lift", 0, 1).to_row()
        self.assertEqual(tuple(row), RESULT_COLUMNS)

    def test_trace_carries_turns_but_row_does_not(self):
        ep = Episode("e0", "qwen3-8b", "S1", "cube_lift", 0, 1)
        ep.turns = [{"code": "print(1)"}]
        self.assertIn("turns", ep.to_trace())
        self.assertNotIn("turns", ep.to_row())

    def test_extension_never_lies_about_format(self):
        # A shard named .jsonl.zst must actually be zstd. The original
        # "graceful fallback" wrote plain JSONL when the library was absent,
        # producing a corpus only the writing machine could read -- which is
        # how the first real cell became unreadable off-pod.
        from opencap.store import _ZSTD_MAGIC, serialise, zstd
        blob = serialise([{"a": 1}])
        if zstd is not None:
            self.assertTrue(blob.startswith(_ZSTD_MAGIC))
        # and a legacy uncompressed shard still round-trips
        self.assertEqual(deserialise(b'{"a": 1}\n'), [{"a": 1}])

    def test_serialise_is_jsonl(self):
        blob = serialise([{"a": 1}, {"a": 2}])
        self.assertEqual(deserialise(blob), [{"a": 1}, {"a": 2}])


class TestCapxBridge(unittest.TestCase):
    def test_naming_quirks(self):
        self.assertEqual(config_path("cube_stack", "S3"),
                         "env_configs/cube_stack/franka_robosuite_cube_stack_reduced_api.yaml")
        # two_arm_handover drops the franka_robosuite_ prefix ...
        self.assertEqual(config_path("two_arm_handover", "S1"),
                         "env_configs/two_arm_handover/two_arm_handover_privileged.yaml")
        # ... and both two-arm tasks use _reduced, not _reduced_api
        self.assertEqual(config_path("two_arm_lift", "S4"),
                         "env_configs/two_arm_lift/franka_robosuite_two_arm_lift_reduced_exampleless.yaml")

    def test_restack_single_turn_excluded(self):
        # Fig. 17: CaP-X evaluated Re-Stack only in the multi-turn setup,
        # because single-turn-without-vision fails for task-design reasons.
        from opencap.cells import cell_exists
        for tier in ("S1", "S2", "S3", "S4"):
            self.assertFalse(cell_exists("cube_restack", tier))
        for tier in ("M1", "M3", "M4"):
            self.assertTrue(cell_exists("cube_restack", tier))

    def test_oracle_paths(self):
        # Deterministic, LLM-free reference programs -- the Gate 0 anchor.
        self.assertTrue(oracle_config_path("cube_stack").endswith(
            "franka_robosuite_cube_stack_privileged_oracle.yaml"))
        self.assertTrue(oracle_config_path("two_arm_handover", False).endswith(
            "two_arm_handover_oracle.yaml"))

    def test_missing_config_raises(self):
        with self.assertRaises(KeyError):
            config_path("two_arm_handover", "M4")

    def _plan(self, tier):
        return LaunchPlan("cube_stack", tier, "Qwen/Qwen3-8B",
                          "http://x/chat/completions", config.VDM_MODEL,
                          "http://y/chat/completions", "/tmp/out")

    def test_overrides_capx_defaults(self):
        argv = self._plan("S1").argv()
        # CaP-X defaults temperature 1.0 and max_tokens 20480; both would
        # silently break our controls.
        self.assertEqual(argv[argv.index("--temperature") + 1], "0.0")
        self.assertEqual(argv[argv.index("--max-tokens") + 1], "2048")
        # tyro rejects lowercase booleans: "invalid choice 'false'".
        self.assertEqual(argv[argv.index("--record-video") + 1], "False")
        self.assertEqual(argv[argv.index("--use-parallel-ensemble") + 1], "False")
        for flag in ("--record-video", "--use-parallel-ensemble", "--use-multimodel"):
            self.assertIn(argv[argv.index(flag) + 1], ("True", "False", "None"))

    def test_vdm_pinned_only_where_used(self):
        self.assertNotIn("--visual-differencing-model", self._plan("S1").argv())
        self.assertNotIn("--visual-differencing-model", self._plan("M1").argv())
        for tier in ("M3", "M4"):
            argv = self._plan(tier).argv()
            i = argv.index("--visual-differencing-model")
            self.assertEqual(argv[i + 1], config.VDM_MODEL)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestEngineReadinessGate(unittest.TestCase):
    """Guards for the 2026-09-09 cross-contamination.

    An orphaned pilot.sh held :8110 with coder-1.5b; the relaunch's own
    qwen3-0.6b shim died on "Address already in use"; the readiness check --
    a bare curl on /v1/models -- passed anyway, because that endpoint answers
    from the shim's --model flag without touching the engine. qwen3-0.6b's
    episodes were generated by coder-1.5b's weights and labelled 0.6b.
    """

    SWEEP = Path(__file__).resolve().parents[1] / "scripts" / "run_sweep.sh"
    PILOT = Path(__file__).resolve().parents[1] / "scripts" / "pilot.sh"

    def test_readiness_probes_real_generation(self):
        # /v1/models is metadata and stays 200 after the EngineCore dies.
        self.assertIn("/v1/chat/completions", self.SWEEP.read_text())

    def test_readiness_checks_served_model_identity(self):
        src = self.SWEEP.read_text()
        self.assertIn("serving_expected_model", src)
        self.assertIn("PORT HIJACK", src)

    def test_hijack_check_precedes_the_cell_loop(self):
        src = self.SWEEP.read_text()
        self.assertLess(src.index("PORT HIJACK"), src.index("run_cell.py"),
                        "identity check must gate the cells, not follow them")

    @staticmethod
    def _launch_offset(src):
        """Offset of the line that actually launches the shim.

        Not a bare substring search: `setsid` is also discussed in the comment
        block above the port check, so matching prose puts the "launch" before
        the guard and inverts this test.
        """
        off = 0
        for line in src.splitlines(keepends=True):
            if line.strip().startswith("setsid "):
                return off
            off += len(line)
        raise AssertionError("no setsid launch line in run_sweep.sh")

    def test_refuses_to_start_on_an_occupied_port(self):
        src = self.SWEEP.read_text()
        occupied = src.index("already in use")
        self.assertLess(occupied, self._launch_offset(src),
                        "port check must precede the shim launch")

    def test_pilot_is_single_instance(self):
        # A relaunch must not coexist with an orphan of the previous attempt.
        self.assertIn("flock -n 9", self.PILOT.read_text())


class TestGpuPinning(unittest.TestCase):
    """Every GPU query must name the card this sweep owns.

    nvidia-smi reads NVML, which ignores CUDA_VISIBLE_DEVICES and always
    enumerates from physical index 0. On Colab there was one card and the
    distinction was invisible; on a shared box an unqualified query reports
    somebody else's job. Two things break quietly: the post-teardown VRAM
    check never clears (so it warns on every model), and the utilisation
    sampler that sizes the worker count records the wrong card entirely.
    """

    SWEEP = Path(__file__).resolve().parents[1] / "scripts" / "run_sweep.sh"
    PILOT = Path(__file__).resolve().parents[1] / "scripts" / "pilot.sh"
    CELL = Path(__file__).resolve().parents[1] / "scripts" / "run_cell.py"

    def test_every_nvidia_smi_call_names_a_device(self):
        for path in (self.SWEEP, self.PILOT):
            for i, line in enumerate(path.read_text().splitlines(), 1):
                stripped = line.strip()
                if stripped.startswith("#") or "nvidia-smi" not in stripped:
                    continue
                self.assertIn("-i ", stripped,
                              f"{path.name}:{i} queries NVML without -i, so it "
                              f"reports physical GPU 0: {stripped}")

    def test_sweep_exports_cuda_visible_devices(self):
        src = self.SWEEP.read_text()
        self.assertIn('export CUDA_VISIBLE_DEVICES="$GPU"', src)
        # MuJoCo's EGL backend does its own device enumeration and would
        # otherwise render on physical GPU 0 whatever CUDA is told.
        self.assertIn("MUJOCO_EGL_DEVICE_ID", src)

    def test_env_pins_every_caller_not_only_the_sweep(self):
        # Gate 0 and verify_repairs.py never pass through run_sweep.sh. With
        # only OPENCAP_GPU set, their IK server and EGL context landed on GPU 0.
        env = (Path(__file__).resolve().parents[1] / "scripts" / "env_server.sh").read_text()
        self.assertIn('export CUDA_VISIBLE_DEVICES="$OPENCAP_GPU"', env)
        self.assertIn("MUJOCO_EGL_DEVICE_ID", env)

    def test_gpu_name_honours_the_pin(self):
        # host_kind is derived from this, and host_kind is what
        # analysis.host_effect() groups by.
        self.assertIn('os.environ.get("OPENCAP_GPU"', self.CELL.read_text())

    def test_engine_interpreter_is_checked_before_use(self):
        # cap-x pins conflicting vLLM versions behind its own extras, so the
        # engine runs from a separate venv. A missing import must fail loudly
        # here rather than as an unexplained engine timeout 3 minutes later.
        src = self.SWEEP.read_text()
        self.assertIn("OPENCAP_VLLM_PYTHON", src)
        launch = TestEngineReadinessGate._launch_offset(src)
        self.assertLess(src.index("cannot import vllm"), launch)


class TestTimeoutGuard(unittest.TestCase):
    """A timed-out request must not enter the corpus as a model failure.

    CaP-X's HTTP client gives up on a generation after 200s and records the
    trial with an empty program, reward 0.0 and `Num Code Blocks: 0`. That is
    indistinguishable from a model that wrote nothing useful, so four cells
    committed on 2026-09-10 reading 0/30 with ~25 of each 30 scenes never
    attempted. Same failure shape as a corpus with no stderr: a clean-looking
    number that cannot be attributed to anything.
    """

    CELL = Path(__file__).resolve().parents[1] / "scripts" / "run_cell.py"

    def test_fraction_counts_timed_out_trials(self):
        import tempfile
        from opencap.results import timeout_fraction
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for i in range(10):
                d = root / f"trial_{i:02d}_sandboxrc_0_reward_0.000_taskcompleted_0"
                d.mkdir()
                (d / "summary.txt").write_text(
                    "Trial 5 timed out after 1000 seconds." if i < 3 else "fine")
            frac, n_to, n = timeout_fraction(root)
            self.assertEqual((n_to, n), (3, 10))
            self.assertAlmostEqual(frac, 0.3)

    def test_empty_dir_is_not_a_timeout(self):
        import tempfile
        from opencap.results import timeout_fraction
        with tempfile.TemporaryDirectory() as td:
            self.assertEqual(timeout_fraction(Path(td)), (0.0, 0, 0))

    def test_run_cell_refuses_before_committing(self):
        src = self.CELL.read_text()
        self.assertIn("timeout_fraction", src)
        self.assertIn("MAX_TIMEOUT_FRACTION", src)
        # The check must gate the commit, not follow it.
        self.assertLess(src.index("timeout_fraction(run_dir)"),
                        src.index("store.write_cell"))


class TestDeadServerGuard(unittest.TestCase):
    """A dead IK server must invalidate a verification, even partially.

    The check used to require that EVERY trial failed. On 2026-09-10 a repair
    scored 26/30 with four trials dying on "Connection refused" from :8116 --
    a second launch.py was running concurrently and both bind that fixed port
    -- and the shortfall was charged to the code. Partial loss is the normal
    shape of this failure, so the guard counts trials rather than requiring a
    clean sweep of them.
    """

    VERIFY = Path(__file__).resolve().parents[1] / "scripts" / "verify_repairs.py"

    def test_guard_does_not_require_total_failure(self):
        src = self.VERIFY.read_text()
        self.assertIn("n_unreachable", src)
        # the old all()-gated form must be gone
        self.assertNotIn("all(not t.success for t in trials_parsed)", src)

    def test_guard_reports_how_many_trials_were_lost(self):
        src = self.VERIFY.read_text()
        self.assertIn("unreachable_trials", src)

    def test_guard_precedes_the_cross_check(self):
        src = self.VERIFY.read_text()
        self.assertLess(src.index("n_unreachable"), src.index("cross_check(run_dir)"))


class TestSeedingPatchCoversEveryTask(unittest.TestCase):
    """ARTIFACT #13: the seeding patch targeted three simulator files and every
    task was assumed to route through one of them. Three of the four S1 tasks
    did not -- cube_lifting, spill_wipe and nut_assembly each bind their own
    simulator -- so their scenes were random on every run and every paired
    statistic on them compared different scenes. verify_pairing_states.py
    passed because it probed cube_stack alone.

    These tests pin the two things that let it through: the target list must
    name every simulator an S1 config binds, and the pairing check must probe
    every task rather than one.
    """

    ROOT = Path(__file__).resolve().parents[1]

    #: low_level binding -> simulator module, from capx/envs/simulators/__init__.py
    BINDINGS = {
        "franka_robosuite_cube_lift_low_level":    "robosuite_cube_lift.py",
        "franka_robosuite_cubes_low_level":        "robosuite_cubes.py",
        "franka_robosuite_spill_wipe_low_level":   "robosuite_spill_wipe.py",
        "franka_robosuite_nut_assembly_low_level": "robosuite_nut_assembly.py",
    }

    def test_every_s1_simulator_is_a_patch_target(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "ps", self.ROOT / "scripts" / "patch_seeding.py")
        ps = importlib.util.module_from_spec(spec); spec.loader.exec_module(ps)
        targets = {Path(t).name for t in ps.TARGETS}
        for binding, module in self.BINDINGS.items():
            self.assertIn(module, targets,
                          f"{binding} resets through {module}, which the "
                          f"seeding patch does not touch -- its scenes are random")

    def test_patch_handles_both_reset_shapes(self):
        # nut_assembly keeps reset()'s return value; the others call it bare.
        src = (self.ROOT / "scripts" / "patch_seeding.py").read_text()
        self.assertIn("first_obs = self.robosuite_env.reset()", src)
        self.assertIn("PATTERNS", src)

    def test_pairing_check_probes_every_task(self):
        src = (self.ROOT / "scripts" / "verify_pairing_states.py").read_text()
        for task in ("cube_lifting", "cube_stack", "spill_wipe", "nut_assembly"):
            self.assertIn(f'"{task}"', src)
        # and no longer hard-codes one config
        self.assertNotIn("franka_robosuite_cube_stack_privileged_oracle.yaml", src)


class TestVerifyStartsFromEmptyDir(unittest.TestCase):
    """CaP-X names trial dirs by outcome, so a re-run into an existing output
    dir overwrites same-outcome trials and adds differing ones; the parse then
    reads a mixture of two runs. Stage 8 of the rebuild ran its repair and
    minimal passes into one scratch directory and every minimal record read
    the repair run's leftovers (31 trial dirs per pid). Same trap as the
    shared-scratch A/B arms in §7, reproduced inside the verifier."""

    ROOT = Path(__file__).resolve().parents[1]

    def test_verifier_clears_the_pid_dir_before_launching(self):
        src = (self.ROOT / "scripts" / "verify_repairs.py").read_text()
        self.assertIn("shutil.rmtree", src)
        self.assertLess(src.index("shutil.rmtree"), src.index("build_config(capx, task, code"))

    def test_pipeline_passes_use_separate_scratch(self):
        src = (self.ROOT / "scripts" / "run_all_h200.sh").read_text()
        self.assertIn("verify_repairs\"", src.replace("/verify_repairs\"", "verify_repairs\""))
        self.assertIn("/verify_minimal", src)
        self.assertIn("/verify_repairs", src)

    def test_verifier_creates_the_out_dir_before_spending_gpu_time(self):
        # 25 second-draw records were lost to a FileNotFoundError on the
        # first write, after the runs had completed; the console `| grep`
        # hid the traceback. The results were rebuilt from the run dirs.
        src = (self.ROOT / "scripts" / "verify_repairs.py").read_text()
        self.assertIn("out_path.parent.mkdir(parents=True, exist_ok=True)", src)
        self.assertLess(src.index("out_path.parent.mkdir"),
                        src.index("for pid, rep in todo"))


class TestProvenanceFields(unittest.TestCase):
    """vllm_version and harness_commit were empty on every Colab episode.

    STATUS.md lists turns_used/tokens_out as dead fields; these two were dead
    the same way. They matter more than most: the serving stack and the
    harness revision are exactly what is confounded with the host when a
    corpus is collected on two machines.
    """

    CELL = Path(__file__).resolve().parents[1] / "scripts" / "run_cell.py"

    def test_episode_records_the_serving_stack(self):
        src = self.CELL.read_text()
        self.assertIn('vllm_version=os.environ.get("OPENCAP_VLLM_VERSION"', src)
        self.assertIn("harness_commit=_harness_commit()", src)

    def test_sweep_exports_the_version_it_serves(self):
        sweep = (Path(__file__).resolve().parents[1]
                 / "scripts" / "run_sweep.sh").read_text()
        self.assertIn("export OPENCAP_VLLM_VERSION=", sweep)


class TestPhase2Grid(unittest.TestCase):
    """The completion grid: four families, four S1 tasks, nothing above 32B."""

    FAMILIES = ("A", "B", "C", "D")
    TASKS = ("cube_lifting", "cube_stack", "spill_wipe", "nut_assembly")

    def models(self):
        return [m for m in LADDER if m.spine in self.FAMILIES]

    def test_no_model_above_32b(self):
        for m in self.models():
            self.assertLessEqual(m.params_b, 32.0, m.key)

    def test_one_precision_across_the_curve(self):
        # A quantised rung on a bf16 curve confounds precision with size.
        # These fit bf16 on a 141GB H200; see the precision note in models.py.
        precisions = {m.precision for m in self.models()}
        self.assertEqual(precisions, {"bf16"},
                         f"mixed precision on the curve: {precisions}")

    def test_every_grid_task_exists_at_s1(self):
        # cube_restack is deliberately absent: it has no S1 config, so it
        # cannot satisfy the privileged single-turn constraint.
        for task in self.TASKS:
            self.assertTrue(cell_exists(task, "S1"), task)
        self.assertFalse(cell_exists("cube_restack", "S1"))

    def test_grid_size(self):
        self.assertEqual(len(self.models()), 18)
        self.assertEqual(len(self.models()) * len(self.TASKS), 72)


class TestSaveRepairDoesNotClobber(unittest.TestCase):
    """results/repairs/<task>.json holds EVERY repair for a task in one dict.

    Writing it from a fresh dict deletes the others, and nothing downstream
    notices -- the missing entries look identical to programs nobody repaired.
    """

    def setUp(self):
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        self.tmp = Path(tempfile.mkdtemp())

    def test_second_repair_preserves_the_first(self):
        from save_repair import save
        save("cube_lifting", "aaa", "code A", root=self.tmp)
        save("cube_lifting", "bbb", "code B", root=self.tmp)
        got = json.loads((self.tmp / "results/repairs/cube_lifting.json").read_text())
        self.assertEqual(sorted(got), ["aaa", "bbb"])
        self.assertEqual(got["aaa"]["code"], "code A")

    def test_replacing_an_existing_pid_is_refused(self):
        from save_repair import save
        save("cube_stack", "aaa", "first", root=self.tmp)
        with self.assertRaises(SystemExit):
            save("cube_stack", "aaa", "second", root=self.tmp)
        got = json.loads((self.tmp / "results/repairs/cube_stack.json").read_text())
        self.assertEqual(got["aaa"]["code"], "first")

    def test_force_replaces_only_that_entry(self):
        from save_repair import save
        save("spill_wipe", "aaa", "first", root=self.tmp)
        save("spill_wipe", "bbb", "other", root=self.tmp)
        save("spill_wipe", "aaa", "second", force=True, root=self.tmp)
        got = json.loads((self.tmp / "results/repairs/spill_wipe.json").read_text())
        self.assertEqual(got["aaa"]["code"], "second")
        self.assertEqual(got["bbb"]["code"], "other")

    def test_minimal_and_repairs_are_separate_files(self):
        from save_repair import save
        save("cube_lifting", "aaa", "raw", root=self.tmp)
        save("cube_lifting", "aaa", "min", minimal=True, root=self.tmp)
        base = json.loads((self.tmp / "results/repairs/cube_lifting.json").read_text())
        mini = json.loads((self.tmp / "results/minimal/cube_lifting.json").read_text())
        self.assertEqual(base["aaa"]["code"], "raw")
        self.assertEqual(mini["aaa"]["code"], "min")


class TestFrozenStores(unittest.TestCase):
    """A cell's shard path is (wave, model, tier, task), so re-running a cell
    into the same store REPLACES it in place. There is no version history and
    nothing left to compare against. For a corpus other analyses are measured
    against, that silently moves a reference point."""

    def setUp(self):
        from opencap import store as store_mod
        self.store = store_mod
        os.environ.pop("OPENCAP_ALLOW_FROZEN", None)

    def tearDown(self):
        os.environ.pop("OPENCAP_ALLOW_FROZEN", None)

    def test_published_baseline_is_frozen(self):
        self.assertIn("savoji/capx-ab-on", self.store.FROZEN_STORES)
        self.assertIn("savoji/capx-promptfix", self.store.FROZEN_STORES)

    def test_write_to_frozen_store_is_refused(self):
        with self.assertRaises(PermissionError):
            self.store._check_not_frozen("savoji/capx-ab-on")

    def test_new_store_is_writable(self):
        self.store._check_not_frozen("savoji/capx-ab-on-seeded")  # must not raise

    def test_explicit_override_is_honoured(self):
        os.environ["OPENCAP_ALLOW_FROZEN"] = "1"
        self.store._check_not_frozen("savoji/capx-ab-on")  # must not raise

    def test_write_cell_consults_the_guard(self):
        # A LocalStore carrying a frozen repo_id must be refused too, so the
        # check lives on the write path rather than on one backend.
        from opencap.cells import Cell
        s = self.store.LocalStore(tempfile.mkdtemp())
        s.repo_id = "savoji/capx-ab-on"
        ep = Episode(episode_id="x", model="qwen3-4b", tier="S1",
                     task="cube_lifting", state_id=0, wave=1).to_trace()
        with self.assertRaises(PermissionError):
            s.write_cell(Cell(model="qwen3-4b", tier="S1", task="cube_lifting"),
                         [ep], verified=True)


class TestCorpusScanAndRerun(unittest.TestCase):
    """Guards for the timeout artifact (2026-09-10) and for re-collection.

    run_cell.py's timeout check reads a run directory, so it protects
    collection and is blind to already-committed cells. Two contaminated cells
    in the Colab half went unnoticed for weeks because nothing scanned the
    corpus itself.
    """

    SCAN = Path(__file__).resolve().parents[1] / "scripts" / "scan_corpus.py"
    RERUN = Path(__file__).resolve().parents[1] / "scripts" / "run_rerun_h200.sh"

    def test_empty_limit_separates_observed_data(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("scan_corpus", self.SCAN)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        # smallest contaminated cell seen was 0.83 empty; clean cells were 0.00
        self.assertLess(mod.EMPTY_LIMIT, 0.83)
        self.assertGreater(mod.EMPTY_LIMIT, 0.0)

    def test_rerun_refuses_to_write_a_published_store(self):
        src = self.RERUN.read_text()
        self.assertIn("REFUSING", src)
        self.assertLess(src.index("REFUSING"), src.index("pilot.sh"),
                        "the refusal must precede the run, not follow it")

    def test_rerun_pins_the_timeout_fix(self):
        # 5 workers behind a batch-1 shim is what produced the empty episodes.
        self.assertIn("OPENCAP_WORKERS", self.RERUN.read_text())
        self.assertIn("OPENCAP_WORKERS:-2", self.RERUN.read_text())

    def test_rerun_scans_its_own_output(self):
        self.assertIn("scan_corpus.py", self.RERUN.read_text())


class TestSingleHostRebuild(unittest.TestCase):
    """Artifact #12: the generated program itself changes across GPU
    architectures in half of all cells. Anything that lets a Colab-era corpus
    back into an analysis reintroduces that confound."""

    S = Path(__file__).resolve().parents[1] / "scripts"

    def test_every_colab_store_is_retired(self):
        from opencap.store import FROZEN_STORES, SUPERSEDED_BY
        for r in ("capx-ab-on", "capx-ab-off", "capx-promptfix",
                  "capx-entrypoint", "capx-oracle", "capx-scaling-traces"):
            self.assertIn(f"savoji/{r}", FROZEN_STORES, r)
        # a refusal that names the replacement beats one that just says no
        # Replacements now carry -seeded: -h200 fixed the host, and artifact
        # #13 then retired those too for drawing random scenes.
        for old, new in SUPERSEDED_BY.items():
            self.assertTrue(new.endswith("-seeded"), new)
            self.assertNotIn(new, FROZEN_STORES)

    def test_arm_runner_covers_every_arm(self):
        src = (self.S / "run_arm.sh").read_text()
        for arm in ("baseline", "numpy_off", "promptfix", "entrypoint"):
            self.assertIn(f"{arm})", src, arm)

    def test_arm_flags_are_set_or_unset_never_inherited(self):
        # An arm that picks up a stray flag from the shell is
        # indistinguishable from a real effect.
        src = (self.S / "run_arm.sh").read_text()
        for flag in ("OPENCAP_PREIMPORT_NUMPY", "OPENCAP_PROMPT_FIX",
                     "OPENCAP_CALL_ENTRYPOINT"):
            self.assertIn(f"unset {flag}", src, flag)

    def test_arm_runner_refuses_retired_stores(self):
        src = (self.S / "run_arm.sh").read_text()
        self.assertIn("REFUSING", src)
        self.assertLess(src.index("REFUSING"), src.index("pilot.sh"))

    def test_pipeline_never_aborts_on_a_failed_stage(self):
        # Assert the MECHANISM, not a comment: a wrapper that records the
        # failure and returns 0 so the next stage still runs. The previous
        # version of this test anchored on a comment string and broke when the
        # file was rewritten, while the behaviour was never in question.
        src = (self.S / "run_all_h200.sh").read_text()
        self.assertNotIn("set -e\n", src)
        self.assertIn("RC=$r", src)
        self.assertIn("return 0", src)

    def test_pipeline_reverifies_into_new_files(self):
        # Reusing an old --out would skip every repair already at ceiling,
        # which is exactly the set we need re-measured on the seeded scenes.
        # Phase 4 wrote the -h200 files; the seeded run writes its own pair
        # and must not write into either earlier one.
        src = (self.S / "run_all_h200.sh").read_text()
        self.assertIn("--out results/verify_seeded.json", src)
        self.assertIn("--out results/minimal_verify_seeded.json", src)
        self.assertNotIn("--out results/verify_h200.json", src)
        self.assertNotIn("--out results/verify.json", src)

    def test_oracle_records_provenance(self):
        src = (self.S / "run_gate0_oracle.sh").read_text()
        for f in ("host_kind", "host_gpu", "harness_commit"):
            self.assertIn(f, src, f)


class TestCompleteRebuild(unittest.TestCase):
    """Every pre-2026-09-11 conclusion was computed over Colab data. The
    rebuild must re-validate the CONTROLS before the arms, because three
    constants in config.py are themselves Colab measurements."""

    S = Path(__file__).resolve().parents[1] / "scripts"

    def test_controls_run_before_any_arm(self):
        src = (self.S / "run_all_h200.sh").read_text()
        self.assertLess(src.index("run_controls.sh"), src.index("run_arm.sh"),
                        "a control validated elsewhere is an assumption here")

    def test_controls_cover_every_colab_derived_constant(self):
        src = (self.S / "run_controls.sh").read_text()
        self.assertIn("OPENCAP_BATCH_MAX=4", src)            # BATCH_MAX = 1
        self.assertIn("OPENCAP_ALLOW_PREFIX_CACHE=1", src)   # DISABLE_PREFIX_CACHING
        self.assertIn("replicate", src)                    # GENERATION_NOISE

    def test_control_subset_is_not_all_floor_cells(self):
        # A noise control measured only on 0/30 and 30/30 cells reports a floor
        # of zero and proves nothing; flips concentrate mid-range.
        src = (self.S / "run_controls.sh").read_text()
        self.assertIn("mid-range", src)

    def test_prefix_cache_knob_defaults_to_deterministic(self):
        src = (self.S / "run_sweep.sh").read_text()
        self.assertIn("OPENCAP_ALLOW_PREFIX_CACHE:-0", src)
        self.assertIn("--deterministic", src)

    def test_migration_refuses_bad_cells(self):
        src = (self.S / "copy_clean_cells.py").read_text()
        self.assertIn("EMPTY_LIMIT", src)
        self.assertIn('WANT_HOST = "h200"', src)

    def test_pipeline_does_not_migrate_unseeded_cells(self):
        # Migration was right while the only defect was the host. Artifact #13
        # made it wrong: every -h200 cell predates the seeding fix, so a
        # migrated cell would carry scenes nothing else in the corpus shares,
        # and state-level pairing is the whole reason for re-collecting.
        src = (self.S / "run_all_h200.sh").read_text()
        self.assertNotIn("copy_clean_cells.py", src)


class TestClaimsRegisterAndAudit(unittest.TestCase):
    """The register re-derives what the rebuild settled instead of listing
    it as pending; the audit does not fail the oracle store for the serving
    stack it never ran."""

    ROOT = Path(__file__).resolve().parents[1]

    def test_register_reads_the_control_arm_and_verify_sources(self):
        src = (self.ROOT / "scripts" / "claims_register.py").read_text()
        for store in ("capx-ctrl-replicate-seeded", "capx-ctrl-batch-seeded",
                      "capx-ctrl-prefix-seeded", "capx-ab-off-seeded",
                      "capx-entrypoint-seeded", "capx-promptfix-seeded"):
            self.assertIn(store, src)
        self.assertIn("verify_h200.json", src)
        # stage 8 of the seeded run writes verify_seeded.json; the register
        # reads that when it exists, and the unseeded file otherwise
        self.assertIn("verify_seeded.json", src)
        # an arm that was never collected reads as an empty store (open_store
        # creates what it cannot find) and must not become a verdict on 0 cells
        self.assertIn("not scores", src)
        # the stage-settled claims are no longer static PENDING entries
        pending = src[src.index("PENDING = ["):src.index("]", src.index("PENDING = ["))]
        for settled in ("stage 1", "stage 5", "stage 6", "stage 7", "stage 8"):
            self.assertNotIn(settled, pending)
        # The taxonomy and the distance distribution were PENDING on "the
        # rebuild was told not to tag". They are done, and they are RE-DERIVED
        # from results/tags_seeded.json rather than written down -- the first
        # version of those numbers was published out of a script that is not in
        # this repo, which is how a tagger bug reached a headline table.
        # Pinned in both directions: a claim that moves back to static text
        # must break this loudly.
        self.assertNotIn("told not to tag", pending)
        self.assertNotIn("Defect taxonomy", pending)
        self.assertNotIn("Distance to solution", pending)
        self.assertIn("def taxonomy_rows", src)
        self.assertIn("rows += taxonomy_rows()", src)
        self.assertIn("tags_seeded.json", src)
        # the retraction itself is recorded where the claim was made
        self.assertIn("RETRACTED", src)
        # Two rows sat PENDING on "once the repairs are tagged" after the
        # tagging was done. A register whose PENDING list drifts is no better
        # than remembering, so they are re-derived now. Pinned in both
        # directions: moving one back to static text must break this.
        self.assertNotIn("RE-EXAMINE on the seeded corpus", pending)
        self.assertNotIn("trajectory hygiene", pending)
        self.assertIn("def legacy_rows", src)
        self.assertIn("rows += legacy_rows()", src)
        # The last GPU-run entry, the trial 9 / 13 alternation, had its run on
        # 2026-10-13 and is re-derived now. Pinned in both directions.
        self.assertNotIn("WORKERS=1 ON THIS PROGRAM", pending)
        self.assertIn("def alternation_rows", src)
        self.assertIn("rows += alternation_rows()", src)

    def test_register_discounts_only_regime_changes_on_noisy_cells(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "claims_register", self.ROOT / "scripts" / "claims_register.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.assertGreaterEqual(mod.NOISY_MOVE, 15)
        self.assertLess(mod.MOVE, mod.NOISY_MOVE)

    def test_audit_accepts_llm_free_oracle_episodes(self):
        src = (self.ROOT / "scripts" / "audit_provenance.py").read_text()
        self.assertIn('e.get("model") == "oracle"', src)
        # still demands a host and a harness commit of them
        self.assertIn("no_commit += not e.get(\"harness_commit\")", src)


class TestSeededRebuild(unittest.TestCase):
    """Artifact #13: the seeding patch reached robosuite_cubes.py only, so
    cube_lifting, spill_wipe and nut_assembly drew a random scene every run.
    Everything collected before the fix lacks state-level pairing on three
    tasks in four."""

    S = Path(__file__).resolve().parents[1] / "scripts"

    def test_patch_covers_every_simulator_an_s1_task_binds(self):
        src = (self.S / "patch_seeding.py").read_text()
        for sim in ("robosuite_cubes.py", "robosuite_cube_lift.py",
                    "robosuite_spill_wipe.py", "robosuite_nut_assembly.py"):
            self.assertIn(sim, src, sim)

    def test_unseeded_stores_are_retired(self):
        from opencap.store import FROZEN_STORES, SUPERSEDED_BY
        # These names are the RETIRED stores, asserted as data. A rename sweep
        # turned three of them into -seeded once, which made this test assert
        # that the LIVE stores were frozen -- and it still passed, because the
        # live names were absent from FROZEN_STORES for the opposite reason.
        for r in ("capx-ab-on-h200", "capx-ab-off-h200", "capx-promptfix-h200",
                  "capx-entrypoint-h200", "capx-oracle-h200"):
            self.assertIn(f"savoji/{r}", FROZEN_STORES, r)
            self.assertTrue(SUPERSEDED_BY[f"savoji/{r}"].endswith("-seeded"), r)
        for r in ("capx-ab-on-seeded", "capx-oracle-seeded",
                  "capx-ctrl-replicate-seeded"):
            self.assertNotIn(f"savoji/{r}", FROZEN_STORES, r)

    def test_arms_write_to_seeded_stores_only(self):
        src = (self.S / "run_arm.sh").read_text()
        self.assertIn("capx-ab-on-seeded", src)
        self.assertNotIn("capx-ab-on-h200", src.split("REFUSING")[0])

    def test_nut_assembly_is_back_in_the_panel(self):
        # It was dropped on oracle draws of 11/30 and 13/30 -- random-scene
        # draws, not a gate failure. Seeded, its oracle scores 30/30.
        src = (self.S / "run_arm.sh").read_text()
        self.assertIn("nut_assembly", src)
        self.assertIn("21 models x 4 tasks = 84 cells", src)

    def test_control_subset_includes_a_task_that_was_already_seeded(self):
        # The first control run used exactly the two unseeded tasks, so it
        # measured scene randomness and reported it as a harness noise floor.
        src = (self.S / "run_controls.sh").read_text()
        self.assertIn("cube_stack", src.split("CTRL_MODELS")[0] + src)

    def test_pipeline_defaults_to_the_essential_stages(self):
        # The intervention arms are opt-in: both came back negative, and the
        # baseline is what the failure analysis runs on.
        src = (self.S / "run_all_h200.sh").read_text()
        self.assertIn('STAGES="${*:-0 1 2 3 4 8}"', src)


class TestRepairCoverageAndNoise(unittest.TestCase):
    """Guards for the two gaps found on 2026-09-16."""

    S = Path(__file__).resolve().parents[1] / "scripts"

    def test_pipeline_reports_missing_repairs(self):
        # Stage 4 exported work and stage 8 checked existing work; nothing
        # reported the difference, so 21 nut_assembly programs were skipped
        # in silence for a whole phase.
        src = (self.S / "run_all_h200.sh").read_text()
        self.assertIn("repair_status.py", src)
        self.assertLess(src.index("export_failures.py"),
                        src.index("repair_status.py"))

    def test_pipeline_exports_the_oracle_before_repairs(self):
        src = (self.S / "run_all_h200.sh").read_text()
        self.assertIn("export_oracle_code.py", src)

    def test_verify_redraws_inside_the_noise_band(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "vr", self.S / "verify_repairs.py")
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        # measured: cube_stack flips one scene with identical code and seeds
        self.assertEqual(m.TASK_NOISE.get("cube_stack"), 1)
        self.assertGreaterEqual(m.NOISE_REDRAWS, 1)
        # the band must NOT lower the acceptance bar
        self.assertEqual(m.TASK_CEILING["cube_stack"], 30)
        src = (self.S / "verify_repairs.py").read_text()
        self.assertIn("NOT a lowered acceptance bar", src)

    def test_scan_checks_all_episodes_for_determinism(self):
        # Four of the five multi-program cells were at ceiling, so a check
        # that inspects only failing episodes cannot see them.
        src = (self.S / "scan_corpus.py").read_text()
        self.assertIn("ALL episodes, not just failing", src)

    def test_gpu_util_auto_is_quantised(self):
        # An unrounded pool varies between runs, and a different pool changes
        # scheduling, which can flip a token.
        src = (self.S / "run_sweep.sh").read_text()
        self.assertIn("quantised", src.lower())
        self.assertIn("int(u * 10) / 10", src)


class TestOracleExport(unittest.TestCase):
    """The oracle code lives on the env CLASS, not in the oracle YAML.

    `use_oracle_code: true` in the YAML means "run env.oracle_code"; the
    first version of export_oracle_code.py looked for a YAML field that no
    task has, and failed on the very task it was written for.
    """

    S = Path(__file__).resolve().parents[1] / "scripts"

    def _mod(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "eoc", self.S / "export_oracle_code.py")
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m

    def test_resolves_target_class_attribute_without_importing(self):
        import tempfile
        m = self._mod()
        with tempfile.TemporaryDirectory() as d:
            capx = Path(d)
            pkg = capx / "capx" / "envs" / "tasks" / "franka"
            pkg.mkdir(parents=True)
            (pkg / "franka_nut_assembly.py").write_text(
                'import this_module_does_not_exist\n'
                'PROMPT = "p"\n'
                'ORACLE_CODE = """\n# Code block 1\nclose_gripper()\n"""\n'
                'class FrankaNutAssemblyCodeEnv:\n'
                '    prompt = PROMPT\n'
                '    oracle_code = ORACLE_CODE\n')
            y = capx / "env_configs" / "human_oracle_code" / "robosuite"
            y.mkdir(parents=True)
            (y / "franka_robosuite_nut_assembly_privileged_oracle.yaml").write_text(
                "env:\n  _target_: capx.envs.tasks.franka.franka_nut_assembly."
                "FrankaNutAssemblyCodeEnv\n  cfg:\n    privileged: true\n"
                "use_oracle_code: true\n")
            # the unresolvable import proves nothing was imported, and the
            # marker proves the code goes through strip_markers like every
            # other reference
            self.assertEqual(m.extract(capx, "nut_assembly"), "close_gripper()")

    def test_yaml_override_wins_and_empty_fails_loudly(self):
        import tempfile
        m = self._mod()
        with tempfile.TemporaryDirectory() as d:
            capx = Path(d)
            y = capx / "env_configs" / "human_oracle_code" / "robosuite"
            y.mkdir(parents=True)
            p = y / "franka_robosuite_nut_assembly_privileged_oracle.yaml"
            p.write_text("env:\n  _target_: a.b.C\n  cfg:\n"
                         "    oracle_code: 'open_gripper()'\n")
            self.assertEqual(m.extract(capx, "nut_assembly"), "open_gripper()")
            pkg = capx / "a"
            pkg.mkdir()
            (pkg / "b.py").write_text('class C:\n    oracle_code = ""\n')
            p.write_text("env:\n  _target_: a.b.C\n")
            with self.assertRaises(KeyError):
                m.extract(capx, "nut_assembly")

    def test_committed_reference_is_the_checkout_oracle(self):
        # references.json must carry what the seeded oracle store scored 30/30
        # with, in the same {model, tier, successes, n, code} shape as the
        # three model-derived references.
        refs = json.loads((self.S.parent / "results" / "references.json").read_text())
        self.assertIn("nut_assembly", refs)
        r = refs["nut_assembly"]
        self.assertEqual(set(r), {"model", "tier", "successes", "n", "code"})
        self.assertEqual(r["model"], "oracle")
        self.assertIn("sample_grasp_pose(\"extruded handle", r["code"])
        self.assertIn("T_handle_to_center", r["code"])
        capx = Path(os.environ.get("CAPX_DIR", "/data/cap-x"))
        if (capx / "capx").exists():
            self.assertEqual(self._mod().extract(capx, "nut_assembly"), r["code"])


class TestGate0ParsesItsOwnRun(unittest.TestCase):
    """Gate 0 committed spill_wipe's trials as every task's oracle cell.

    Found 2026-09-16 when the nut_assembly oracle, recorded 30/30 in
    capx-oracle-seeded, scored 17/30 through the verifier: the commit step
    globbed oracle/* and kept whichever task's directory parsed last.
    CaP-X's own summary of the seeded draw said 16.
    """

    S = Path(__file__).resolve().parents[1] / "scripts"

    def test_parses_only_the_task_directory(self):
        src = (self.S / "run_gate0_oracle.sh").read_text()
        self.assertNotIn('glob("oracle/*")', src)
        self.assertIn('out.parent / "oracle" / task', src)

    def test_starts_from_an_empty_run_directory(self):
        src = (self.S / "run_gate0_oracle.sh").read_text()
        self.assertIn('rm -rf "$OUT/$task" "$OUT/oracle/$task"', src)

    def test_cross_checks_before_committing(self):
        src = (self.S / "run_gate0_oracle.sh").read_text()
        self.assertIn("cross_check(run_dir)", src)
        self.assertLess(src.index("cross_check(run_dir)"), src.index("write_cell("))

    def test_verifier_knows_the_seeded_programs(self):
        # verify_repairs.py read results/failures.json only, which holds none
        # of the 21 nut_assembly programs: the first repair saved for that
        # task would have crashed stage 8 with a KeyError on its pid.
        src = (self.S / "verify_repairs.py").read_text()
        self.assertIn('"failures_seeded.json"', src)
        self.assertIn("match no exported program", src)


class TestLeakAB(unittest.TestCase):
    """The IK-seed A/B, and the ways it could quietly stop being an A/B.

    Its failure mode is a silent no-op: an arm that does not actually differ
    from the reference arm produces a clean null that reads as an answer. Every
    assertion here pins one way the arms could collapse into each other.
    """

    ROOT = Path(__file__).resolve().parents[1]

    def _mod(self):
        import importlib.util
        import sys
        spec = importlib.util.spec_from_file_location(
            "run_leak_ab", self.ROOT / "scripts" / "run_leak_ab.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["run_leak_ab"] = mod
        spec.loader.exec_module(mod)
        return mod

    def test_the_arms_differ_in_exactly_one_thing_each(self):
        m = self._mod()
        # A2 is the replicate control: it MUST be identical to A, because a
        # control that differs from what it replicates measures nothing, and
        # an A-vs-B difference is only readable against it.
        self.assertEqual(m.ARMS["A"]["workers"], m.ARMS["A2"]["workers"])
        self.assertEqual(m.ARMS["A"]["prepend"], m.ARMS["A2"]["prepend"])
        # B differs from A in worker count and nothing else: that is the one
        # variable its result gets attributed to.
        self.assertEqual(m.ARMS["B"]["workers"], 1)
        self.assertNotEqual(m.ARMS["A"]["workers"], m.ARMS["B"]["workers"])
        self.assertEqual(m.ARMS["A"]["prepend"], m.ARMS["B"]["prepend"])
        # C differs from A in the prepended line and nothing else.
        self.assertEqual(m.ARMS["A"]["workers"], m.ARMS["C"]["workers"])
        self.assertEqual(m.ARMS["C"]["prepend"], m.HOME_CALL)

    def test_arm_c_only_where_the_api_binds_the_primitive(self):
        from opencap.editclass import PROVIDED
        m = self._mod()
        # The cube_stack API has no home function and never resets cfg.
        # Prepending a call to one would NameError every trial and read as a
        # catastrophic result rather than a missing primitive.
        self.assertEqual(m.ARM_C_TASKS, ("nut_assembly",))
        self.assertIn(m.HOME_CALL.replace("()", ""), PROVIDED)

    def test_it_reuses_the_verifier_execution_path(self):
        src = (self.ROOT / "scripts" / "run_leak_ab.py").read_text()
        # run_one clears the stale output dir, detects a dead IK server and
        # cross-checks the parse against CaP-X's own summary. A second copy of
        # that would drift, exactly as the four per-arm sweep scripts did.
        self.assertIn("from verify_repairs import run_one", src)
        self.assertNotIn("def run_one", src)
        self.assertNotIn("launch.py", src)

    def test_each_arm_gets_its_own_scratch(self):
        src = (self.ROOT / "scripts" / "run_leak_ab.py").read_text()
        # Two arms sharing an output directory made the first A/B's parse read
        # a mixture of both arms' trial dirs.
        self.assertIn("Path(a.scratch_root) / arm", src)

    def test_per_trial_outcomes_are_recorded(self):
        src = (self.ROOT / "scripts" / "run_leak_ab.py").read_text()
        # Cell totals cannot tell "the same scenes fail in both arms" from
        # "different scenes fail in each", and those are opposite conclusions
        # about the leak. Pairing needs the trial index.
        self.assertIn('rec["trials"]', src)
        self.assertIn("parse_trials", src)

    def test_the_shell_driver_sources_the_environment(self):
        src = (self.ROOT / "scripts" / "run_leak_ab.sh").read_text()
        # phase5_launch.sh did not, so its outer-shell stages ran with no
        # HF_TOKEN and no CAPX_DIR.
        self.assertIn("source scripts/env_server.sh", src)
        # and a run's exit status must not be eaten by a pipe
        self.assertIn("PIPESTATUS", src)

    def test_report_is_paired_and_has_a_yardstick(self):
        src = (self.ROOT / "scripts" / "leak_report.py").read_text()
        self.assertIn("mcnemar", src)
        self.assertIn("p_value", src)
        # A vs A2 must be computed and printed before A vs B: the leak number
        # is not readable without the replicate it is judged against.
        self.assertLess(src.index('"A", "A2"'), src.index('"A", "B"'))
        # a ratio against a zero baseline is a division by zero with a decimal
        # point on it
        self.assertIn("base > 0", src)

    def test_the_human_reference_is_included_as_a_positive_control(self):
        m = self._mod()
        # Every other program in this A/B is a repair at 30/30, so a null
        # result cannot be told from a measurement that does not work. The
        # nut_assembly reference is the one program whose response to this
        # exact intervention is already known: 16/30 stock, 30/30 with the
        # start-home prepended.
        progs, meta = m.load_references(("nut_assembly",))
        self.assertIn("reference_nut_assembly", progs)
        self.assertTrue(progs["reference_nut_assembly"]["code"])
        self.assertEqual(meta["reference_nut_assembly"]["known"], "16/30")
        # and it must be a task arm C can actually run
        self.assertIn(meta["reference_nut_assembly"]["task"], m.ARM_C_TASKS)

    def test_the_reference_is_not_silently_dropped(self):
        src = (self.ROOT / "scripts" / "run_leak_ab.py").read_text()
        # A missing references.json must say what it costs, not vanish.
        self.assertIn("will not be interpretable", src)
        self.assertIn("--no-reference", src)

    def test_register_rederives_the_leak_result(self):
        import importlib.util
        import sys
        spec = importlib.util.spec_from_file_location(
            "cr_leak", self.ROOT / "scripts" / "claims_register.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["cr_leak"] = mod
        spec.loader.exec_module(mod)
        rows = mod.leak_rows()
        self.assertTrue(rows)
        blob = " ".join(f"{c} {s} {d}" for c, s, d in rows)
        # The status must be derived from the records, not typed in. If the
        # file is present the claim is settled; if it is not, it says PENDING
        # and does not pretend.
        if mod.LEAK.exists():
            self.assertIn("REFUTED", blob)
            # and the null must never be printed without the control that
            # makes it readable -- a null from a broken instrument looks
            # identical (artifact #10).
            self.assertIn("positive control", blob)
        else:
            self.assertIn("PENDING", blob)

    def test_the_leak_null_is_never_reported_without_its_control(self):
        src = (self.ROOT / "scripts" / "claims_register.py").read_text()
        # If the reference is missing from the records, the row must say the
        # null is not interpretable rather than report a clean nothing.
        self.assertIn("not interpretable", src)
        self.assertIn("reference_nut_assembly", src)


class TestSampledArmGuards(unittest.TestCase):
    """A sampled cell and a greedy cell share a shard path.

    (wave, model, tier, task) is the whole path, so writing one over the other
    REPLACES it in place and leaves a corpus mixing two decode conditions with
    nothing downstream able to tell them apart. That is the same shape as the
    shared-scratch A/B and the shared-output-dir arms.
    """

    ROOT = Path(__file__).resolve().parents[1]

    def test_greedy_process_refuses_a_sampled_store(self):
        from opencap.store import _check_decode_matches_store
        with self.assertRaises(PermissionError):
            _check_decode_matches_store("savoji/capx-nut-sampled-k1")
        # and the matching case is allowed, so the guard is not just "always no"
        _check_decode_matches_store("savoji/capx-ab-on-seeded")

    def test_sampled_process_refuses_the_greedy_baseline(self):
        import importlib
        import os
        from opencap import config, store
        os.environ["OPENCAP_TEMPERATURE"] = "0.7"
        try:
            importlib.reload(config)
            importlib.reload(store)
            self.assertTrue(config.is_sampled())
            with self.assertRaises(PermissionError):
                store._check_decode_matches_store("savoji/capx-ab-on-seeded")
            store._check_decode_matches_store("savoji/capx-nut-sampled-k1")
        finally:
            del os.environ["OPENCAP_TEMPERATURE"]
            importlib.reload(config)
            importlib.reload(store)
        self.assertFalse(config.is_sampled())

    def test_the_arm_script_names_a_sampled_store(self):
        src = (self.ROOT / "scripts" / "run_sampled_arm.sh").read_text()
        self.assertIn("sampled", src)
        self.assertIn("OPENCAP_SAMPLE_INDEX", src)
        self.assertIn("source scripts/env_server.sh", src)
        # Scoped to one task by DEFAULT: 21 cells per draw, not 84. It is an
        # argument now (A4 needs another task and tier), so pin the default
        # and pin that a non-default scope cannot share the store.
        self.assertIn('OPENCAP_SAMPLED_TASK:-nut_assembly', src)
        self.assertIn('OPENCAP_SAMPLED_TIER:-S1', src)
        self.assertIn('capx-nut-sampled${SCOPE}-k${K}', src)

    def test_each_draw_is_a_distinct_reproducible_stream(self):
        src = (self.ROOT / "scripts" / "vllm_shim.py").read_text()
        # Without a per-draw seed, k re-runs of draw 3 are three different
        # experiments and "resume" silently means "collect something else".
        self.assertIn("_SAMPLE_SEED", src)
        self.assertIn("OPENCAP_SAMPLE_INDEX", src)


class TestScenePrObe(unittest.TestCase):
    """The scene-13 probe: its controls are the whole argument."""

    ROOT = Path(__file__).resolve().parents[1]

    def test_controls_are_on_by_default(self):
        src = (self.ROOT / "scripts" / "run_scene_probe.py").read_text()
        # If untagged programs flip the scene at the same rate, "the repair
        # fixed scene 13" says nothing about the repair.
        self.assertIn('"--controls"', src)
        self.assertIn("default=3", src)
        self.assertIn("cannot be interpreted", src)

    def test_it_reuses_the_verifier_execution_path(self):
        src = (self.ROOT / "scripts" / "run_scene_probe.py").read_text()
        self.assertIn("from verify_repairs import run_one", src)
        self.assertNotIn("launch.py", src)

    def test_control_set_is_deterministic(self):
        src = (self.ROOT / "scripts" / "run_scene_probe.py").read_text()
        # A control set that changes between runs is not a control.
        self.assertIn("sorted(", src)

    def test_the_report_carries_intervals(self):
        src = (self.ROOT / "scripts" / "scene_probe_report.py").read_text()
        # At N=8, 6/8 and 8/8 are not distinguishable and bare point estimates
        # invite reading a difference that is not there.
        self.assertIn("wilson", src)

    def test_the_shell_driver_sources_the_environment(self):
        src = (self.ROOT / "scripts" / "run_scene_probe.sh").read_text()
        self.assertIn("source scripts/env_server.sh", src)
        self.assertIn("PIPESTATUS", src)


class TestSweepClearsItsOutputDir(unittest.TestCase):
    """The sweep must start each cell from an empty output directory.

    verify_repairs.run_one has had this guard since stage 8 of the rebuild;
    run_cell.py did not, and in Phase 7 two interrupted cells left trial dirs
    behind -- one of them WITH an aaa_done_flag, so is_complete() returned
    true on the wreckage. CaP-X names trial dirs by outcome, so a re-run into
    a populated dir yields a parse over a mixture of two runs.
    """

    ROOT = Path(__file__).resolve().parent.parent

    def test_run_cell_removes_a_stale_output_dir(self):
        src = (self.ROOT / "scripts" / "run_cell.py").read_text()
        self.assertIn("rmtree", src)
        # Both spellings of the path: CaP-X rewrites /x/run -> /x/<slug>/run,
        # so clearing only the requested path leaves the real one behind.
        self.assertIn("resolved_output_dir(requested, spec.repo)", src)
        head = src[:src.index("plan = LaunchPlan(")]
        self.assertIn("rmtree", head,
                      "the clean must happen BEFORE the launch, not after")

    def test_both_runners_agree_on_the_rule(self):
        for name in ("run_cell.py", "verify_repairs.py"):
            src = (self.ROOT / "scripts" / name).read_text()
            self.assertIn("rmtree", src, f"{name} lost its empty-dir guard")


class TestTruncationAudit(unittest.TestCase):
    """Candidate artifact #15 needs a generator, like every other claim.

    `results/tags_seeded.json` was once produced by a script that lived
    nowhere, which is precisely how a tagger bug survived into a published
    table. The truncation rates are now a published number too.
    """

    ROOT = Path(__file__).resolve().parent.parent

    def test_the_audit_exists_and_names_the_cap(self):
        p = self.ROOT / "scripts" / "truncation_audit.py"
        self.assertTrue(p.exists(), "the audit has no generator")
        src = p.read_text()
        self.assertIn("2048", src)
        self.assertIn("wilson", src, "rates without intervals invite the "
                                     "same error four findings died of")

    def test_it_cuts_by_band(self):
        # The per-model table is the evidence; the band cut is the finding,
        # because the scale claim is measured across exactly those bands.
        src = (self.ROOT / "scripts" / "truncation_audit.py").read_text()
        self.assertIn("band", src)
        self.assertIn("params_b", src)

    def test_it_can_window_by_time(self):
        # The shim's stats files are cumulative across phases, so a rate for
        # one run is only obtainable with a cutoff.
        src = (self.ROOT / "scripts" / "truncation_audit.py").read_text()
        self.assertIn("--since", src)


class TestBigCapArmGuards(unittest.TestCase):
    """The raised-cap arm, and the ways it could contaminate the corpus."""

    ROOT = Path(__file__).resolve().parents[1]

    def test_the_default_cap_is_frozen(self):
        from opencap import config
        self.assertEqual(config.DEFAULT_MAX_NEW_TOKENS, 2_048)
        self.assertEqual(config.MAX_NEW_TOKENS, 2_048)
        self.assertFalse(config.is_bigcap())
        self.assertIn("bigcap", config.context_manifest())

    def test_standard_cap_refuses_a_bigcap_store(self):
        from opencap.store import _check_cap_matches_store
        with self.assertRaises(PermissionError):
            _check_cap_matches_store("savoji/capx-lowtrunc-bigcap")
        _check_cap_matches_store("savoji/capx-ab-on-seeded")

    def test_bigcap_refuses_the_standard_baseline(self):
        import importlib
        import os
        from opencap import config, store
        os.environ["OPENCAP_MAX_NEW_TOKENS"] = "8192"
        try:
            importlib.reload(config)
            importlib.reload(store)
            self.assertTrue(config.is_bigcap())
            with self.assertRaises(PermissionError):
                store._check_cap_matches_store("savoji/capx-ab-on-seeded")
            store._check_cap_matches_store("savoji/capx-lowtrunc-bigcap")
        finally:
            del os.environ["OPENCAP_MAX_NEW_TOKENS"]
            importlib.reload(config)
            importlib.reload(store)
        self.assertFalse(config.is_bigcap())

    def test_the_arm_refuses_a_non_bigcap_store_early(self):
        src = (self.ROOT / "scripts" / "run_bigcap_arm.sh").read_text()
        self.assertIn("REFUSING", src)
        self.assertIn("source scripts/env_server.sh", src)
        # the three models above 20% truncation, and only those
        self.assertIn("qwen35-0.8b qwen35-2b qwen3vl-2b", src)


class TestTierGrid(unittest.TestCase):
    """The scaffold axis, and the way it could repeat artifact #13.

    The seeding patch reached one simulator in four and the pairing probe
    checked the one task where it worked. Both guards added afterwards are
    S1-scoped. A reduced-API config binding an unpatched simulator would put
    random scenes back into a fresh set of cells, and it would look exactly as
    plausible as the first time.
    """

    ROOT = Path(__file__).resolve().parents[1]

    def test_the_grid_tiers_have_configs_for_every_task(self):
        from opencap.capx_bridge import config_path, has_config
        for tier in ("S1", "S3", "S4"):
            for task in ("cube_lifting", "cube_stack", "spill_wipe", "nut_assembly"):
                self.assertTrue(has_config(task, tier), f"{tier} {task}")
                self.assertTrue(config_path(task, tier).endswith(".yaml"))

    def test_s3_and_s4_are_distinct_conditions(self):
        from opencap.capx_bridge import config_path
        s1 = config_path("cube_stack", "S1")
        s3 = config_path("cube_stack", "S3")
        s4 = config_path("cube_stack", "S4")
        self.assertEqual(len({s1, s3, s4}), 3)
        self.assertIn("reduced_api", s3)
        self.assertIn("exampleless", s4)

    def test_the_grid_refuses_to_collect_before_the_simulator_check(self):
        src = (self.ROOT / "scripts" / "run_tier_grid.sh").read_text()
        self.assertIn("verify_tier_simulators.py", src)
        self.assertIn("--strict", src)
        self.assertIn("REFUSING TO COLLECT", src)
        # the check must come BEFORE any collection
        self.assertLess(src.index("verify_tier_simulators"), src.index("run_arm.sh"))

    def test_an_unresolved_binding_is_not_a_pass(self):
        src = (self.ROOT / "scripts" / "verify_tier_simulators.py").read_text()
        # An unchecked binding is how #13 survived; it must not read as clean.
        self.assertIn("NOT a pass", src)

    def test_run_arm_accepts_a_tier(self):
        src = (self.ROOT / "scripts" / "run_arm.sh").read_text()
        self.assertIn('PILOT_TIERS="${PILOT_TIERS:-S1}"', src)

    def test_tier_report_will_not_imply_an_unmeasured_mix(self):
        src = (self.ROOT / "scripts" / "tier_report.py").read_text()
        # A tier with no tagging has an UNKNOWN mix, not an unchanged one.
        self.assertIn("NOT TAGGED", src)
        self.assertIn("it is unmeasured", src)


class TestFinishReasonIsRecorded(unittest.TestCase):
    """Artifact #15's permanent fix.

    `truncation_fired` is ours and describes the PROMPT. Nothing recorded why
    GENERATION stopped, and the shim hardcoded finish_reason="stop" on every
    response -- a field that always says the same thing is a check that cannot
    fail, which is the shape of all fifteen artifacts here.
    """

    ROOT = Path(__file__).resolve().parents[1]

    def test_the_episode_carries_it(self):
        from opencap.schema import Episode, RESULT_COLUMNS
        self.assertIn("finish_reason", RESULT_COLUMNS)
        self.assertIn("completion_tokens", RESULT_COLUMNS)
        e = Episode(episode_id="x", model="m", tier="S1", task="cube_stack",
                    state_id=0, wave=1)
        # Absence means UNKNOWN, never "stop": most of the corpus predates this.
        self.assertEqual(e.finish_reason, "")

    def test_columns_were_appended_not_reordered(self):
        from opencap.schema import RESULT_COLUMNS
        # The parquet column order is declared append-only; inserting in the
        # middle silently rewrites the meaning of every existing file.
        self.assertEqual(RESULT_COLUMNS[-2:], ("finish_reason", "completion_tokens"))
        self.assertEqual(RESULT_COLUMNS[0], "episode_id")

    def test_the_shim_does_not_hardcode_stop(self):
        src = (self.ROOT / "scripts" / "vllm_shim.py").read_text()
        self.assertNotIn('"finish_reason": "stop",', src)
        self.assertIn('getattr(out.outputs[0], "finish_reason", None)', src)
        # and the fallback must be derived, not assumed
        self.assertIn('"length" if n_out >= j.get("max_tokens"', src)


class TestLlmTimeoutPatch(unittest.TestCase):
    """The read-timeout raise must behave like patch_seeding.py.

    A dependency patch that a fresh checkout does not get, or that reports
    success when it changed nothing, is a patch that silently stops being
    true. Artifact #13 was exactly that shape.
    """

    ROOT = Path(__file__).resolve().parent.parent
    STOCK = (
        "import os\n\n"
        "def a():\n"
        "    r = requests.post(\n"
        "        url, headers=h, data=d, timeout=200\n"
        "    )\n\n"
        "def b():\n"
        "    r = requests.post(\n"
        "        url, headers=h, data=d, timeout=200\n"
        "    )\n\n"
        "def c():\n"
        "    with requests.post(\n"
        "        url,\n"
        "        data=d,\n"
        "        timeout=200,\n"
        "        stream=True,\n"
        "    ) as r:\n"
        "        pass\n"
    )

    def _mod(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "patch_llm_timeout", self.ROOT / "scripts" / "patch_llm_timeout.py")
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m

    def test_it_patches_all_three_shapes(self):
        # Two sites close the argument list; the streaming one is followed by
        # another kwarg. A pattern anchored on the closing paren matches two
        # of three and reports success -- that happened while writing this.
        m = self._mod()
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "client.py"
            p.write_text(self.STOCK)
            status, n = m.patch_file(p, 600)
            self.assertEqual(n, 3, status)
            self.assertNotIn("timeout=200", p.read_text())
            self.assertEqual(p.read_text().count("timeout=600"), 3)

    def test_it_is_idempotent_to_the_byte(self):
        m = self._mod()
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "client.py"
            p.write_text(self.STOCK)
            m.patch_file(p, 600)
            once = p.read_text()
            status, n = m.patch_file(p, 600)
            self.assertEqual(n, 0, status)
            self.assertEqual(p.read_text(), once, "re-run rewrote the file")

    def test_it_reports_pattern_not_found_rather_than_success(self):
        m = self._mod()
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "client.py"
            p.write_text("x = 1\n")
            status, n = m.patch_file(p, 600)
            self.assertIn("NOT FOUND", status)
            self.assertEqual(n, 0)

    def test_it_can_re_raise_to_a_new_value(self):
        m = self._mod()
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "client.py"
            p.write_text(self.STOCK)
            m.patch_file(p, 600)
            status, n = m.patch_file(p, 900)
            self.assertEqual(n, 3, status)
            self.assertEqual(p.read_text().count("timeout=900"), 3)

    def test_it_stays_under_capx_per_trial_timeout(self):
        # CaP-X fails a trial at 1000s. A read timeout above that would turn a
        # dead request into a hung run instead of a failed trial.
        m = self._mod()
        self.assertLess(m.DEFAULT_TIMEOUT, 1000)
        self.assertGreater(m.DEFAULT_TIMEOUT, m.STOCK_TIMEOUT)


class TestTierSimulatorResolver(unittest.TestCase):
    """The tier precondition must resolve real bindings AND still be able to fail.

    Its first version could not resolve anything -- the binding is a registry
    key, not the imported name -- so all twelve rows came back UNRESOLVED,
    including the four S1 tasks collected correctly for weeks. A checker that
    reports the same thing for "broken" and "cannot tell" is the shape that
    let artifact #13 survive.
    """

    ROOT = Path(__file__).resolve().parent.parent
    INIT = (
        "from .robosuite_cube_lift import FrankaRobosuiteCubeLiftLowLevel\n"
        "from .robosuite_nut_assembly import FrankaRobosuiteNutAssembly\n"
        "from .robosuite_nut_assembly import FrankaRobosuiteNutAssemblyVisual\n"
        "register_env('franka_robosuite_cube_lift_low_level',"
        " FrankaRobosuiteCubeLiftLowLevel)\n"
        "register_env('franka_robosuite_nut_assembly_low_level_visual',"
        " FrankaRobosuiteNutAssemblyVisual)\n"
    )

    def _mod(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "verify_tier_simulators",
            self.ROOT / "scripts" / "verify_tier_simulators.py")
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m

    def _tree(self, d, marker=True):
        sim = Path(d) / "capx" / "envs" / "simulators"
        sim.mkdir(parents=True)
        (sim / "__init__.py").write_text(self.INIT)
        body = "x = 1\n" + (self._mod().MARKER + "\n" if marker else "")
        for f in ("robosuite_cube_lift.py", "robosuite_nut_assembly.py"):
            (sim / f).write_text(body)
        return Path(d)

    def test_it_follows_the_registry_two_hops(self):
        m = self._mod()
        with tempfile.TemporaryDirectory() as d:
            root = self._tree(d)
            self.assertEqual(
                m.module_for("franka_robosuite_cube_lift_low_level", root),
                "robosuite_cube_lift.py")

    def test_a_subclass_binding_resolves_to_its_own_module(self):
        # S3/S4 nut_assembly binds the *_visual key, a different class in the
        # same file. Resolving it is what says the patched reset() applies.
        m = self._mod()
        with tempfile.TemporaryDirectory() as d:
            root = self._tree(d)
            self.assertEqual(
                m.module_for("franka_robosuite_nut_assembly_low_level_visual",
                             root),
                "robosuite_nut_assembly.py")

    def test_an_unknown_binding_is_unresolved_not_ok(self):
        m = self._mod()
        with tempfile.TemporaryDirectory() as d:
            root = self._tree(d)
            self.assertIsNone(m.module_for("franka_something_nobody_wrote", root))

    def test_listed_but_unpatched_is_not_a_pass(self):
        # TARGETS membership is intent; the marker is fact. A fresh checkout
        # or a reverted hunk leaves a module listed and unseeded.
        m = self._mod()
        with tempfile.TemporaryDirectory() as d:
            root = self._tree(d, marker=False)
            self.assertFalse(m.patch_applied("robosuite_cube_lift.py", root))
        with tempfile.TemporaryDirectory() as d:
            root = self._tree(d, marker=True)
            self.assertTrue(m.patch_applied("robosuite_cube_lift.py", root))

    def test_the_marker_is_not_duplicated(self):
        # Two copies of a marker drift; the one that drifts is the one nobody
        # re-reads.
        src = (self.ROOT / "scripts" / "verify_tier_simulators.py").read_text()
        self.assertIn("_patch_seeding().MARKER", src)


class TestSweepRefusesPartialServerLoss(unittest.TestCase):
    """A dead service must not be charged to the model, even partially.

    verify_repairs.run_one counts the affected trials; run_cell.py fired only
    when EVERY trial failed, so a cell with ten dead trials and twenty real
    ones committed silently. That matters most at S3/S4, where the reduced
    API is perception-backed and whether a trial reaches a vision server
    depends on whether the generated program happens to call one.
    """

    ROOT = Path(__file__).resolve().parent.parent

    def test_the_sweep_counts_trials_rather_than_requiring_all(self):
        src = (self.ROOT / "scripts" / "run_cell.py").read_text()
        self.assertIn("n_unreachable", src)
        self.assertNotIn("if _t and all(not t.success for t in _t):", src,
                         "the all-or-nothing form leaves the partial case "
                         "silent")

    def test_both_runners_refuse_on_any_unreachable_trial(self):
        for name in ("run_cell.py", "verify_repairs.py"):
            src = (self.ROOT / "scripts" / name).read_text()
            self.assertIn("n_unreachable", src, f"{name} lost the guard")
            self.assertIn("Connection refused", src)
            self.assertIn("Max retries exceeded", src)


class TestApiMismatchAudit(unittest.TestCase):
    """Candidate artifact #16: the S3/S4 prompt names functions S3/S4 lack.

    The reason this needs a generator and a test, rather than a paragraph in
    STATUS: the number it produces decides which failing programs are held out
    of the tag mix, and the tag mix is the deliverable of the whole tier grid.
    A hold-out chosen by hand is a hold-out nobody can re-derive.
    """

    ROOT = Path(__file__).resolve().parents[1]

    def _src(self) -> str:
        p = self.ROOT / "scripts" / "api_mismatch_audit.py"
        self.assertTrue(p.exists(), "artifact #16 has no generator")
        return p.read_text()

    def test_it_reads_the_bindings_instead_of_hardcoding_them(self):
        # If upstream binds `segment_sam3`, this must stop reporting it. A
        # frozen list would keep flagging a defect that had been fixed --
        # the mirror image of the artifact it exists to catch.
        src = self._src()
        self.assertIn("control_reduced.py", src)
        self.assertIn("NOW BOUND UPSTREAM", src)

    def test_a_phantom_must_be_named_by_the_prompt(self):
        # The distinction the whole script turns on: a name the PROMPT gave
        # the model is a prompt defect, a name the model invented is a model
        # defect. Both raise NameError and they must not be pooled.
        src = self._src()
        self.assertIn("PROMPTED_PHANTOMS", src)
        for name in ("segment_sam3", "get_object_pose", "sample_grasp_pose"):
            self.assertIn(name, src)
        self.assertIn("ARE model defects", src,
                      "the script must say which NameErrors it is NOT excusing")

    def test_it_separates_killed_from_merely_steered(self):
        # 60 trials called get_object_pose and every one died several lines
        # earlier on an IndexError. Folding those into the causal count would
        # overstate the artifact; dropping them would hide contamination.
        src = self._src()
        self.assertIn("steered", src)

    def test_rates_carry_intervals(self):
        self.assertIn("wilson", self._src())

    def test_the_prompt_really_does_name_an_unbound_function(self):
        """The claim itself, checked against the checkout rather than trusted.

        Skips rather than fails when the checkout is absent: this suite runs
        on machines that have no cap-x, and a test that cannot see its subject
        must say so, not pass.
        """
        capx = Path("/data/cap-x")
        api = capx / "capx" / "integrations" / "franka" / "control_reduced.py"
        if not api.exists():
            self.skipTest("no cap-x checkout here")
        bound = set(re.findall(r"^\s*(?:async\s+)?def\s+(\w+)",
                               api.read_text(errors="ignore"), re.M))
        self.assertIn("segment_sam3_text_prompt", bound,
                      "sanity: the real primitive should be bound")
        self.assertNotIn("segment_sam3", bound,
                         "if this fails the docstring example is now correct "
                         "and artifact #16's first half is fixed upstream")


class TestServiceLossGuardAndAudit(unittest.TestCase):
    """A live service returning 500 is a lost service too.

    2026-09-29: qwen3-1.7b S3 cube_lifting committed with 4 of 30 trials dead
    on "Failed to communicate with GraspNet service ... 500 Server Error",
    which the server log attributes to CUDA OOM with a neighbour holding 77 GB
    of the card. Neither "Connection refused" nor "Max retries exceeded"
    appears in that text, so the guard passed it and four environment failures
    were scored as model defects. Second time this guard has been one string
    short; hence a test on the strings themselves.
    """

    ROOT = Path(__file__).resolve().parents[1]

    def test_the_guard_refuses_on_a_live_service_error(self):
        src = (self.ROOT / "scripts" / "run_cell.py").read_text()
        self.assertIn("Failed to communicate with", src,
                      "a 500 from a live service must also refuse the cell")
        self.assertIn("Connection refused", src)
        self.assertIn("Max retries exceeded", src)

    def test_the_guard_refuses_on_any_trial_not_a_majority(self):
        # The partial case is the dangerous one: it looks like a real result.
        src = (self.ROOT / "scripts" / "run_cell.py").read_text()
        self.assertIn("if n_unreachable:", src)
        self.assertNotIn("if n_unreachable > len(_t) // 2", src)

    def test_the_audit_and_the_guard_share_their_strings(self):
        """Drift here is how the same hole reopens a third time."""
        audit = self.ROOT / "scripts" / "service_loss_audit.py"
        self.assertTrue(audit.exists(), "the retrospective half is missing")
        guard_src = (self.ROOT / "scripts" / "run_cell.py").read_text()
        block = re.search(r"SERVICE_LOST\s*=\s*\((.*?)\)",
                          audit.read_text(), re.S)
        self.assertIsNotNone(block, "audit has no SERVICE_LOST tuple")
        strings = re.findall(r'"([^"]+)"', block.group(1))
        self.assertTrue(strings, "audit's SERVICE_LOST is empty")
        for s in strings:
            self.assertIn(s, guard_src,
                          f"audit looks for {s!r} but run_cell.py does not "
                          f"refuse on it -- new cells will commit dirty")

    def test_the_audit_separates_partial_from_total_loss(self):
        src = (self.ROOT / "scripts" / "service_loss_audit.py").read_text()
        self.assertIn("PARTIAL", src)
        self.assertIn("TOTAL", src)
        self.assertIn("wilson", src)

    def test_the_audit_warns_when_it_has_drifted_from_the_guard(self):
        # The script must notice the drift itself, not rely on this test
        # running on the machine where the collection happens.
        src = (self.ROOT / "scripts" / "service_loss_audit.py").read_text()
        self.assertIn("guard_is_in_step", src)


class TestVisionServerLogsSurviveRestart(unittest.TestCase):
    """A restart must not destroy the evidence of why the service died.

    2026-09-29: all three perception services were killed abruptly mid-run.
    GraspNet's log ended on a successful 200 OK -- no traceback, no shutdown
    line -- and the restart, which opened each log with `>`, truncated
    sam3_server.log before anyone read it. SAM3's cause of death is gone. A
    restart is precisely when someone is about to need that file.
    """

    ROOT = Path(__file__).resolve().parents[1]

    def test_the_launcher_appends(self):
        src = (self.ROOT / "scripts" / "start_vision_servers.sh").read_text()
        self.assertIn('>> "$LOG"', src)
        # A substring check cannot tell `>` from `>>` -- '>> "$LOG"' contains
        # '> "$LOG"'. Match a redirect whose '>' is NOT preceded by another.
        truncating = re.findall(r'(?<!>)>\s*"\$LOG"', src)
        self.assertEqual(truncating, [],
                         "a truncating redirect destroys the previous run's "
                         "log, which is the only record of how it died")

    def test_each_start_is_delimited_in_the_log(self):
        # Appending is only useful if you can tell the runs apart.
        src = (self.ROOT / "scripts" / "start_vision_servers.sh").read_text()
        self.assertIn("========", src)
        self.assertIn("date -u", src)


class TestBigRungPass(unittest.TestCase):
    """The 27B/32B rungs the grid could not start, and why they matter.

    With the grid's fixed 0.35 pool every engine above 14B failed, so the S3
    corpus contains no model above 14B -- while the rate table still prints a
    ">=12B" row for both tiers. S1's row is 24 cells spanning 12B to 32B;
    S3's is 5 cells of gemma3-12b and qwen3-14b. The band is absent from S3
    exactly where S1 scores highest, so the gap is overstated by construction.
    This pass exists to make that row comparable, not to tidy up.
    """

    ROOT = Path(__file__).resolve().parents[1]

    def _src(self) -> str:
        p = self.ROOT / "scripts" / "run_bigrung_pass.sh"
        self.assertTrue(p.exists())
        return p.read_text()

    def test_it_refuses_to_run_beside_the_grid(self):
        # Same card, same :8110, same flock. A second instance does not
        # queue -- it fails one engine at a time, and those failures look
        # exactly like the memory failures this pass exists to fix.
        src = self._src()
        self.assertIn("run_tier_grid", src)
        self.assertIn("REFUSING", src)
        self.assertIn("8110", src)

    def test_it_checks_the_card_before_spending_an_hour(self):
        src = self._src()
        self.assertIn("memory.free", src)
        self.assertIn("NEED_MIB", src)

    def test_it_keeps_the_perception_and_simulator_preconditions(self):
        src = self._src()
        self.assertIn("start_vision_servers.sh --check", src)
        self.assertIn("verify_tier_simulators.py", src)
        self.assertIn("--strict", src)

    def test_the_pool_is_fixed_never_auto(self):
        # `auto` reads free memory then starts seconds later; a neighbour
        # moving either way costs the whole rung. Both directions have.
        src = self._src()
        self.assertIn('OPENCAP_GPU_UTIL="$UTIL"', src)
        self.assertNotIn('OPENCAP_GPU_UTIL=auto', src)

    def test_the_gpu_is_read_before_env_server_overwrites_it(self):
        """env_server.sh sets OPENCAP_GPU=2 unconditionally.

        Reading it afterwards cannot distinguish an explicit 2 from a default,
        which is how the first version announced gpu=2 while the grid, the
        services and the free card were all on 5.
        """
        src = self._src()
        gpu_line = src.index('GPU="${OPENCAP_GPU:-5}"')
        source_line = src.index("source scripts/env_server.sh")
        self.assertLess(gpu_line, source_line,
                        "OPENCAP_GPU must be captured before env_server.sh runs")


class TestTrialAlternationIsOnTheRegister(unittest.TestCase):
    """The owner asked specifically that this observation not drop.

    qwen3-4b cube_stack scores 28/30 on every draw; trial 3 always fails, and
    the second failure alternates between trial 9 and trial 13 -- never both,
    never neither, across eight draws. 1/256 under independence. It was in
    STATUS but not in the structured register, and STATUS is long; the
    register's PENDING section is what survives a reader who is skimming.

    The test ran on 2026-10-13 (results/scene_probe_alt.json), so the row left
    PENDING in the same commit and is re-derived by alternation_rows() --
    verdict included, from whether trials 9 and 13 still vary at workers=1.
    """

    ROOT = Path(__file__).resolve().parents[1]

    def _mod(self):
        import importlib.util
        import sys
        spec = importlib.util.spec_from_file_location(
            "cr_alt", self.ROOT / "scripts" / "claims_register.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["cr_alt"] = mod
        spec.loader.exec_module(mod)
        return mod

    def _rows_for(self, recs):
        mod = self._mod()
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "scene_probe_alt.json"
            f.write_text(json.dumps(recs))
            mod.ALT_FILE = f
            return mod.alternation_rows()

    @staticmethod
    def _rec(workers, draw, failed, version="original"):
        return {"pid": "81ee85d0f933", "version": version, "draw": draw,
                "workers": workers, "ok": True, "failed_trials": failed}

    def test_it_left_pending_and_is_rederived(self):
        src = (self.ROOT / "scripts" / "claims_register.py").read_text()
        block = re.search(r"^PENDING = \[(.*?)^\]", src, re.S | re.M)
        self.assertIsNotNone(block, "PENDING list not found")
        self.assertNotIn("1/256", block.group(1))
        self.assertIn("rows += alternation_rows()", src)

    def test_a_one_sided_file_stays_pending(self):
        # Exactly the failure the resume key guards against: one condition
        # collected, the other skipped, and a verdict printed anyway.
        rows = self._rows_for([self._rec(4, d, [3, 9]) for d in range(4)])
        self.assertEqual(rows[0][1], "PENDING")

    def test_the_verdict_comes_from_the_records(self):
        w4 = [self._rec(4, 0, [3, 9]), self._rec(4, 1, [3, 13])]
        stops = self._rows_for(w4 + [self._rec(1, d, [3]) for d in range(2)])
        self.assertTrue(stops[0][1].startswith("STOPS AT WORKERS=1"))
        persists = self._rows_for(w4 + [self._rec(1, 0, [3, 9]),
                                        self._rec(1, 1, [3, 13])])
        self.assertTrue(persists[0][1].startswith("PERSISTS AT WORKERS=1"))
        # repair records are a different program version and never count
        repair_only = self._rows_for(w4 + [self._rec(1, 0, [13], "repair"),
                                           self._rec(1, 1, [], "repair")])
        self.assertEqual(repair_only[0][1], "PENDING")

    def test_the_qualifier_survives_in_every_verdict(self):
        w4 = [self._rec(4, 0, [3, 9]), self._rec(4, 1, [3, 13])]
        for w1 in ([self._rec(1, 0, [3])], [self._rec(1, 0, [3, 9]),
                                            self._rec(1, 1, [3, 13])]):
            claim, status, detail = self._rows_for(w4 + w1)[0]
            self.assertIn("n = 1", status)
            self.assertIn("OBSERVATION, not a claim", detail)

    def test_it_is_labelled_an_observation_not_a_claim(self):
        # n = 1 program. Promoting it would be the same error as the four
        # point estimates this project has already retracted.
        src = (self.ROOT / "scripts" / "claims_register.py").read_text()
        self.assertIn("OBSERVATION, not a claim", src)


class TestMolmoSkipList(unittest.TestCase):
    """Cells that cannot be collected while :8122 has no backend.

    At temperature 0 the program for a cell is identical on every attempt, so
    a cell that calls point_prompt_molmo refuses identically every time: 17 of
    17 reproduced on the 2026-09-30 resume at ~20 min each. Skipping them is a
    SCHEDULING decision -- run_cell.py already refuses them, so the corpus is
    unchanged either way.

    The danger this guards against is the opposite one: a skip list that
    outlives the outage becomes a permanent hole in the corpus while every run
    still looks clean.
    """

    ROOT = Path(__file__).resolve().parents[1]

    def _src(self, name: str) -> str:
        p = self.ROOT / "scripts" / name
        self.assertTrue(p.exists(), f"{name} missing")
        return p.read_text()

    def test_the_list_is_generated_from_evidence_not_handwritten(self):
        src = self._src("molmo_skiplist.py")
        self.assertIn("port=8122", src)
        self.assertIn("summary.txt", src,
                      "entries must come from trial summaries on disk")

    def test_it_is_reversible_three_ways(self):
        src = self._src("molmo_skiplist.py")
        self.assertIn("--clear", src)
        self.assertIn("OPENCAP_IGNORE_MOLMO_SKIP", src)
        self.assertIn("remove_when", src,
                      "the written file must say when to delete it")

    def test_run_cell_honours_the_override(self):
        src = self._src("run_cell.py")
        self.assertIn("OPENCAP_IGNORE_MOLMO_SKIP", src)

    def test_the_skip_is_checked_before_the_stale_dir_cleanup(self):
        """Order matters: the cleanup deletes the evidence the list is built from.

        Check after it and the list erases its own source the first time it
        is used, so a regeneration would come back empty and the cells would
        silently re-enter the rotation.
        """
        src = self._src("run_cell.py")
        skip_at = src.index("OPENCAP_IGNORE_MOLMO_SKIP")
        clean_at = src.index("removing stale output dir")
        self.assertLess(skip_at, clean_at,
                        "molmo skip must be checked before stale-dir cleanup")

    def test_a_missing_or_broken_list_does_not_block_a_run(self):
        # A skip list is an optimisation. It must never be able to stop
        # collection because of a JSON typo.
        src = self._src("run_cell.py")
        self.assertIn("except Exception", src)

    def test_load_returns_empty_when_absent(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "molmo_skiplist", self.ROOT / "scripts" / "molmo_skiplist.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        with tempfile.TemporaryDirectory() as d:
            mod.SKIPFILE = Path(d) / "nope.json"
            self.assertEqual(mod.load(), set())


class TestFourXXIsAModelDefect(unittest.TestCase):
    """A rejected request is not a lost service, and the guard used to confuse them.

    2026-10-01, once molmo was finally served, programs reached the
    molmo -> point -> segment_sam3_point_prompt stage and SAM3 began
    answering 422. The cause was the generated program:

        point_prompt = point_prompt_molmo(rgb, "brown spill")
        if "point_coords" in point_prompt:        # keyed by OBJECT NAME
            x, y = point_prompt["point_coords"]   # so never runs
        segment_sam3_point_prompt(rgb, (x, y))    # x, y still None

    SAM3 was healthy -- a hand-built request with real coords returns 200 and
    three masks. Refusing that cell discards a textbook api_misuse defect,
    which damages the tag mix exactly as much as admitting an environment
    failure would.
    """

    ROOT = Path(__file__).resolve().parents[1]

    def _src(self) -> str:
        return (self.ROOT / "scripts" / "run_cell.py").read_text()

    def test_5xx_still_refuses(self):
        src = self._src()
        for s in ("500 Server Error", "Connection refused", "Max retries exceeded"):
            self.assertIn(s, src)

    def test_the_bare_catch_all_is_gone(self):
        """'Failed to communicate with' must no longer refuse on its own.

        It matches both a dead service and a rejected payload; keeping it as
        an unconditional refusal is what discarded the model defect.
        """
        src = self._src()
        block = re.search(r"SERVICE_LOST\s*=\s*\((.*?)\)", src, re.S)
        self.assertIsNotNone(block)
        self.assertNotIn('"Failed to communicate with"', block.group(1),
                         "that string alone cannot distinguish 4xx from 5xx")

    def test_it_resolves_ambiguity_against_the_service_log(self):
        # The trial summary cannot carry the status: SAM3's post_with_retries
        # reports only "failed after N retries". The code is in the server log.
        src = self._src()
        self.assertIn("_service_rejected_us", src)
        self.assertIn("_server.log", src)
        self.assertIn("4\\d\\d", src)

    def test_a_sick_service_still_wins(self):
        """A service that was in trouble refuses the cell whatever else the
        window holds, and that check runs FIRST.

        The first version of this guard let four CUDA-OOM 500s into the corpus
        as model defects. That direction of error is worse than the opposite
        one, so sickness is tested before any bad-input pattern.
        """
        src = self._src()
        self.assertIn("CUDA out of memory", src)
        self.assertIn("SERVICE_SICK", src)
        body = src[src.index("def _classify_5xx"):src.index("def _service_rejected_us")]
        self.assertLess(body.index("SERVICE_SICK"), body.index("BAD_INPUT"),
                        "sickness must be checked before bad input")
        # and a window with no 2xx in it is a window the service may have been
        # dead for, so it cannot support a model verdict
        self.assertIn('" 2\\d\\d ', body)

    def test_a_bad_input_5xx_is_a_model_defect_but_only_in_this_cells_window(self):
        """Owner's decision 2026-10-27, with the hazard that makes it work.

        The services raise 5xx for a malformed payload -- 2026-10-21, a
        program passed a box where a point belongs and SAM3 answered 'Point
        prompt inference failed: Sizes of tensors must match'. That is a model
        defect, and refusing those cells loses exactly the failing programs.

        But the old guard read a FIXED 200 KB TAIL of the log, and both logs
        already carry thousands of older lines (150 500s in sam3's, 90 'Grasp
        planning failed' in graspnet's), so any rule over that tail answers
        about someone else's request. The window is taken by byte offset,
        snapshotted before the cell runs -- no timestamp parsing, no agreement
        about log format.
        """
        src = self._src()
        self.assertIn("BAD_INPUT", src)
        self.assertIn("Point prompt inference failed", src)
        # the window, and the fact that it is taken BEFORE the run
        self.assertIn("log_marks = _service_log_marks()", src)
        self.assertLess(src.index("log_marks = _service_log_marks()"),
                        src.index("proc = subprocess.run"),
                        "the log offsets must be taken before the cell runs")
        self.assertIn("def _log_since", src)
        # the 200 KB tail is gone
        self.assertNotIn("[-200_000:]", src)
        # a rotated or truncated log yields no window, which refuses
        self.assertIn("if size < start:", src)
        # three states, and unknown refuses
        self.assertIn('return "unknown"', src)

    def test_the_kept_cell_says_why(self):
        src = self._src()
        self.assertIn("model defect, so the cell is KEPT", src)
        # and it must name BOTH routes in, or the log reads as if only 4xx
        # cells were ever kept
        self.assertIn("bad-input 5xx", src)


class TestTheTokenIsFoundWithoutCeremony(unittest.TestCase):
    """A missing credential turns every private store into a null result.

    2026-10-27: the local token had been rotated, huggingface_hub fell back to
    a stale cached one, and condition_report.py printed "not collected (store
    absent)" for an ablation that was 70/70 complete. The working token was in
    .env the whole time. Two halves: store_exists refuses to answer when the
    hub cannot be asked, and the token is loaded wherever the project is
    entered from.
    """

    ROOT = Path(__file__).resolve().parents[1]

    def test_python_entry_points_load_dotenv(self):
        src = (self.ROOT / "opencap" / "config.py").read_text()
        self.assertIn("def _load_dotenv", src)
        self.assertIn("_load_dotenv()", src)
        # an exported variable always wins: a shell that set a different token
        # is making a choice, and overriding it would be its own version of
        # this bug
        self.assertIn("if k and k not in _os.environ:", src)

    def test_it_really_puts_the_value_in_the_environment(self):
        import importlib
        import opencap.config as cfg
        d = tempfile.mkdtemp()
        f = Path(d) / ".env"
        f.write_text('# a comment\nOPENCAP_TEST_DOTENV="hunter2"\nnot a pair\n')
        # point the loader at our file by calling it with the repo layout it
        # expects: parent of the module's directory
        old = cfg._Path
        try:
            cfg._Path = lambda *_a, **_k: type(
                "P", (), {"resolve": lambda s: type(
                    "Q", (), {"parent": type(
                        "R", (), {"parent": Path(d)})()})()})()
            os.environ.pop("OPENCAP_TEST_DOTENV", None)
            cfg._load_dotenv()
            self.assertEqual(os.environ.get("OPENCAP_TEST_DOTENV"), "hunter2")
            # and it does not override
            os.environ["OPENCAP_TEST_DOTENV"] = "mine"
            cfg._load_dotenv()
            self.assertEqual(os.environ["OPENCAP_TEST_DOTENV"], "mine")
        finally:
            cfg._Path = old
            os.environ.pop("OPENCAP_TEST_DOTENV", None)
            importlib.reload(cfg)

    def test_the_shell_reads_dotenv_too_and_never_sources_it(self):
        sh = (self.ROOT / "scripts" / "env_server.sh").read_text()
        self.assertIn(".env", sh)
        # sed for the one value, NOT `source`: a stray command in .env must not
        # be executed by every script that sets up an environment
        self.assertIn("sed -n 's/^HF_TOKEN=//p'", sh)
        block = sh[sh.index("# --- Token:"):sh.index("# --- Serving")]
        # Check the CODE, not the prose: the comments here discuss sourcing in
        # order to rule it out, so a substring search over the whole block
        # fails on its own explanation.
        code = [ln.strip() for ln in block.splitlines()
                if ln.strip() and not ln.strip().startswith("#")]
        for ln in code:
            self.assertNotIn("source ", ln, f"env_server.sh sources .env: {ln}")
            self.assertFalse(ln.startswith(". "),
                             f"env_server.sh dot-sources .env: {ln}")
        # OPENCAP_DIR defaults to the POD path, so a workstation needs the
        # script's own parent as a fallback
        self.assertIn("BASH_SOURCE", sh)
        # and presence only, never the value
        self.assertIn("hf token: present", sh)
        self.assertNotIn("echo \"[env] hf token: $HF_TOKEN", sh)


class TestTheFrameTaggingPassIsDrivable(unittest.TestCase):
    """A2's claim is about FRAME DEFECTS; the success rate is a proxy for it.

    The measurement needs the arm's failing programs exported, repaired to
    ceiling, minimised and tagged -- the same pass S3/S4 went through. It had
    no driver, so the expensive half of A2 was undocumented.
    """

    ROOT = Path(__file__).resolve().parents[1]

    def test_the_arm_export_cannot_overwrite_the_baselines(self):
        """results/failures_s3.json is the BASELINE's population, and every
        published S3/S4 tag derives from it."""
        sh = (self.ROOT / "scripts" / "run_frame_tagging.sh").read_text()
        self.assertIn("failures_frame_", sh)
        self.assertIn("--failures", sh)
        # and the batch must accept being pointed elsewhere
        batch = (self.ROOT / "scripts" / "spill_wipe_batch.py").read_text()
        self.assertIn('ap.add_argument("--failures"', batch)
        # with a default that keeps existing invocations working
        self.assertIn('results/failures_{a.tier.lower()}.json', batch)
        # and it must refuse a path that is not there rather than crash
        self.assertIn("no such failures file", batch)

    def test_repairing_requires_the_stack_but_tagging_does_not(self):
        """A repair is VERIFIED BY RUNNING it, so at S3/S4 perception must
        answer -- a repair verified against a dead stack is a verified nothing,
        and it would enter the corpus as a measured distance."""
        sh = (self.ROOT / "scripts" / "run_frame_tagging.sh").read_text()
        self.assertIn("verify_servers.py", sh)
        self.assertIn("--dry-run", sh)
        # the check is skipped only for the no-GPU path
        self.assertIn('if [ -z "$DRY" ]; then', sh)

    def test_it_says_the_result_is_the_share_not_the_rate(self):
        sh = (self.ROOT / "scripts" / "run_frame_tagging.sh").read_text()
        self.assertIn("THE RESULT IS THE FRAME SHARE, NOT THE RATE", sh)
        # and that a null is publishable, which is the standing instruction
        self.assertIn("comes out of the paper", sh)
        # provenance: a rater who knows the arm is not measuring the arm
        self.assertIn("blind", sh)


class TestTheServiceLogWindowIsReallyScoped(unittest.TestCase):
    """Behaviour, not a source grep: the window must hold THIS cell's lines.

    The whole 500-classification decision rests on this. sam3_server.log
    already carries 150 500s and graspnet 90 `Grasp planning failed`, so a
    guard reading a fixed tail answers about somebody else's request -- and
    with the window wrong, no amount of pattern-matching could have fixed it.
    """

    ROOT = Path(__file__).resolve().parents[1]

    def _rc(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "rc_under_test", self.ROOT / "scripts" / "run_cell.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def setUp(self):
        self.d = tempfile.mkdtemp()
        self._old = os.environ.get("OPENCAP_LOGDIR")
        os.environ["OPENCAP_LOGDIR"] = self.d
        self.log = Path(self.d) / "sam3_server.log"
        # fifty OLD bad-input 500s, i.e. the real state of these logs
        self.log.write_text('"POST /segment_point 500 -" '
                            'Point prompt inference failed\n' * 50)

    def tearDown(self):
        if self._old is None:
            os.environ.pop("OPENCAP_LOGDIR", None)
        else:
            os.environ["OPENCAP_LOGDIR"] = self._old

    def test_the_window_excludes_everything_written_before_the_cell(self):
        rc = self._rc()
        marks = rc._service_log_marks()
        # nothing appended yet: an EMPTY window, which classifies as unknown
        # and therefore refuses. That is the conservative default.
        self.assertEqual(rc._log_since("8114", marks), "")
        with self.log.open("a") as f:
            f.write('"POST /segment_point 200 -" ok\n')
            f.write('Point prompt inference failed: Sizes of tensors '
                    'must match\n')
        win = rc._log_since("8114", marks)
        self.assertNotIn("POST /segment_point 500", win,
                         "the window must not reach the 50 older 500s")
        self.assertIn("Sizes of tensors must match", win)
        self.assertEqual(len(win.strip().splitlines()), 2)

    def test_a_rotated_or_truncated_log_yields_no_window(self):
        """Then the offset means nothing, so the cell must be refused rather
        than classified from a window that cannot be trusted."""
        rc = self._rc()
        marks = rc._service_log_marks()
        self.log.write_text("rotated\n")
        self.assertEqual(rc._log_since("8114", marks), "")

    def test_an_unknown_port_is_never_classified(self):
        rc = self._rc()
        self.assertEqual(rc._log_since("9999", rc._service_log_marks()), "")

    def test_a_missing_log_marks_zero_and_reads_whole(self):
        """Marking 0 for an absent log makes the later read fall back to the
        whole file, which can only make the guard refuse more often."""
        rc = self._rc()
        (Path(self.d) / "graspnet_server.log").unlink(missing_ok=True)
        self.assertEqual(rc._service_log_marks()["8115"], 0)


class TestTheMeasuredAcceptanceBar(unittest.TestCase):
    """30/30 is unreachable for a grasping task even by CORRECT code.

    The qwen35-27b cube_lifting program scores 27/26/28 over three draws with
    23 scenes always passing and ZERO always failing. plan_grasp resamples
    (the best grasp moves 1.4-15.2 mm on the same mask), so `score == trials`
    would send correct code back for rework forever.

    The bar is instead: no scene fails in ALL k draws.
    """

    ROOT = Path(__file__).resolve().parents[1]

    def _mod(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "_swb", self.ROOT / "scripts" / "spill_wipe_batch.py")
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m

    def test_a_scene_that_passes_once_is_not_a_defect(self):
        m = self._mod()
        draws = [{1: True, 2: False, 3: True},
                 {1: True, 2: True,  3: False},
                 {1: True, 2: False, 3: True}]
        ok, bad = m.accepted(draws, 3)
        self.assertTrue(ok, "every scene passes somewhere; nothing is systematic")
        self.assertEqual(bad, [])

    def test_a_scene_that_never_passes_blocks_acceptance(self):
        m = self._mod()
        draws = [{1: True, 2: False}, {1: True, 2: False}, {1: True, 2: False}]
        ok, bad = m.accepted(draws, 2)
        self.assertFalse(ok)
        self.assertEqual(bad, [2])

    def test_a_short_run_is_not_accepted_by_accident(self):
        # A run that produced fewer scenes than asked for must not pass just
        # because the scenes it DID produce all succeeded.
        m = self._mod()
        ok, _ = m.accepted([{1: True, 2: True}], 30)
        self.assertFalse(ok)

    def test_no_draws_is_not_acceptance(self):
        m = self._mod()
        ok, _ = m.accepted([], 30)
        self.assertFalse(ok)

    def test_spill_wipe_stays_on_one_draw_and_grasping_tasks_do_not(self):
        m = self._mod()
        self.assertEqual(m.DEFAULT_DRAWS.get("spill_wipe"), 1)
        for t in ("cube_lifting", "cube_stack", "nut_assembly"):
            self.assertEqual(m.DEFAULT_DRAWS.get(t, 3), 3,
                             "grasping tasks need pooled draws")


class TestRepairsKnowWhichTierTheyAreFor(unittest.TestCase):
    """results/repairs/<task>.json is keyed by program id and holds every
    repair for that task. Until 2026-10-03 every record was S1, because S1
    was the only tier with a corpus -- so nothing needed to say so.

    The first S3 repair landed in the same file as the 47 S1 ones and was
    indistinguishable from them. A tag or distance report would then mix two
    tiers and nothing would notice, which is exactly the defect
    export_failures.py had (group_failures buckets by program_id alone).
    """

    ROOT = Path(__file__).resolve().parents[1]

    def test_every_saved_repair_declares_its_tier(self):
        import json
        missing = []
        for kind in ("repairs", "minimal"):
            for f in sorted((self.ROOT / "results" / kind).glob("*.json")):
                for pid, rec in json.loads(f.read_text()).items():
                    if "tier" not in rec:
                        missing.append(f"{kind}/{f.name}:{pid}")
        self.assertEqual(missing, [], f"records with no tier: {missing[:5]}")

    def test_save_repair_writes_the_tier_through(self):
        import json, sys, importlib.util, tempfile
        spec = importlib.util.spec_from_file_location(
            "_save_repair", self.ROOT / "scripts" / "save_repair.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            mod.save("spill_wipe", "aaa", "x = 1", root=root, tier="S4")
            mod.save("spill_wipe", "bbb", "x = 2", root=root)          # default
            d = json.loads((root / "results" / "repairs" / "spill_wipe.json").read_text())
            self.assertEqual(d["aaa"]["tier"], "S4")
            self.assertEqual(d["bbb"]["tier"], "S1",
                             "the default must stay S1: the existing corpus is S1")

    def test_it_still_refuses_to_clobber(self):
        # The original reason this helper exists -- writing a fresh dict
        # silently deletes every other repair for the task.
        import importlib.util, tempfile
        spec = importlib.util.spec_from_file_location(
            "_save_repair2", self.ROOT / "scripts" / "save_repair.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            mod.save("spill_wipe", "aaa", "x = 1", root=root, tier="S3")
            with self.assertRaises(SystemExit):
                mod.save("spill_wipe", "aaa", "x = 2", root=root, tier="S3")


class TestTheWorkLandsOnTheCardYouNamed(unittest.TestCase):
    """OPENCAP_GPU and CUDA_VISIBLE_DEVICES must not be allowed to disagree.

    env_server.sh derives CUDA_VISIBLE_DEVICES from OPENCAP_GPU at SOURCE time
    and defaults OPENCAP_GPU to 2, so the careful-looking

        source scripts/env_server.sh
        export OPENCAP_GPU=5

    leaves CUDA_VISIBLE_DEVICES=2 and runs the job on a neighbour's card. The
    NVML reads (nvidia-smi -i $OPENCAP_GPU) report the card you meant, so
    nothing in the log looks wrong. This put a verification run on GPU 2 on
    2026-10-02 while the owner had asked for GPU 5 only.
    """

    ROOT = Path(__file__).resolve().parents[1]

    def test_env_server_really_does_derive_at_source_time(self):
        # The premise of every other test here. If env_server ever starts
        # recomputing on demand, these guards become dead weight and should go.
        src = (self.ROOT / "scripts" / "env_server.sh").read_text()
        self.assertIn('export CUDA_VISIBLE_DEVICES="$OPENCAP_GPU"', src)
        self.assertIn('export OPENCAP_GPU="${OPENCAP_GPU:-2}"', src)

    def test_gate0_refuses_when_the_two_disagree(self):
        import subprocess
        env = dict(os.environ, OPENCAP_GPU="5", CUDA_VISIBLE_DEVICES="2")
        r = subprocess.run(
            ["python3", str(self.ROOT / "scripts" / "reduced_api_gate0.py"),
             "cube_stack", "--code", "/dev/null"],
            capture_output=True, text=True, timeout=60, env=env)
        self.assertEqual(r.returncode, 1)
        self.assertIn("REFUSING", r.stdout)
        self.assertIn("would run on GPU 2", r.stdout)

    def test_the_guard_passes_when_they_agree_and_when_unset(self):
        """Unit-tested, NOT through the CLI.

        The first version of this drove the CLI with matching values, which
        meant it sailed past the guard AND the (live) service check and
        launched a real 30-trial run. subprocess killed the parent at its
        timeout and the uv child kept going, competing with the verification
        it was supposed to be testing. A test that starts GPU work is a bug
        whichever way it is written.
        """
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "_gate0", self.ROOT / "scripts" / "reduced_api_gate0.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        saved = {k: os.environ.get(k) for k in ("OPENCAP_GPU", "CUDA_VISIBLE_DEVICES")}
        try:
            os.environ["OPENCAP_GPU"] = "5"
            os.environ["CUDA_VISIBLE_DEVICES"] = "5"
            self.assertIsNone(mod.assert_gpu_is_consistent())

            os.environ["CUDA_VISIBLE_DEVICES"] = "2"
            msg = mod.assert_gpu_is_consistent()
            self.assertIsNotNone(msg)
            self.assertIn("would run on GPU 2, not 5", msg)

            # Neither set: nothing to contradict, so do not block the run.
            os.environ.pop("OPENCAP_GPU")
            os.environ.pop("CUDA_VISIBLE_DEVICES")
            self.assertIsNone(mod.assert_gpu_is_consistent())
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_bigrung_rederives_rather_than_reexports(self):
        # Re-exporting OPENCAP_GPU alone moves the NVML reads without moving the
        # work -- the worst possible combination, because the guards then
        # measure the right card and the job uses the wrong one.
        src = (self.ROOT / "scripts" / "run_bigrung_pass.sh").read_text()
        i = src.index("source scripts/env_server.sh")
        after = src[i:]
        self.assertIn('export CUDA_VISIBLE_DEVICES="$GPU"', after)
        self.assertIn('export MUJOCO_EGL_DEVICE_ID="$GPU"', after)


class TestWatchdogSurvivesWhatTheSupervisorCannot(unittest.TestCase):
    """cron puts the supervisor back when the sweep takes IT.

    `setsid nohup` was never the gap -- the collection already survives a
    closed session (PPID 1, no TTY, stdin on /dev/null). The gap is that the
    external GPU sweep kills by user, so it takes the supervisor too: four
    times in ~77 hours, and the 2026-10-01 23:40 one cost four hours because
    nothing was left to notice. A user-space daemon cannot protect against a
    sweep of user-space daemons.

    Every test here drives the script for real against stubs. The supervisor
    bug this layer exists beside was a control-flow bug that no source grep
    would have found.
    """

    ROOT = Path(__file__).resolve().parents[1]

    def _sandbox(self, td, *, supervisor_running, joblog_text="[sweep] nothing\n"):
        """A throwaway repo with the real watchdog and a stub supervisor."""
        import shutil, textwrap

        td = Path(td)
        (td / "scripts").mkdir(parents=True, exist_ok=True)
        shutil.copy(self.ROOT / "scripts" / "supervisor_watchdog.sh",
                    td / "scripts" / "supervisor_watchdog.sh")
        # The stub records that it ran; the real one would start a collection.
        (td / "scripts" / "supervise.sh").write_text(
            f"#!/usr/bin/env bash\necho started \"$@\" > {td}/SUPERVISOR_STARTED\n")
        (td / "scripts" / "supervise.sh").chmod(0o755)

        binn = td / "bin"
        binn.mkdir(exist_ok=True)
        (binn / "pgrep").write_text(
            "#!/usr/bin/env bash\nexit %d\n" % (0 if supervisor_running else 1))
        (binn / "pgrep").chmod(0o755)

        joblog = td / "phase9.log"
        joblog.write_text(joblog_text)

        env = dict(os.environ)
        env.update(
            PATH=f"{binn}:{env['PATH']}",
            OPENCAP_LOGDIR=str(td),
            OPENCAP_JOBLOG=str(joblog),
        )
        return td, env

    def _run(self, td, env):
        import subprocess
        return subprocess.run(
            ["bash", str(Path(td) / "scripts" / "supervisor_watchdog.sh")],
            capture_output=True, text=True, timeout=60, env=env,
        )

    def test_it_never_starts_a_second_supervisor(self):
        # Two supervisors means two collections on one card, which is worse
        # than none: they fight over :8110 and the flock, and the failures
        # look like the memory failures the big-rung pass exists to fix.
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            td, env = self._sandbox(td, supervisor_running=True)
            r = self._run(td, env)
            self.assertEqual(r.returncode, 0)
            self.assertFalse((td / "SUPERVISOR_STARTED").exists())
            # And it stays silent: this is the 99% case and must not fill
            # a log that someone has to read after an outage.
            self.assertFalse((td / "watchdog.log").exists())

    def test_it_puts_a_dead_supervisor_back(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            td, env = self._sandbox(td, supervisor_running=False)
            r = self._run(td, env)
            self.assertEqual(r.returncode, 0)
            self.assertTrue((td / "SUPERVISOR_STARTED").exists())
            self.assertIn("starting one", (td / "watchdog.log").read_text())

    def test_an_explicit_human_stop_is_permanent(self):
        # If --stop could be undone by cron five minutes later, the operator
        # would be fighting their own tooling, which is worse than no tooling.
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            td, env = self._sandbox(td, supervisor_running=False)
            (td / "watchdog.off").write_text("stopped by hand\n")
            r = self._run(td, env)
            self.assertEqual(r.returncode, 0)
            self.assertFalse((td / "SUPERVISOR_STARTED").exists())
            # And it did not quietly delete the operator's stop file.
            self.assertTrue((td / "watchdog.off").exists())

    def test_it_disables_itself_when_the_work_is_done(self):
        # Otherwise it respawns a supervisor every 5 minutes forever, each one
        # exiting on the same marker, writing noise until someone notices.
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            td, env = self._sandbox(
                td, supervisor_running=False,
                joblog_text="[sweep] all done\nBIGRUNG_S4_RC=0\n")
            r = self._run(td, env)
            self.assertEqual(r.returncode, 0)
            self.assertFalse((td / "SUPERVISOR_STARTED").exists())
            self.assertIn("completed", (td / "watchdog.off").read_text())

    def test_stop_disarms_the_watchdog_and_resume_rearms_it(self):
        import subprocess, tempfile
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            env = dict(os.environ)
            env["OPENCAP_LOGDIR"] = str(td)
            sup = str(self.ROOT / "scripts" / "supervise.sh")

            r = subprocess.run(["bash", sup, "--stop"], capture_output=True,
                               text=True, timeout=30, env=env)
            self.assertEqual(r.returncode, 0)
            self.assertTrue((td / "watchdog.off").exists(),
                            "--stop must stop BOTH layers, or cron undoes it")
            self.assertIn("--resume", r.stdout, "the way back must be printed")

            r = subprocess.run(["bash", sup, "--resume"], capture_output=True,
                               text=True, timeout=30, env=env)
            self.assertEqual(r.returncode, 0)
            self.assertFalse((td / "watchdog.off").exists())

    def test_the_cron_installer_is_idempotent(self):
        # Run by someone who does not remember whether they already ran it.
        # Three copies of the line is three supervisors.
        import subprocess, tempfile, textwrap
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            binn = td / "bin"
            binn.mkdir()
            tab = td / "crontab.txt"
            tab.write_text("")
            (binn / "crontab").write_text(textwrap.dedent(f"""\
                #!/usr/bin/env bash
                if [ "${{1:-}}" = "-l" ]; then cat {tab}; exit 0; fi
                if [ "${{1:-}}" = "-" ]; then cat > {tab}; exit 0; fi
                exit 0
            """))
            (binn / "crontab").chmod(0o755)

            env = dict(os.environ)
            env["PATH"] = f"{binn}:{env['PATH']}"
            inst = str(self.ROOT / "scripts" / "install_watchdog_cron.sh")

            for _ in range(3):
                r = subprocess.run(["bash", inst], capture_output=True,
                                   text=True, timeout=30, env=env)
                self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

            lines = [l for l in tab.read_text().splitlines()
                     if l.startswith("*/5") and "supervisor_watchdog" in l]
            self.assertEqual(len(lines), 1, f"crontab after 3 installs: {tab.read_text()!r}")

            r = subprocess.run(["bash", inst, "--remove"], capture_output=True,
                               text=True, timeout=30, env=env)
            self.assertEqual(r.returncode, 0)
            self.assertNotIn("supervisor_watchdog", tab.read_text())


class TestSupervisorStopsRatherThanPersists(unittest.TestCase):
    """An auto-restarter on a shared GPU must be biased toward giving up.

    It exists because an external sweep killed the collection AND all its
    services three times in ~54 hours (2026-09-29 19:05, 2026-09-30 07:51,
    2026-10-01 13:16) and each time the run sat idle until a human noticed --
    the second cost 6h12m. But a restarter that loops forever on a broken job
    burns a card other people want, so every guard here fails closed.
    """

    ROOT = Path(__file__).resolve().parents[1]

    def _src(self) -> str:
        p = self.ROOT / "scripts" / "supervise.sh"
        self.assertTrue(p.exists())
        return p.read_text()

    def test_it_exits_on_the_jobs_own_completion_marker(self):
        # Not a timeout, not a cell count -- the job says when it is done, so
        # a finished run can never be restarted into a loop.
        src = self._src()
        self.assertIn("DONE_MARK", src)
        self.assertIn("BIGRUNG_S4_RC=", src)
        self.assertIn("TIER_S4_RC=", src)

    def test_it_caps_restarts(self):
        src = self._src()
        self.assertIn("MAX_RESTARTS", src)
        self.assertIn("refusing to keep going", src)

    def test_a_fast_redeath_is_treated_as_a_startup_failure(self):
        # A job that dies seconds after launch is broken, not unlucky, and
        # restarting it is how you burn a GPU all night for nothing.
        src = self._src()
        self.assertIn("MIN_ALIVE", src)
        self.assertIn("startup failure", src)

    def test_it_refuses_to_start_the_job_without_services(self):
        # launch.py warns and runs the cell anyway against a dead service,
        # scoring an environment failure as a model defect.
        src = self._src()
        self.assertIn("start_vision_servers.sh", src)
        self.assertIn("start_molmo_server.sh", src)
        self.assertIn("NOT starting the job", src)

    def test_it_never_starts_a_second_collection(self):
        src = self._src()
        self.assertIn("JOB_PAT", src)
        self.assertIn("pgrep -f \"$JOB_PAT\"", src)

    def test_it_can_be_stopped_and_inspected_without_guessing(self):
        src = self._src()
        self.assertIn("--stop", src)
        self.assertIn("--status", src)

    def test_a_full_card_is_a_queue_not_a_failure(self):
        """The 2026-10-02 idle: a neighbour took 81 GB of GPU 5, the pass
        refused in four seconds, and the supervisor would have read that as a
        startup failure and quit on work that was only waiting its turn.

        Driven for real against a stub nvidia-smi, because the bug was in the
        control flow, not in any string the source contains.
        """
        import subprocess, tempfile, textwrap

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            binn = td / "bin"
            binn.mkdir()
            # A card with nothing free: every pool is refused.
            (binn / "nvidia-smi").write_text(textwrap.dedent("""\
                #!/usr/bin/env bash
                case "$*" in
                  *memory.free*)  echo 1024 ;;
                  *memory.total*) echo 143771 ;;
                  *) exit 0 ;;
                esac
            """))
            # No job is running.
            (binn / "pgrep").write_text("#!/usr/bin/env bash\nexit 1\n")
            for f in binn.iterdir():
                f.chmod(0o755)
            joblog = td / "phase9.log"
            joblog.write_text("[sweep] nothing finished here\n")

            env = dict(os.environ)
            env.update(
                PATH=f"{binn}:{env['PATH']}",
                OPENCAP_LOGDIR=str(td),
                OPENCAP_JOBLOG=str(joblog),
                OPENCAP_SUPERVISE_INTERVAL="1",
                OPENCAP_SUPERVISE_MAX_WAIT="3",
                OPENCAP_GPU="5",
            )
            r = subprocess.run(
                ["bash", str(self.ROOT / "scripts" / "supervise.sh"), "bigrung"],
                capture_output=True, text=True, timeout=60, env=env,
            )

            # It waited and said so, rather than calling it a startup failure.
            self.assertIn("card busy", r.stdout)
            self.assertNotIn("startup failure", r.stdout)
            # It never tried to launch anything into a card that cannot hold it.
            self.assertNotIn("relaunching", r.stdout)
            self.assertNotIn("restart 1/", r.stdout)
            # And the wait is bounded: it gives up rather than holding forever.
            self.assertEqual(r.returncode, 1)
            self.assertIn("gave-up-waiting", (td / "supervise.state").read_text())

    def test_a_single_lucky_sample_does_not_launch_an_engine(self):
        """GPU 5 read 34, 57 and 36 GB free inside five minutes while a
        neighbour's jobs burst. One clear sample is not headroom; launching
        on it earns vLLM's memory-profiling error and loses the rung.
        """
        import subprocess, tempfile, textwrap, time

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            binn = td / "bin"
            binn.mkdir()
            # A card with room to spare, every single time it is asked.
            (binn / "nvidia-smi").write_text(textwrap.dedent("""\
                #!/usr/bin/env bash
                case "$*" in
                  *memory.free*)  echo 140000 ;;
                  *memory.total*) echo 143771 ;;
                  *) exit 0 ;;
                esac
            """))
            (binn / "pgrep").write_text("#!/usr/bin/env bash\nexit 1\n")
            # If the streak guard fails, THIS is what gets run. Make that
            # loud rather than letting the test quietly start real servers.
            (binn / "start_vision_servers.sh").write_text(
                "#!/usr/bin/env bash\necho LAUNCHED-SERVICES\n")
            for f in binn.iterdir():
                f.chmod(0o755)
            joblog = td / "phase9.log"
            joblog.write_text("[sweep] nothing finished here\n")
            out = td / "sup.log"

            env = dict(os.environ)
            env.update(
                PATH=f"{binn}:{env['PATH']}",
                OPENCAP_LOGDIR=str(td),
                OPENCAP_JOBLOG=str(joblog),
                OPENCAP_SUPERVISE_INTERVAL="1",
                OPENCAP_SUPERVISE_HEADROOM_CHECKS="99",
                OPENCAP_GPU="5",
            )
            with out.open("w") as fh:
                proc = subprocess.Popen(
                    ["bash", str(self.ROOT / "scripts" / "supervise.sh"), "bigrung"],
                    stdout=fh, stderr=subprocess.STDOUT, env=env,
                )
                try:
                    time.sleep(4)
                    (td / "supervise.stop").write_text("")
                    proc.wait(timeout=30)
                finally:
                    if proc.poll() is None:
                        proc.kill()

            log = out.read_text()
            self.assertIn("confirming", log)
            # Several clear samples in a row, and it still has not launched.
            self.assertNotIn("relaunching", log)
            self.assertNotIn("LAUNCHED-SERVICES", log)

    def test_the_wait_is_bounded_and_the_bound_is_configurable(self):
        src = self._src()
        self.assertIn("MAX_WAIT", src)
        self.assertIn("OPENCAP_SUPERVISE_MAX_WAIT", src)
        # A wait must never be mistaken for a live job by the MIN_ALIVE parse,
        # which reads field 2 of the state file as an epoch.
        self.assertIn("waiting-for-card", src)


class TestAlternationProbe(unittest.TestCase):
    """The one open mechanism question, and the ways the probe could fake it.

    qwen3-4b's cube_stack program fails trial 3 every draw and then EITHER
    trial 9 or trial 13 -- never both, never neither, across eight draws
    (1/256 under independence). At workers=1 the trial order is fixed, so a
    per-worker carry-over has nothing left to vary; the prediction is that the
    alternation stops.
    """

    ROOT = Path(__file__).resolve().parents[1]

    def test_the_probe_records_every_failing_trial(self):
        src = (self.ROOT / "scripts" / "run_scene_probe.py").read_text()
        # "Do trials 9 and 13 fail together" cannot be asked of one scene's
        # outcome. The first probe stored only the scene of interest and the
        # alternation had to be dug out of the run dirs by hand.
        self.assertIn('rec["failed_trials"]', src)

    def test_worker_count_is_recorded_and_separates_the_scratch(self):
        src = (self.ROOT / "scripts" / "run_scene_probe.py").read_text()
        # Two worker settings are two conditions. They must not pool, and must
        # not share an output directory -- a re-run into a dir that already
        # holds trial dirs makes the parse read a mixture.
        self.assertIn('"workers": a.workers', src)
        self.assertIn('f"w{a.workers}_{version}_{draw}"', src)

    def test_resume_does_not_confuse_two_worker_settings(self):
        src = (self.ROOT / "scripts" / "run_scene_probe.py").read_text()
        # Without the worker count in the resume key, the workers=1 pass would
        # skip every draw the workers=4 pass already did and collect nothing,
        # reporting success.
        self.assertIn('r.get("workers", 4))', src)
        self.assertIn("(pid, version, draw, a.workers) in seen", src)

    def test_a_targeted_probe_adds_no_silent_controls(self):
        src = (self.ROOT / "scripts" / "run_scene_probe.py").read_text()
        # The comparison here is one program against itself under a different
        # condition. Controls would be three unrelated programs and would make
        # the run look like the tag probe, which it is not.
        self.assertIn('"--pids"', src)
        self.assertIn("no controls", src)

    def test_the_report_needs_both_conditions(self):
        src = (self.ROOT / "scripts" / "alternation_report.py").read_text()
        # One worker setting alone cannot answer it; the report must say so
        # rather than print a one-sided table that reads like a result. It
        # must also say what TWO counts cannot answer -- that parallelism
        # matters is not why it matters.
        self.assertIn("answers nothing", src)
        self.assertIn("not enough", src)
        self.assertIn("n = 1 program", src)

    def test_the_report_runs_on_both_conditions(self):
        import importlib.util
        import json
        import sys
        import tempfile
        spec = importlib.util.spec_from_file_location(
            "alternation_report", self.ROOT / "scripts" / "alternation_report.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["alternation_report"] = mod
        spec.loader.exec_module(mod)
        pattern = {0: [3, 9], 1: [3, 9], 2: [3, 13], 3: [3, 13],
                   4: [3, 9], 5: [3, 13], 6: [3, 13], 7: [3, 9]}
        recs = [{"pid": "p", "version": "original", "draw": d, "task": "cube_stack",
                 "model": "qwen3-4b", "scene": 13, "workers": 4, "ok": True,
                 "successes": 28, "n": 30, "failed_trials": f}
                for d, f in pattern.items()]
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump(recs, fh)
            path = fh.name
        loaded = mod.load([path])
        self.assertEqual(len(loaded), 8)


class TestSecondRaterPacket(unittest.TestCase):
    """The packet's job is to produce a BLIND rating. Everything here pins one
    way it could stop being blind, or stop being scorable.

    The Phase 9 family contrasts rest on one rater. A second rating that is
    contaminated is worse than none, because it gets reported as agreement.
    """

    ROOT = Path(__file__).resolve().parents[1]
    PACKET = ROOT / "artifacts" / "second_rater"

    def test_the_rater_brief_does_not_state_the_hypothesis(self):
        brief = (self.PACKET / "RATER_TASK.md").read_text()
        tax = (self.PACKET / "taxonomy_full.md").read_text()
        # Neither file the rater reads may name the comparison, the tiers, the
        # expected direction, or the measurement.
        for leak in ("S1", "S3", "S4", "kappa", "reliability", "inter-rater",
                     "hypothesis is", "we expect", "rises", "scaffold",
                     "abstraction", "privileged", "reduced API"):
            self.assertNotIn(leak, brief, f"RATER_TASK.md leaks {leak!r}")
            self.assertNotIn(leak, tax, f"taxonomy_full.md leaks {leak!r}")

    def test_the_brief_names_the_files_not_to_open(self):
        brief = (self.PACKET / "RATER_TASK.md").read_text()
        for f in ("key.json", "retag_session2.json", "README.md", "results/",
                  "STATUS.md", "editclass.py"):
            self.assertIn(f, brief)
        # and asks the rater to report contamination rather than hide it
        self.assertIn("stop and say so", brief)

    def test_the_operator_notes_are_not_given_to_the_rater(self):
        op = (self.PACKET / "OPERATOR.md").read_text()
        # The operator file is where the hypothesis lives, so it must say
        # plainly that the rater does not get it.
        self.assertIn("Do not give this file to the rater", op)
        # and it must distinguish the two kinds of second rater, because
        # reporting either as plain "inter-rater agreement" overstates it
        self.assertIn("LLM", op)
        self.assertIn("human", op)

    def test_the_taxonomy_documents_every_real_tag_and_no_others(self):
        import re
        from opencap.editclass import FAMILY
        tax = (self.PACKET / "taxonomy_full.md").read_text()
        named = set(re.findall(r"^### `(\w+)`", tax, re.M))
        self.assertEqual(named, set(FAMILY))

    def test_the_worked_examples_are_not_in_the_packet(self):
        import json
        # Illustrating a tag with a packet item would hand the rater an answer.
        key = json.loads((self.PACKET / "key.json").read_text())
        pids = {v["pid"] for v in key.values()}
        brief = (self.PACKET / "RATER_TASK.md").read_text()
        tax = (self.PACKET / "taxonomy_full.md").read_text()
        for pid in pids:
            self.assertNotIn(pid, brief)
            self.assertNotIn(pid, tax)
        self.assertIn("not** in the packet", brief)

    def test_the_scorer_reproduces_the_published_same_rater_figures(self):
        """The published kappa had no committed generator. This is it, and it
        must agree with the numbers already in STATUS -- otherwise one of the
        two is wrong and we would not know which."""
        import csv
        import importlib.util
        import json
        import subprocess
        import sys
        import tempfile
        retag = json.loads((self.PACKET / "retag_session2.json").read_text())
        with (self.PACKET / "answers_template.csv").open(newline="") as fh:
            rows = [{"id": r["id"], "task": r["task"],
                     "tags": ";".join(retag.get(r["id"], [])), "notes": ""}
                    for r in csv.DictReader(fh)]
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False,
                                         newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["id", "task", "tags", "notes"])
            w.writeheader()
            w.writerows(rows)
            path = fh.name
        r = subprocess.run([sys.executable,
                            str(self.ROOT / "scripts" / "score_second_rater.py"),
                            path], capture_output=True, text=True,
                           cwd=str(self.ROOT))
        self.assertEqual(r.returncode, 0, r.stderr[-800:])
        out = r.stdout
        self.assertIn("pooled tag-level kappa  0.89", out)
        for fam, k in (("FRAME", "0.67"), ("INTERFACE", "0.90"),
                       ("SEMANTIC", "1.00"), ("STRUCTURE", "0.96"),
                       ("TRAJECTORY", "0.96")):
            self.assertRegex(out, rf"{fam}\s+{k}")

    def test_the_result_is_printed_before_the_agreement_table(self):
        src = (self.ROOT / "scripts" / "score_second_rater.py").read_text()
        # A kappa table printed first invites reading agreement as the
        # finding. The claim is whether the contrasts reproduce.
        self.assertLess(src.index("DO THE FAMILY CONTRASTS REPRODUCE"),
                        src.index("AGREEMENT WITH THE FILED TAGS"))
        self.assertLess(src.index("AGREEMENT WITH THE FILED TAGS"),
                        src.index("THE COUNT GAP"))

    def test_an_undefined_kappa_is_not_reported_as_zero(self):
        import importlib.util
        import sys
        spec = importlib.util.spec_from_file_location(
            "score_second_rater",
            self.ROOT / "scripts" / "score_second_rater.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["score_second_rater"] = mod
        spec.loader.exec_module(mod)
        # Both raters always said "no": agreement is perfect and kappa is
        # undefined. Reporting 0.0 would read as total disagreement.
        self.assertIsNone(mod.kappa([False] * 10, [False] * 10))
        self.assertAlmostEqual(mod.kappa([True, False] * 5, [True, False] * 5), 1.0)

    def test_the_register_derives_a_row_per_rater_csv(self):
        """A human rating must drop in with no code change."""
        import importlib.util
        import sys
        from pathlib import Path as P
        spec = importlib.util.spec_from_file_location(
            "cr_rater", self.ROOT / "scripts" / "claims_register.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["cr_rater"] = mod
        spec.loader.exec_module(mod)
        rows = mod.second_rater_rows(P(self.ROOT))
        csvs = [f for f in (self.PACKET).glob("answers_*.csv")
                if f.name != "answers_template.csv"]
        self.assertEqual(len(rows), len(csvs))
        for _, _, detail in rows:
            # the claim is the contrasts, not the agreement
            self.assertIn("rater's OWN tags", detail)
            self.assertIn("FRAME", detail)
            self.assertIn("kappa", detail)
            # and it must name the command that re-derives it
            self.assertIn("score_second_rater.py", detail)

    def test_the_template_is_not_scored_as_a_rating(self):
        src = (self.ROOT / "scripts" / "claims_register.py").read_text()
        # answers_template.csv has 60 empty tag cells; scoring it would report
        # a rater who found no defects anywhere.
        self.assertIn('answers_template.csv', src)

    def test_every_rating_has_recorded_provenance(self):
        """A rating with no recorded conditions is not a usable rating.

        "hand (CLI session)" in the tag records does not say which model, or
        whether it was blind. The paper has to name both.
        """
        import json
        meta = json.loads((self.PACKET / "ratings.json").read_text())
        self.assertIn("filed", meta)
        self.assertIn("who", meta["filed"])
        for csv_f in self.PACKET.glob("answers_*.csv"):
            if csv_f.name == "answers_template.csv":
                continue
            self.assertIn(csv_f.name, meta,
                          f"{csv_f.name} has no entry in ratings.json")
            e = meta[csv_f.name]
            for field in ("who", "date", "blind"):
                self.assertIn(field, e)

    def test_an_unrecorded_rating_is_flagged_not_passed(self):
        src = (self.ROOT / "scripts" / "claims_register.py").read_text()
        self.assertIn("CONDITION NOT RECORDED", src)

    def test_the_writeup_names_both_raters_as_models(self):
        w = (self.ROOT / "docs" / "WRITEUP.md").read_text()
        # "hand-tagged" carries an implication the method does not support.
        self.assertIn("both raters are language models", w.lower())
        self.assertIn("No human has rated the packet", w)
        # and the granularity boundary the second rater exposed
        self.assertIn("approach_param", w)

    def test_the_noise_floor_row_uses_both_programs(self):
        """The alternation row excludes repairs on purpose; this one must not.

        The repair is the cleaner evidence for the worker effect: it is at
        ceiling at workers=1 and loses scene 13 in half its workers=4 draws,
        which is exactly the program class Phase 6 tested and found nothing on.
        """
        import importlib.util
        import sys
        spec = importlib.util.spec_from_file_location(
            "cr_wn", self.ROOT / "scripts" / "claims_register.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["cr_wn"] = mod
        spec.loader.exec_module(mod)
        rows = mod.worker_noise_rows()
        self.assertEqual(len(rows), 1)
        _, status, detail = rows[0]
        self.assertIn(status, ("REFUTED", "HOLDS", "PENDING"))
        if status != "PENDING":
            # pooled over BOTH programs: 8 draws each per condition
            self.assertRegex(detail, r"workers=4: \d+/16")
            self.assertRegex(detail, r"workers=1: \d+/16")
            # and it must name what the comparison does NOT isolate
            self.assertIn("WHAT IS NOT SHOWN", detail)
            self.assertIn("workers=2", detail)

    def test_alternation_and_noise_rows_disagree_about_repairs(self):
        src = (self.ROOT / "scripts" / "claims_register.py").read_text()
        i = src.index("def alternation_rows")
        j = src.index("def worker_noise_rows")
        alt = src[i:] if i > j else src[i:j]
        # alternation is about WHICH trial and is original-only
        self.assertIn('r.get("version") == "original"', alt)
        # the noise row is about WHETHER a varying trial failed, both versions
        wn = src[j:i] if j < i else src[j:]
        self.assertIn('VARY = {"original"', wn)
        self.assertIn('"repair"', wn)

    def test_the_noise_row_reads_worker_counts_from_the_file(self):
        """The first version hardcoded (4, 1).

        A workers=2 arm would have been collected and silently ignored -- the
        same shape as a PENDING row left behind by the run that settled it.
        """
        src = (self.ROOT / "scripts" / "claims_register.py").read_text()
        j = src.index("def worker_noise_rows")
        k = src.index("def alternation_rows")
        body = src[j:k] if j < k else src[j:]
        self.assertNotIn("for w in (4, 1)", body)
        self.assertIn("tally.setdefault(int(w)", body)
        # and with a middle count it must print the discriminating reading
        for word in ("BINARY", "MONOTONE", "THRESHOLD"):
            self.assertIn(word, body)

    def _noise_detail(self, recs):
        import importlib.util
        import sys
        spec = importlib.util.spec_from_file_location(
            "cr_wn4", self.ROOT / "scripts" / "claims_register.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["cr_wn4"] = mod
        spec.loader.exec_module(mod)
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "scene_probe_alt.json"
            f.write_text(json.dumps(recs))
            mod.ALT_FILE = f
            return mod.worker_noise_rows()[0][2]

    @staticmethod
    def _nrec(workers, draw, failed, version):
        return {"pid": "81ee85d0f933", "version": version, "draw": draw,
                "workers": workers, "ok": True, "failed_trials": failed,
                "successes": 30 - len(failed), "n": 30}

    def test_a_fourth_count_does_not_replace_the_registered_comparison(self):
        """2026-10-15: with workers=8 present, the reading became 4-vs-8.

        The first version took the first middle count against the largest, so
        adding workers=8 printed BINARY from a comparison nobody registered
        while the registered one (2 against 4 and 1) read MONOTONE. This is
        the 2026-10-15 shape: original 0,0,7,5 and repair 0,8,4,3 of 8.
        """
        recs = []
        for w, o, r in ((1, 0, 0), (2, 0, 8), (4, 7, 4), (8, 5, 3)):
            for d in range(8):
                recs.append(self._nrec(w, d, [3, 9] if d < o else [3],
                                       "original"))
                recs.append(self._nrec(w, d, [13] if d < r else [], "repair"))
        detail = self._noise_detail(recs)
        # The label changed on 2026-10-15 when the registered statistic was
        # found to be mis-specified; the GUARD is that a fourth count must not
        # become the comparison, which both labels must preserve.
        self.assertTrue(
            "pre-registered, workers=2 vs 4 and 1" in detail
            or "VERDICT WITHHELD" in detail,
            "the registered comparison is neither reported nor withheld")
        self.assertNotIn("workers=4 is already at", detail)   # 4-vs-8 hijack
        # On THIS shape the registered statistic is mis-specified -- the
        # middle count is original 0/8 against repair 8/8, two deterministic
        # opposite behaviours whose mean sits on the decision threshold. The
        # verdict must be withheld rather than printed as MONOTONE, which is
        # what a pooled rate over units moving in opposite directions would
        # have made it. Pinned in both directions: a future change that
        # resumes printing a verdict off that average breaks this.
        self.assertIn("VERDICT WITHHELD", detail)
        self.assertIn("UNRESOLVED", detail)
        self.assertNotIn("MONOTONE in worker count, which points", detail)
        # the shape observation is not a verdict and must survive withholding
        self.assertIn("a monotone reading would need it to", detail)
        self.assertIn("original 0/8, 0/8, 7/8, 5/8", detail)
        self.assertIn("does NOT continue the rise", detail)
        self.assertIn("repair 0/8, 8/8, 4/8, 3/8", detail)
        # the repair sentence is computed, not the workers=4 prose
        self.assertNotIn("half its", detail)
        self.assertIn("3/8 at workers=8", detail)
        self.assertIn("30/30 in all 8 draws", detail)

    def test_the_sweep_records_card_occupancy(self):
        src = (self.ROOT / "scripts" / "run_worker_sweep.sh").read_text()
        # Contention is the quantity under test, so who else is on the card is
        # part of the condition, not an aside.
        self.assertIn("nvidia-smi", src)
        self.assertIn("query-compute-apps", src)
        self.assertIn('-i "${OPENCAP_GPU}"', src)
        self.assertIn("BEFORE", src)
        self.assertIn("AFTER", src)
        self.assertIn("source scripts/env_server.sh", src)

    def test_the_sweep_states_it_needs_no_servers(self):
        src = (self.ROOT / "scripts" / "run_worker_sweep.sh").read_text()
        # Established empirically: the servers were down for Phase 10 and
        # every record returned ok. Saying so stops someone restarting a 27B
        # engine for an LLM-free job.
        self.assertIn("no vision servers", src)
        self.assertIn("NO vLLM", src)

    def test_the_report_pools_both_versions_for_the_worker_table(self):
        src = (self.ROOT / "scripts" / "alternation_report.py").read_text()
        self.assertIn('VARY = {"original"', src)
        self.assertIn("BY WORKER COUNT", src)
        # and must say what two counts cannot answer
        self.assertIn("not enough", src)


class TestGapClosingExperiments(unittest.TestCase):
    """A1-A4. Each assertion pins one way an experiment could run and answer
    nothing, which is the expensive failure here -- a null that is an artifact
    of the harness costs a day of pod time and a wrong sentence in the paper.
    """

    ROOT = Path(__file__).resolve().parents[1]

    # --- A1: held-out scenes -------------------------------------------
    def test_heldout_sample_is_stratified_and_deterministic(self):
        import importlib.util
        import sys
        sys.path.insert(0, str(self.ROOT / "scripts"))
        spec = importlib.util.spec_from_file_location(
            "run_heldout_scenes", self.ROOT / "scripts" / "run_heldout_scenes.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["run_heldout_scenes"] = mod
        spec.loader.exec_module(mod)
        reps, fails = mod.load_repairs(), mod.load_failures()
        a = mod.sample(reps, fails, 30, 41)
        self.assertEqual(a, mod.sample(reps, fails, 30, 41),
                         "a sample that changes between runs is not a sample")
        tasks = {fails[p]["task"] for p in a}
        bands = {mod.band_of(fails[p]["models"][0]) for p in a}
        # all four tasks and all four bands, or it is a sample of easy repairs
        self.assertEqual(len(tasks), 4)
        self.assertEqual(len(bands), 4)

    def test_heldout_measures_both_halves_in_one_run(self):
        src = (self.ROOT / "scripts" / "run_heldout_scenes.py").read_text()
        # Two jobs would reintroduce every between-run difference this
        # project spent months removing.
        self.assertIn("--trials", src)
        self.assertIn("default=60", src)
        self.assertIn('rec["in_sample"]', src)
        self.assertIn('rec["heldout"]', src)
        # and the split must come from the observed indices, not a constant:
        # the corpus is 1..30 in one place and 0..29 in another
        self.assertIn("idx[len(idx) // 2]", src)

    def test_heldout_report_needs_the_references(self):
        src = (self.ROOT / "scripts" / "heldout_report.py").read_text()
        # A repair at 26/30 on unseen scenes means nothing until you know
        # what the human reference scores there.
        self.assertIn("NO REFERENCES IN THIS FILE", src)
        self.assertIn("not interpretable", src)
        for verdict in ("NO OVERFIT", "FITTED TO THE SCENE SET", "PARTIAL"):
            self.assertIn(verdict, src)

    # --- B1: condition ablation ----------------------------------------
    def test_every_ablation_condition_has_its_own_store(self):
        sh = (self.ROOT / "scripts" / "run_condition_ablation.sh").read_text()
        py = (self.ROOT / "scripts" / "condition_report.py").read_text()
        import re
        rows = re.findall(r"^(\w+)\|([\w.-]+)\|", sh, re.M)
        self.assertGreaterEqual(len(rows), 6)
        names = [n for n, _ in rows]
        stores = [s for _, s in rows]
        # a cell path has no room for the condition, so two conditions in one
        # store overwrite each other in place
        self.assertEqual(len(set(stores)), len(stores))
        # and the report must know about every one of them
        for n, s in rows:
            self.assertIn(f'"{n}"', py)
            self.assertIn(s, py)

    def test_gated_conditions_use_a_store_the_guards_allow(self):
        sh = (self.ROOT / "scripts" / "run_condition_ablation.sh").read_text()
        import re
        rows = {n: s for n, s, in re.findall(r"^(\w+)\|([\w.-]+)\|", sh, re.M)}
        # store.py refuses a sampled cell into a non-sampled store and a
        # raised-cap cell into a non-bigcap one.
        self.assertIn("sampled", rows.get("sampled", ""))
        self.assertIn("bigcap", rows.get("bigcap", ""))

    def test_ablation_reports_visibility_not_just_delta(self):
        py = (self.ROOT / "scripts" / "condition_report.py").read_text()
        # The methods question is whether any check here would have caught it.
        self.assertIn("visible", py.lower())
        self.assertIn("NOISE", py)
        self.assertIn("worst", py.lower())

    # --- A2: the frame-hint intervention --------------------------------
    def test_frame_patch_fails_loudly_and_is_gated(self):
        src = (self.ROOT / "scripts" / "patch_frame_hint.py").read_text()
        self.assertIn("PATTERN NOT FOUND", src)
        self.assertIn("OPENCAP_FRAME_HINT", src)
        # The addendum must name conventions only -- a line about grasping or
        # approach offsets would make a FRAME movement unattributable. Slice
        # the literal itself: the module docstring legitimately says "no
        # mention of grasping", and matching that would be matching the
        # explanation rather than the text sent to the model.
        q = chr(39) * 3
        start = src.index("ADDENDUM = " + q) + len("ADDENDUM = " + q)
        body = src[start:src.index(q, start)].lower()
        self.assertIn("quaternion", body)
        for leak in ("grasp", "lift", "z_approach", "sweep", "never_invoked"):
            self.assertNotIn(leak, body)

    def test_the_frame_arm_refuses_without_its_patch(self):
        sh = (self.ROOT / "scripts" / "run_frame_arm.sh").read_text()
        # A gated patch that never applied is indistinguishable from an
        # intervention that did nothing -- and "no effect" is one of the two
        # outcomes, so this check is what keeps the null honest.
        self.assertIn("patch_frame_hint.py", sh)
        self.assertIn("REFUSING", sh)
        self.assertIn("verify_servers.py", sh)

    def test_frame_hint_is_wired_as_an_arm_and_off_elsewhere(self):
        sh = (self.ROOT / "scripts" / "run_arm.sh").read_text()
        self.assertIn("frame_hint)", sh)
        self.assertIn("FRAME=0", sh)      # default for every other arm
        self.assertIn("unset OPENCAP_FRAME_HINT", sh)

    # --- A3 / A4 --------------------------------------------------------
    def test_m1_runner_states_what_it_cannot_test(self):
        sh = (self.ROOT / "scripts" / "run_m1_arm.sh").read_text()
        # There is no text-only reduced-API multi-turn tier. Saying so stops
        # the result being read as the sharp test it is not.
        self.assertIn("DOES NOT EXIST", sh)
        self.assertIn("27%", sh)

    def test_sampled_arm_scopes_its_store_portably(self):
        sh = (self.ROOT / "scripts" / "run_sampled_arm.sh").read_text()
        # ${TIER,,} is bash 4+; on 3.2 it expands to nothing and a cube_stack
        # run would have landed in the nut_assembly store. The rule is "do not
        # USE it" -- the comment explaining why names it on purpose, so check
        # the code lines only.
        code = "\n".join(l for l in sh.splitlines()
                          if not l.lstrip().startswith("#"))
        self.assertNotIn("${TIER,,}", code)
        self.assertIn("tr '[:upper:]' '[:lower:]'", sh)
        self.assertIn("REFUSING: store scope did not expand", sh)

    def test_a_pooled_rate_over_opposite_programs_is_withheld(self):
        """The 2026-10-14 statistic was mis-specified and the register says so.

        At workers=2 the original fails the registered trials 0/8 and the
        repair 8/8 -- two deterministic opposite behaviours whose mean is 50%,
        which is where the BINARY/MONOTONE threshold was drawn. A pooled rate
        over units moving in opposite directions is not a rate, and reporting
        a verdict off it would make a design error look like a finding.
        """
        import importlib.util
        import sys
        spec = importlib.util.spec_from_file_location(
            "cr_mis", self.ROOT / "scripts" / "claims_register.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["cr_mis"] = mod
        spec.loader.exec_module(mod)
        detail = mod.worker_noise_rows()[0][2]
        if mod.ALT_FILE.exists():
            self.assertIn("VERDICT WITHHELD", detail)
            self.assertIn("MIS-SPECIFIED", detail.upper())
            self.assertIn("UNRESOLVED", detail)
            # and the thing that IS established must still be stated
            self.assertIn("workers=1", detail)

    def test_the_probe_records_the_per_worker_order(self):
        src = (self.ROOT / "scripts" / "run_scene_probe.py").read_text()
        # The mechanism claim is reconstructed from trial-dir birth times in
        # pod scratch, which is not committed -- so the evidence for the
        # project's best hypothesis could not be checked from the repo.
        self.assertIn('rec["worker_chains"]', src)
        self.assertIn('rec["finish_order"]', src)
        # it is a MODEL of the queue, not a reading of it, and must not be
        # mixed with the measured outcomes
        self.assertIn("MODEL of the dynamic queue", src)

    def test_read_only_reports_do_not_create_stores(self):
        """Opening an HF store calls create_repo(exist_ok=True).

        condition_report.py created seven empty dataset repos the first time
        it ran, before the ablation had collected a cell -- and an empty repo
        is indistinguishable from a collected zero, a confusion already on the
        record for the three `-seeded` arm repos.
        """
        from opencap.store import StoreUnreachable, store_exists
        # It must never claim a store that is not there. It may legitimately
        # refuse to answer: without working credentials the hub returns the
        # same 401 for a private repo and a missing one, so False would be a
        # guess. Both outcomes are correct; True is not.
        try:
            self.assertFalse(
                store_exists("hf:savoji/capx-definitely-not-a-repo-xyz"))
        except StoreUnreachable:
            pass
        for name in ("condition_report.py", "tier_report.py", "sampled_report.py"):
            src = (self.ROOT / "scripts" / name).read_text()
            self.assertIn("store_exists", src, f"{name} opens stores unguarded")
            self.assertLess(src.index("store_exists("), src.index("open_store("),
                            f"{name} must check before it opens")

    def test_an_unreachable_hub_never_prints_as_a_null_result(self):
        """2026-10-27: a dead local token made every private store answer 401,
        store_exists swallowed it, and condition_report.py printed "not
        collected (store absent)" for all seven conditions of a B1 that was
        70/70 complete. An infrastructure failure must not read as a finding.
        """
        src = (self.ROOT / "opencap" / "store.py").read_text()
        # the no-answer case is an exception, not a False
        self.assertIn("raise StoreUnreachable", src)
        # "absent" is only returned after the credentials are known good,
        # because a private store we cannot see 401s exactly like a missing one
        self.assertLess(src.index("_check_credentials(spec)"),
                        src.index("        return False"),
                        "store_exists must verify credentials before saying no")
        for name in ("condition_report.py", "tier_report.py", "sampled_report.py"):
            src = (self.ROOT / "scripts" / name).read_text()
            self.assertIn("StoreUnreachable", src,
                          f"{name} must distinguish unreachable from uncollected")
            self.assertIn("says NOTHING about what was", src,
                          f"{name} must say the refusal is not a result")
        # and a store that exists but cannot be READ is not an empty store:
        # {} is reported as "exists, holds no cells", which is also a claim
        cond = (self.ROOT / "scripts" / "condition_report.py").read_text()
        self.assertNotIn("    except Exception:\n        return {}", cond)

    def test_the_chain_orders_jobs_by_server_state(self):
        """Starting the perception stack changes what else fits on the card,
        and B1 measures exactly that kind of thing. The servers-down jobs must
        all run before the stack comes up, and it must come up once."""
        sh = (self.ROOT / "scripts" / "run_all_experiments.sh").read_text()
        down = sh.index('JOBS_DOWN="A1 B1"')
        up = sh.index('JOBS_UP="A7 A8 A3 A2 A4"')
        self.assertLess(down, up)
        # A3 is NOT a servers-down job. It was listed as one, because
        # run_m1_arm.sh called M1 the privileged API; every `_multiturn`
        # config is privileged:false and declares three perception services.
        # The chain ran it with the stack down on 2026-10-15 and lost 7h20m.
        self.assertNotIn("A3", sh[down:sh.index("\n", down)])
        # A7 (S2) runs before A3, because M1 is the S2 API plus multi-turn and
        # A3 has nothing to subtract until S2 exists.
        jobs_up = sh[up:sh.index("\n", up)]
        self.assertLess(jobs_up.index("A7"), jobs_up.index("A3"))
        # A8 is the re-draw bound A2 needs: A4 ran cube_stack/S3, where every
        # greedy cell is 0/30, so it bounded movement at the floor only.
        self.assertIn("A8", jobs_up)
        # phase 1 loop before the stack, phase 2 after it
        start = sh.index("starting the perception stack")
        self.assertLess(sh.index("for J in $JOBS_DOWN"), start)
        self.assertLess(start, sh.index("for J in $JOBS_UP;\n", start)
                        if "for J in $JOBS_UP;\n" in sh[start:] else len(sh))
        self.assertIn("start_vision_servers.sh", sh)
        self.assertIn("start_molmo_server.sh", sh)

    def test_every_arm_checks_what_its_tier_declares(self):
        """The A3 bug generalised: no arm may decide from prose whether it
        needs the perception stack.

        Three separate places asserted M1 needed no servers -- run_m1_arm.sh
        ("privileged API"), run_tier_grid.sh ("text-only, no perception
        servers") and opencap.cells ("wave 1 needs no perception GPU") -- and all
        three were wrong. verify_servers.py reads the tier's OWN config and is
        a no-op for a tier that declares nothing, so calling it unconditionally
        is both correct and cheap. A tier added later is covered by default.
        """
        for name in ("run_m1_arm.sh", "run_tier_grid.sh", "run_s2_arm.sh",
                     "run_frame_arm.sh", "run_sampled_arm.sh"):
            sh = (self.ROOT / "scripts" / name).read_text()
            self.assertIn("verify_servers.py", sh,
                          f"{name} does not check its declared servers")
            # and every call passes its two positional arguments: called bare,
            # argparse exits non-zero, which reads as "servers down" -- that
            # bug kept A2 from ever starting.
            for line in sh.splitlines():
                if "verify_servers.py" in line and "python3" in line:
                    after = line.split("verify_servers.py", 1)[1]
                    self.assertTrue(after.strip() and not after.strip().startswith("|"),
                                    f"{name}: bare verify_servers.py call: {line}")
            # the refusal must stop the arm, not warn and continue
            self.assertIn("exit 1", sh)

    def test_m1_is_the_s2_api_not_the_privileged_one(self):
        """M1's comparator is S2. Pinning this because the file said S1 and
        the chain acted on it.

        Every `_multiturn` config is `privileged: false` on FrankaControlApi,
        so M1 - S1 moves privilege AND turns, and M1 - S3 moves API surface AND
        turns. Only M1 - S2 is one axis. S2 is also the rung Part A's ladder
        lacks: the headline S1 -> S3 collapse is the SUM of losing privileged
        state and dropping to a lower-level API, and nothing published so far
        can separate them.
        """
        m1 = (self.ROOT / "scripts" / "run_m1_arm.sh").read_text()
        self.assertIn("at S2", m1, "run_m1_arm.sh must point at the S2 comparator")
        self.assertNotIn("multi-turn, privileged API", m1)
        s2 = (self.ROOT / "scripts" / "run_s2_arm.sh").read_text()
        self.assertIn("privileged: false", s2)
        # and S2 must be collected into the SAME store as the other tiers, or
        # the ladder cannot be read cell by cell
        self.assertIn("hf:savoji/capx-ab-on-seeded", s2)
        # S2 needs the seeding check as much as S3/S4 did: artifact #13 was the
        # seeding patch reaching one simulator in four
        self.assertIn("verify_tier_simulators.py", s2)

    def test_deleting_a_cell_is_guarded_and_shows_its_work(self):
        """A cell collected under the wrong conditions is sticky: resume keeps
        it and nothing downstream can tell it from a good one. Deleting from a
        published corpus still must not be a one-liner.
        """
        src = (self.ROOT / "scripts" / "delete_cell.py").read_text()
        self.assertIn("--yes", src)              # dry run by default
        self.assertIn("DRY RUN", src)
        self.assertIn("_check_not_frozen", src)  # frozen corpora stay frozen
        self.assertIn("store_exists", src)       # never create what you delete from
        self.assertIn("StoreUnreachable", src)   # and never guess when offline
        # it must print the shard's contents before cutting: the premise is
        # that a bad cell is invisible from outside, so look first
        self.assertLess(src.index("describe(list(s.read_cell(cell)))"),
                        src.index("if not a.yes:"))

    def test_turns_used_is_populated_and_never_read_as_a_measurement(self):
        """A field declared with a default and assigned by nothing reads as a
        measurement of zero.

        `turns_used` was declared in opencap/schema.py with `= 0` and written by
        no code path, so it is 0 in all 2490 collected M1 episodes -- and
        HANDOFF told the pod to "report turns_used above all: if the loop never
        engaged, the success numbers are a re-run of S1". Followed literally
        that discards A3 on a field no code ever wrote. The loop DID engage:
        len(turns) > 1 in 785 of those episodes.
        """
        rc = (self.ROOT / "scripts" / "run_cell.py").read_text()
        self.assertIn("turns_used=len(_read_turns(", rc,
                      "run_cell.py must record the turn COUNT it observed")
        # derived from the same read, so the two cannot drift apart again
        self.assertIn("turns=_read_turns(", rc)
        m1 = (self.ROOT / "scripts" / "m1_report.py").read_text()
        # the report must prefer the turn list over the stored scalar, since
        # every already-collected episode has the dead 0
        self.assertIn("len(t) if isinstance(t, list) else 0", m1)
        self.assertIn("DEAD FIELD", m1)

    def test_a_cell_is_the_unit_not_a_scene(self):
        """One cell is ONE program run on 30 scenes, so 30 scene outcomes are
        30 looks at a single draw.

        Pooling scenes across cells reports a precision the design does not
        have. That mis-specification produced the Phase 11 worker verdict,
        which had to be withdrawn. Both new readouts lead with a cell-level
        test and label the scene-level one.
        """
        for name in ("frame_report.py", "m1_report.py"):
            src = (self.ROOT / "scripts" / name).read_text()
            self.assertIn("sign", src.lower(), f"{name} needs a cell-level test")
            self.assertIn("NOT independent", src,
                          f"{name} must label the scene-level number")
            # and the cell-level test must come first in the output
            self.assertLess(src.index("sign"), src.index("NOT independent"),
                            f"{name} prints the over-precise number first")

    def test_a_redraw_bound_is_not_borrowed_across_tasks(self):
        """A4 ran cube_stack at S3, where every greedy cell is 0/30.

        That bounds how far a re-draw moves a cell AT THE FLOOR, where a cell
        has nowhere to fall. It says nothing about a cell at 27/30, and all ten
        of A2's movers are on other tasks. Quoting the floor spread against
        them would manufacture a verdict.
        """
        src = (self.ROOT / "scripts" / "frame_report.py").read_text()
        self.assertIn("UNBOUNDED", src)
        self.assertIn("A4_TASK", src)
        self.assertIn("AT THE FLOOR", src)
        # and it must name the arm that WOULD bound them
        self.assertIn("OPENCAP_SAMPLED_TASK=cube_lifting", src)

    def test_the_sampled_report_and_arm_scope_stores_the_same_way(self):
        """If the report's store name and the arm's disagree, the report reads
        an empty store or the wrong arm, and either prints as a result.

        The first arm (nut_assembly/S1) predates scoping and owns the
        unsuffixed names, so that pair must stay unsuffixed on both sides.
        """
        sh = (self.ROOT / "scripts" / "run_sampled_arm.sh").read_text()
        py = (self.ROOT / "scripts" / "sampled_report.py").read_text()
        self.assertIn('nut_assembly', sh)
        self.assertIn('("nut_assembly", "S1")', py)
        # lowercased tier on both sides
        self.assertIn("tier.lower()", py)
        self.assertIn("tr '[:upper:]' '[:lower:]'", sh)
        # and the report must be able to read a non-default arm at all
        self.assertIn("--task", py)
        self.assertIn("--tier", py)

    def test_the_register_refuses_to_compute_a_drop_from_absent_fields(self):
        """Its A1 row first printed "repairs 899/900 -> 0/900, drop +99.9%,
        REFUTED" off two field names that do not exist in the file.

        `.get(name) or 0` turned a missing key into a measurement. The row now
        names the fields it requires and reports PENDING when they are absent.
        """
        src = (self.ROOT / "scripts" / "claims_register.py").read_text()
        self.assertIn("REQUIRED = (", src)
        self.assertIn("refusing to compute a drop from absent fields", src)
        blk = src[src.index("REQUIRED = ("):src.index("REQUIRED = (") + 2500]
        self.assertNotIn('r.get("heldout_k")', blk)
        # the real keys, as results/heldout_scenes.json carries them
        for f in ("in_sample", "in_sample_n", "heldout", "heldout_n"):
            self.assertIn(f'"{f}"', blk)

    def test_the_chain_verifies_the_stack_rather_than_assuming_it(self):
        sh = (self.ROOT / "scripts" / "run_all_experiments.sh").read_text()
        # The whole S4 sweep was thrown away once because only PyRoKi came up
        # and models scored 0/30 following a prompt the environment could not
        # satisfy. Starting the services is not the same as them working.
        self.assertIn("verify_servers.py", sh)
        self.assertIn("SKIPPED", sh)
        self.assertIn("86", sh)   # a distinct rc, not conflated with a failure

    def test_a_failing_job_does_not_abort_the_chain(self):
        sh = (self.ROOT / "scripts" / "run_all_experiments.sh").read_text()
        # Every job is independently resumable and none depends on another's
        # outcome -- a bad A1 changes how the corpus is described, not whether
        # A2-A4 are worth collecting.
        self.assertIn("RC[$J]=$?", sh)
        self.assertIn("does not abort", sh.lower().replace("nothing here aborts",
                                                            "does not abort"))

    def test_verify_servers_is_called_with_its_arguments(self):
        """It takes <task> <tier>. Called bare it exits on an argparse error,
        which run_frame_arm.sh read as 'servers down' and refused on -- so A2
        could never have started."""
        for name in ("run_frame_arm.sh", "run_all_experiments.sh"):
            sh = (self.ROOT / "scripts" / name).read_text()
            for line in sh.splitlines():
                if "verify_servers.py" in line and not line.lstrip().startswith("#"):
                    self.assertRegex(
                        line, r'verify_servers\.py\s+"?\$?\{?\w',
                        f"{name}: verify_servers.py called without task/tier")
