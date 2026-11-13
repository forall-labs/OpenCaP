import sys
import unittest
from pathlib import Path

from opencap.editclass import FAMILY, PROVIDED, classify_edits
from opencap.failures import (EXECUTION, SEMANTIC, classify_root_cause, family,
                            summarise)


def ep(success=False, reward=0.0, stderr="", label=""):
    return {"success": success, "reward": reward, "failure_label": label,
            "turns": [{"summary_tail": stderr}] if stderr else []}


class TestExecutionCauses(unittest.TestCase):
    def test_missing_import_is_its_own_cause(self):
        # The qwen3-4b case: a correct program that lost 30/30 to one line.
        # It must not be lumped with genuine API misuse -- the remedy is a
        # prompt change, not better API grounding.
        e = ep(stderr="NameError: name 'np' is not defined", reward=0.525)
        self.assertEqual(classify_root_cause(e), "missing_import")

    def test_other_undefined_names_are_distinct(self):
        e = ep(stderr="NameError: name 'grasp_object' is not defined")
        self.assertEqual(classify_root_cause(e), "undefined_name")

    def test_syntax_error(self):
        # The Qwen3 <think>-token case.
        e = ep(stderr="SyntaxError: unterminated string literal")
        self.assertEqual(classify_root_cause(e), "syntax")

    def test_wrong_signature_and_bad_index(self):
        self.assertEqual(classify_root_cause(
            ep(stderr="TypeError: goto_pose() missing 1 required argument")),
            "wrong_signature")
        self.assertEqual(classify_root_cause(
            ep(stderr="IndexError: invalid index to scalar variable")),
            "bad_index")

    def test_execution_beats_reward_heuristic(self):
        # An exception is decisive even when partial reward was earned.
        e = ep(stderr="NameError: name 'np' is not defined", reward=0.9)
        self.assertEqual(classify_root_cause(e), "missing_import")


class TestSemanticCauses(unittest.TestCase):
    def test_no_op_vs_incomplete(self):
        # The distinction that separates "cannot act" from "cannot finish":
        # 8B grasped the cube (reward ~0.5) and never wrote the lift.
        self.assertEqual(classify_root_cause(ep(reward=0.001)), "no_op")
        self.assertEqual(classify_root_cause(ep(reward=0.50)), "incomplete")

    def test_marginal_band(self):
        self.assertEqual(classify_root_cause(ep(reward=0.12)), "marginal")

    def test_success_short_circuits(self):
        self.assertEqual(classify_root_cause(ep(success=True, reward=1.0)),
                         "success")


class TestFamilies(unittest.TestCase):
    def test_family_partition(self):
        self.assertEqual(family("missing_import"), "execution")
        self.assertEqual(family("incomplete"), "semantic")
        self.assertEqual(family("success"), "success")
        self.assertFalse(EXECUTION & SEMANTIC)

    def test_missing_stderr_is_not_silently_semantic(self):
        # rc!=0 with no captured stderr must not be mislabelled as a
        # robot-control failure -- 13 early shards have no turns at all.
        e = ep(label="api_misuse", reward=0.4)
        self.assertEqual(classify_root_cause(e), "unknown_execution")
        self.assertEqual(family("unknown_execution"), "execution")

    def test_summarise_counts(self):
        eps = [ep(success=True), ep(reward=0.5),
               ep(stderr="NameError: name 'np' is not defined")]
        self.assertEqual(summarise(eps),
                         {"success": 1, "incomplete": 1, "missing_import": 1})


class TestBalancedPanel(unittest.TestCase):
    """A model carrying extra hard-tier cells looks worse for reasons that have
    nothing to do with its size, so the profile must compare matched cells."""

    def test_only_shared_tier_task_pairs_are_counted(self):
        from pathlib import Path
        src = (Path(__file__).resolve().parent.parent
               / "scripts" / "failure_profile.py").read_text()
        self.assertIn("set.intersection", src)
        self.assertIn("dropped", src)

    def test_intersection_drops_unmatched(self):
        per_model = {
            "a": {("S1", "lift"), ("S1", "stack"), ("S4", "lift")},
            "b": {("S1", "lift"), ("S1", "stack")},
            "c": {("S1", "lift"), ("S1", "stack")},
        }
        shared = set.intersection(*per_model.values())
        self.assertEqual(shared, {("S1", "lift"), ("S1", "stack")})
        self.assertNotIn(("S4", "lift"), shared)

class TestCoverageGuard(unittest.TestCase):
    def test_zero_stderr_corpus_is_refused(self):
        # All 39 cells of the first corpus predate the `turns` fix: 0/1170
        # episodes carry stderr, so every failure classifies as
        # unknown_execution and the table reads as a scaling trend while
        # being a data gap. The profiler must refuse rather than print it.
        from pathlib import Path
        src = (Path(__file__).resolve().parent.parent
               / "scripts" / "failure_profile.py").read_text()
        self.assertIn("MIN_STDERR_COVERAGE", src)
        self.assertIn("REFUSING to profile", src)

    def test_unknown_execution_needs_a_label_not_just_silence(self):
        # A clean run with no exception and low reward is a real semantic
        # no_op, not missing data -- the guard must not swallow it.
        self.assertEqual(classify_root_cause(ep(reward=0.0)), "no_op")


class TestNoRevision(unittest.TestCase):
    def _multi(self, codes, reward=0.5):
        return {"success": False, "reward": reward, "failure_label": "",
                "turns": [{"code": c} for c in codes]}

    def test_identical_code_across_turns_is_its_own_cause(self):
        # qwen3-8b on cube_lifting: same program three turns running, never
        # writes the lift. `incomplete` would prescribe longer horizons; the
        # real problem is that feedback changed nothing.
        e = self._multi(["grasp()\nclose()"] * 3)
        self.assertEqual(classify_root_cause(e), "no_revision")
        self.assertEqual(family("no_revision"), "semantic")

    def test_revised_code_is_not_flagged(self):
        e = self._multi(["grasp()", "grasp()\nlift()"])
        self.assertEqual(classify_root_cause(e), "incomplete")

    def test_single_turn_is_never_no_revision(self):
        # One turn cannot demonstrate a failure to revise.
        self.assertEqual(classify_root_cause(self._multi(["grasp()"])),
                         "incomplete")

    def test_exception_still_wins(self):
        e = {"success": False, "reward": 0.5, "failure_label": "",
             "turns": [{"code": "x", "summary_tail": "NameError: name 'np' is not defined"},
                       {"code": "x", "summary_tail": "NameError: name 'np' is not defined"}]}
        self.assertEqual(classify_root_cause(e), "missing_import")


class TestRevisionProfile(unittest.TestCase):
    def test_turn_codes_drops_empty_turns(self):
        import importlib.util
        from pathlib import Path
        p = Path(__file__).resolve().parent.parent / "scripts" / "revision_profile.py"
        spec = importlib.util.spec_from_file_location("rp", p)
        rp = importlib.util.module_from_spec(spec); spec.loader.exec_module(rp)
        # An empty turn must not make two identical programs look like three,
        # nor a single program look multi-turn.
        self.assertEqual(rp.turn_codes({"turns": [{"code": "a"}, {"code": "  "}]}),
                         ["a"])
        self.assertEqual(rp.turn_codes({"turns": []}), [])
        self.assertEqual(rp.turn_codes({}), [])


class TestCliffPairing(unittest.TestCase):
    def test_pairs_on_task_and_state_not_state_alone(self):
        # state_id is a per-task trial index, so pooling tasks under a bare
        # index pairs cube_lifting trial 7 against spill_wipe trial 7 and
        # reports a cliff that is really a task difference.
        from pathlib import Path
        src = (Path(__file__).resolve().parent.parent
               / "scripts" / "cliff_report.py").read_text()
        self.assertIn('d[(c.task, e["state_id"])]', src)

    def test_collision_would_lose_states(self):
        # Demonstrates the bug the keying avoids: two tasks x 3 trials is six
        # distinct cells, but only three keys if task is dropped.
        eps = [("lift", 0), ("lift", 1), ("lift", 2),
               ("wipe", 0), ("wipe", 1), ("wipe", 2)]
        self.assertEqual(len({k for k in eps}), 6)
        self.assertEqual(len({s for _, s in eps}), 3)


class TestSummaryParsing(unittest.TestCase):
    def _summary(self, tmp, triple):
        from pathlib import Path
        d = Path(tmp); (d / "summaries.txt").write_text(
            "Model: X\nTotal number of trials: 30\n"
            "Code generation success rate / Average reward / Task completed: \n"
            f"{triple}\nAverage code blocks: 1.000\n"
            "Average regenerations: 0.000\n")
        return d

    def test_negative_average_reward_parses(self):
        # Qwen2.5-Coder-7B returned 1.000/-10.143/0. The original [\d.]+ could
        # not match a leading minus, so a well-formed file parsed as None and
        # the cell was reported as "no summaries.txt".
        import tempfile
        from opencap.results import parse_summary
        with tempfile.TemporaryDirectory() as tmp:
            s = parse_summary(self._summary(tmp, "1.000/-10.143/0"))
            self.assertIsNotNone(s)
            self.assertAlmostEqual(s.avg_reward, -10.143)
            self.assertEqual(s.trials, 30)
            self.assertEqual(s.completed, 0)

    def test_positive_still_parses(self):
        import tempfile
        from opencap.results import parse_summary
        with tempfile.TemporaryDirectory() as tmp:
            s = parse_summary(self._summary(tmp, "1.000/0.517/12"))
            self.assertAlmostEqual(s.avg_reward, 0.517)
            self.assertEqual(s.completed, 12)

    def test_unparseable_is_not_reported_as_missing(self):
        import tempfile
        from pathlib import Path
        from opencap.results import cross_check
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "summaries.txt").write_text("garbage, no triple\n")
            ok, msg = cross_check(tmp)
            self.assertFalse(ok)
            self.assertIn("unparseable", msg)

    def test_missing_is_still_reported_as_missing(self):
        import tempfile
        from opencap.results import cross_check
        with tempfile.TemporaryDirectory() as tmp:
            ok, msg = cross_check(tmp)
            self.assertFalse(ok)
            self.assertEqual(msg, "no summaries.txt")


class TestTrialDirParsing(unittest.TestCase):
    def test_negative_reward_dir_is_not_dropped(self):
        # The silent-subset bug: 16 of 30 trial dirs had a negative reward and
        # matched nothing, so parse_trials returned 14 and any rate computed
        # from it would be over a biased sample.
        from opencap.results import TRIAL_DIR
        m = TRIAL_DIR.match(
            "trial_07_sandboxrc_0_reward_-10.143_taskcompleted_0")
        self.assertIsNotNone(m)
        self.assertEqual(m.group("trial"), "07")
        self.assertEqual(float(m.group("reward")), -10.143)

    def test_positive_reward_dir_still_matches(self):
        from opencap.results import TRIAL_DIR
        m = TRIAL_DIR.match(
            "trial_01_sandboxrc_0_reward_0.497_taskcompleted_0")
        self.assertIsNotNone(m)
        self.assertEqual(float(m.group("reward")), 0.497)

    def test_malformed_dir_still_rejected(self):
        from opencap.results import TRIAL_DIR
        self.assertIsNone(TRIAL_DIR.match("trial_01_reward_0.5"))
        self.assertIsNone(TRIAL_DIR.match("scratch_notes"))


class TestNeverInvoked(unittest.TestCase):
    def _ep(self, code, reward=0.0):
        return {"success": False, "reward": reward, "failure_label": "",
                "turns": [{"code": code}]}

    def test_uncalled_function_is_its_own_cause(self):
        # qwen3-4b, 30/30 S4 cube_stack episodes: a complete pick-and-stack
        # program sealed in a function that is never called. Scores 0 with no
        # exception, so it is indistinguishable from "tried and failed".
        e = self._ep("import numpy as np\ndef pick():\n    open_gripper()\n")
        self.assertEqual(classify_root_cause(e), "never_invoked")
        self.assertEqual(family("never_invoked"), "execution")

    def test_called_function_is_not_flagged(self):
        e = self._ep("def pick():\n    open_gripper()\npick()\n")
        self.assertNotEqual(classify_root_cause(e), "never_invoked")

    def test_top_level_code_alongside_def_is_not_flagged(self):
        # A helper plus real top-level work does run.
        e = self._ep("def helper():\n    return 1\nopen_gripper()\n")
        self.assertNotEqual(classify_root_cause(e), "never_invoked")

    def test_no_functions_at_all_is_not_flagged(self):
        self.assertNotEqual(
            classify_root_cause(self._ep("open_gripper()\n")), "never_invoked")

    def test_syntax_error_takes_precedence(self):
        e = {"success": False, "reward": 0.0, "failure_label": "",
             "turns": [{"code": "def f(:\n  pass\n",
                        "summary_tail": "SyntaxError: invalid syntax"}]}
        self.assertEqual(classify_root_cause(e), "syntax")

    def test_success_still_short_circuits(self):
        e = self._ep("def pick():\n    pass\n")
        e["success"] = True
        self.assertEqual(classify_root_cause(e), "success")


class TestCliffTrendExclusion(unittest.TestCase):
    def test_zero_s1_rungs_are_excluded_from_trend(self):
        # A rung scoring 0/30 at S1 has A(m)=0 for a structural reason. Left in
        # the trend it manufactures "the cliff grows with scale" from
        # arithmetic: A(0.6B)=0 -> A(4B)=+0.62 is not a measured slope.
        from pathlib import Path
        src = (Path(__file__).resolve().parent.parent
               / "scripts" / "cliff_report.py").read_text()
        self.assertIn("if a > 0:", src)
        self.assertIn("by construction", src)

    def test_trend_needs_two_readable_rungs(self):
        from pathlib import Path
        src = (Path(__file__).resolve().parent.parent
               / "scripts" / "cliff_report.py").read_text()
        self.assertIn("need >=2 to read", src)


class TestEntryPointPatchRules(unittest.TestCase):
    """The patch fires inside CaP-X's executor, where a wrong guess would be
    indistinguishable from a capability difference in the results. These pin
    the conditions under which it is allowed to act."""

    @staticmethod
    def _candidates(code):
        import ast
        t = ast.parse(code)
        defs = [n for n in t.body if isinstance(n, ast.FunctionDef)
                and not n.args.args and not n.args.posonlyargs
                and not n.args.kwonlyargs]
        called = {n.func.id for n in ast.walk(t)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        runnable = [n for n in t.body
                    if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef,
                                          ast.Import, ast.ImportFrom, ast.ClassDef))]
        cands = [d for d in defs if d.name not in called]
        return cands, runnable

    def _fires(self, code):
        c, r = self._candidates(code)
        return len(c) == 1 and not r

    def test_fires_on_the_observed_case(self):
        self.assertTrue(self._fires(
            "import numpy as np\ndef pick_and_stack():\n    open_gripper()\n"))

    def test_does_not_fire_when_already_called(self):
        self.assertFalse(self._fires("def go():\n    pass\ngo()\n"))

    def test_does_not_fire_with_top_level_code(self):
        self.assertFalse(self._fires("def helper():\n    pass\nopen_gripper()\n"))

    def test_does_not_fire_on_two_candidates(self):
        # Picking one of two would be a guess, and a guess in the executor is
        # indistinguishable from a capability difference downstream.
        self.assertFalse(self._fires("def a():\n    pass\ndef b():\n    pass\n"))

    def test_does_not_fire_on_a_function_needing_args(self):
        self.assertFalse(self._fires("def go(env):\n    pass\n"))

    def test_ambiguity_biases_the_effect_down_not_up(self):
        # Every non-firing case keeps scoring 0, so the measured recovery is a
        # lower bound on the prompt-shaped share of the cliff.
        for code in ("def a():\n    pass\ndef b():\n    pass\n",
                     "def go(env):\n    pass\n"):
            self.assertFalse(self._fires(code))


class TestServerPrecondition(unittest.TestCase):
    """The guard that was missing when the S4 sweep ran without SAM3 and
    Contact-GraspNet. Every other guard checks a run's output; this one checks
    that the environment a cell needs exists before the cell runs."""

    @staticmethod
    def _parse(text):
        import importlib.util
        from pathlib import Path
        p = Path(__file__).resolve().parent.parent / "scripts" / "verify_servers.py"
        spec = importlib.util.spec_from_file_location("vs", p)
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        return m.declared_ports(text)

    S4_CFG = """env:
  _target_: capx.envs.tasks.franka.franka_lift.FrankaLiftCodeEnv
api_servers:
  - _target_: capx.serving.launch_sam3_server.main
    device: cuda
    port: 8114
    host: 127.0.0.1
  - _target_: capx.serving.launch_contact_graspnet_server.main
    port: 8115
    host: 127.0.0.1
  - _target_: capx.serving.launch_pyroki_server.main
    port: 8116
    host: 127.0.0.1
record_video: true
"""

    def test_finds_every_declared_server(self):
        got = self._parse(self.S4_CFG)
        self.assertEqual(got, [("sam3", 8114), ("contact_graspnet", 8115),
                               ("pyroki", 8116)])

    def test_stops_at_end_of_block(self):
        # record_video sits after the block at column 0; a parser that ran on
        # would attach later ports to the wrong server.
        self.assertEqual(len(self._parse(self.S4_CFG)), 3)

    def test_config_without_servers_is_not_flagged(self):
        self.assertEqual(self._parse("env:\n  _target_: x\nrecord_video: true\n"), [])


class TestRunCellEnforcesPrecondition(unittest.TestCase):
    def test_server_failure_is_detected_after_the_run_not_before(self):
        # A pre-run port check refuses EVERY cell: launch.py starts the API
        # servers itself, so the port is legitimately closed until it does.
        # The first version of this guard blocked a whole sweep. Detection has
        # to read the finished run instead.
        from pathlib import Path
        src = (Path(__file__).resolve().parent.parent
               / "scripts" / "run_cell.py").read_text()
        self.assertNotIn("verify_servers.py", src)
        self.assertIn("Connection refused", src)
        # Anchor on the launch target, not the exact argv: flags such as
        # --no-sync sit between "run" and the script and are not the point.
        self.assertGreater(src.index("Connection refused"),
                           src.index('"capx/envs/launch.py"'))

    def test_server_failure_is_not_recorded_as_a_model_result(self):
        from pathlib import Path
        import re
        src = (Path(__file__).resolve().parent.parent
               / "scripts" / "run_cell.py").read_text()
        # The message is one sentence split across adjacent f-string literals
        # ("... environment " f"failure, not a model result"); join them, or a
        # line re-wrap fails a test whose point is that the sentence exists.
        joined = re.sub(r'"\s*\n\s*f?"', "", src)
        self.assertIn("environment failure, not a model result", joined)


class TestEntrypointMarkerBeatsAST(unittest.TestCase):
    """In the entry-point arm the source still holds an uncalled def, so a
    code-structure test keeps reporting never_invoked for episodes that
    demonstrably ran. The runtime marker has to win."""

    CODE = "import numpy as np\ndef pick():\n    open_gripper()\n"

    def test_marker_suppresses_never_invoked(self):
        e = {"success": False, "reward": 0.51, "failure_label": "",
             "turns": [{"code": self.CODE,
                        "summary_tail": "Stdout: [opencap] invoking never-called "
                                        "entry point: pick"}]}
        self.assertNotEqual(classify_root_cause(e), "never_invoked")
        self.assertEqual(classify_root_cause(e), "incomplete")

    def test_without_marker_still_flags(self):
        e = {"success": False, "reward": 0.51, "failure_label": "",
             "turns": [{"code": self.CODE}]}
        self.assertEqual(classify_root_cause(e), "never_invoked")

    def test_exception_outranks_marker(self):
        e = {"success": False, "reward": 0.0, "failure_label": "",
             "turns": [{"code": self.CODE,
                        "summary_tail": "[opencap] invoking never-called entry "
                                        "point: pick\nValueError: bad shape"}]}
        self.assertEqual(classify_root_cause(e), "bad_value")


class TestMockShadowing(unittest.TestCase):
    """gemma3-12b wrote a self-contained 'testable script' with stub versions of
    goto_pose/get_object_pose in a __main__ block. Those stubs shadow the
    injected APIs, so the program ran against its own print statements and never
    reached the robot -- while looking, in the trace, like ordinary code."""

    def _ep(self, code, reward=0.0):
        return {"success": False, "reward": reward, "failure_label": "",
                "turns": [{"code": code}]}

    def test_stub_definition_is_detected(self):
        from opencap.failures import shadowed_apis
        self.assertEqual(shadowed_apis("def goto_pose(p, q):\n    print(p)\n"),
                         {"goto_pose"})

    def test_rebinding_by_assignment_is_detected(self):
        # qwen3-0.6b did this: sample_grasp_pose = sample_grasp_pose("cube")
        from opencap.failures import shadowed_apis
        self.assertEqual(
            shadowed_apis('sample_grasp_pose = sample_grasp_pose("red cube")\n'),
            {"sample_grasp_pose"})

    def test_clean_code_is_not_flagged(self):
        from opencap.failures import shadowed_apis
        self.assertEqual(shadowed_apis("import numpy as np\ngoto_pose(a, b)\n"),
                         set())

    def test_helper_with_a_different_name_is_fine(self):
        from opencap.failures import shadowed_apis
        self.assertEqual(shadowed_apis("def my_helper():\n    pass\n"), set())

    def test_classified_as_execution_family(self):
        e = self._ep("def goto_pose(p, q, z_approach=0):\n    print(p)\n"
                     "goto_pose(1, 2)\n")
        self.assertEqual(classify_root_cause(e), "mock_shadowing")
        self.assertEqual(family("mock_shadowing"), "execution")

    def test_syntax_error_still_outranks(self):
        e = {"success": False, "reward": 0.0, "failure_label": "",
             "turns": [{"code": "def goto_pose(:\n  pass\n",
                        "summary_tail": "SyntaxError: invalid syntax"}]}
        self.assertEqual(classify_root_cause(e), "syntax")

    def test_success_short_circuits(self):
        e = self._ep("def goto_pose(p, q):\n    pass\n")
        e["success"] = True
        self.assertEqual(classify_root_cause(e), "success")


class TestUnboundName(unittest.TestCase):
    """The tag that mis-fired for a whole phase.

    `unbound_name` is INTERFACE, so every false positive inflates the family
    that the >=12B band was reported to be dominated by. Both false-positive
    sources below were live in the published tagging: nine programs were tagged
    for calling a primitive the harness binds, and three for importing scipy.
    """

    @staticmethod
    def tags(original, repaired="x = 1\n"):
        return [d.tag for d in classify_edits(original, repaired)]

    def test_imported_names_are_bound(self):
        # `from scipy.spatial.transform import Rotation as R` binds R. The walk
        # collected ast.Name and nothing else, so every scipy import read as a
        # defect -- hidden because the only import most programs write is numpy,
        # which was excluded by hand.
        src = ("from scipy.spatial.transform import Rotation as R\n"
               "q = R.from_quat([0, 0, 0, 1])\n")
        self.assertNotIn("unbound_name", self.tags(src))

    def test_plain_and_dotted_imports_are_bound(self):
        src = "import math\nimport os.path\nx = math.pi\ny = os.path.sep\n"
        self.assertNotIn("unbound_name", self.tags(src))

    def test_harness_primitive_is_not_a_hallucination(self):
        # goto_home_joint_position is bound at runtime: the human reference
        # calls it, 21 verified repairs call it, and the nine model programs
        # that call it raised no NameError in 270 episodes. Tagging it
        # `unbound_name` scored correct API use as a defect.
        self.assertIn("goto_home_joint_position", PROVIDED)
        self.assertNotIn("unbound_name",
                         self.tags("goto_home_joint_position()\n"))

    def test_preimported_numpy_is_bound_without_an_import(self):
        # OPENCAP_PREIMPORT_NUMPY is set in the baseline arm, which is the corpus
        # everything here is tagged from.
        self.assertNotIn("unbound_name", self.tags("x = np.array([0, 0, 1])\n"))

    def test_a_real_undefined_name_still_tags(self):
        # qwen3-0.6b's get_fixed_rigid_transform, which genuinely does not
        # exist -- it raised NameError in the corpus. The fixes above must not
        # cost us this one.
        src = "T = get_fixed_rigid_transform('nut', 'peg')\n"
        self.assertIn("unbound_name", self.tags(src))
        self.assertEqual(FAMILY["unbound_name"], "INTERFACE")

    def test_an_unset_local_still_tags(self):
        src = "place_z = green_center_z + 0.02\n"
        self.assertIn("unbound_name", self.tags(src))


class TestFrameConvention(unittest.TestCase):
    """The class the taxonomy had no name for.

    Five programs with a verified non-zero-distance repair carried no tag at
    all, and eight of the ten untagged programs were >=8B: the tagger went
    silent exactly where the interesting models are.
    """

    @staticmethod
    def tags(original, repaired):
        return [d.tag for d in classify_edits(original, repaired)]

    def test_quaternion_convention_tags(self):
        # qwen35-27b and qwen3-32b both hand scipy a wxyz quaternion. It takes
        # xyzw, and the human reference does the reorder by hand.
        before = ("from scipy.spatial.transform import Rotation as R\n"
                  "rot = R.from_quat(block_quat)\n")
        after = ("from scipy.spatial.transform import Rotation as R\n"
                 "rot = R.from_quat(block_quat[[1, 2, 3, 0]])\n")
        self.assertIn("frame_convention", self.tags(before, after))
        self.assertEqual(FAMILY["frame_convention"], "FRAME")

    def test_offset_rotated_into_the_target_frame_tags(self):
        # qwen3vl-32b: the handle offset applied in world coordinates.
        before = ("rot = quat_to_rot(handle_quat)\n"
                  "target = block_pos - rel_pos\n")
        after = ("rot = quat_to_rot(handle_quat)\n"
                 "target = block_pos - quat_to_rot(block_quat) @ rot.T @ rel_pos\n")
        self.assertIn("frame_convention", self.tags(before, after))

    def test_a_program_that_never_attempted_the_transform_does_not_tag(self):
        # THE guard. Every nut_assembly repair adds the same insertion recipe,
        # so without requiring the original to have attempted a transform this
        # fires on 21 of 21 programs on that task and 0 of 49 everywhere else --
        # identifying the task, not the defect. Seventeen of those programs
        # never write frame algebra at all; their missing insertion is already
        # missing_action.
        before = "goto_pose(nut_pos, nut_quat)\nclose_gripper()\n"
        after = ("from scipy.spatial.transform import Rotation as R\n"
                 "goto_pose(nut_pos, nut_quat)\nclose_gripper()\n"
                 "off = R.from_quat(peg_quat[[1, 2, 3, 0]]).apply(rel)\n"
                 "goto_pose(peg_pos + off, nut_quat)\n")
        self.assertNotIn("frame_convention", self.tags(before, after))

    def test_unchanged_frame_algebra_does_not_tag(self):
        src = ("from scipy.spatial.transform import Rotation as R\n"
               "rot = R.from_quat(q[[1, 2, 3, 0]])\n")
        self.assertNotIn("frame_convention", self.tags(src, src + "x = 1\n"))


class TestProvidedApiVerifier(unittest.TestCase):
    """PROVIDED must be read off the harness, not remembered.

    Nothing in the pipeline could catch a wrong PROVIDED: the tagger sees a
    program and a repair, and neither knows what the harness binds.
    """

    def _checkout(self, body):
        import tempfile
        d = Path(tempfile.mkdtemp())
        (d / "capx").mkdir()
        (d / "capx" / "base.py").write_text(body)
        self.addCleanup(__import__("shutil").rmtree, d)
        return d

    def test_reads_the_names_out_of_the_binding_site(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "vpa", Path(__file__).resolve().parent.parent
            / "scripts" / "verify_provided_api.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["vpa"] = mod
        spec.loader.exec_module(mod)
        d = self._checkout(
            "class E:\n"
            "    def _init_exec_globals(self):\n"
            "        self._exec_globals['goto_pose'] = self.goto_pose\n"
            "        self._exec_globals['brand_new_primitive'] = self.x\n")
        found, where = mod.injected_names(d)
        self.assertEqual(len(where), 1)
        self.assertIn("goto_pose", found)
        # The point of a structural read: a primitive nobody here has heard of
        # still shows up, which a grep for expected names would never do.
        self.assertIn("brand_new_primitive", found)
        self.assertTrue(found - set(PROVIDED))


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestTruncationDetector(unittest.TestCase):
    """Artifact #15's corpus-side half.

    A program cut at the token cap enters the store with tokens_out = 0,
    truncation_fired false, and a no_op label -- indistinguishable from one the
    model finished and got wrong. Five of the eight cut-off programs in the
    seeded corpus PARSE, because the cut landed inside a trailing comment, so a
    syntax check alone finds none of them.
    """

    def test_a_cut_inside_a_comment_still_counts(self):
        from opencap.repair import looks_truncated
        # qwen3.5-0.8B nut_assembly: 114 lines ending "# We can". Parses fine.
        code = "\n".join(["x = %d" % i for i in range(120)] + ["# We can"])
        self.assertTrue(looks_truncated(code))

    def test_a_cut_mid_statement_counts(self):
        from opencap.repair import looks_truncated
        # qwen3.5-2B cube_stack: ends "goto_pose(green".
        self.assertTrue(looks_truncated("import numpy as np\ngoto_pose(green"))

    def test_an_ordinary_program_does_not(self):
        from opencap.repair import looks_truncated
        self.assertFalse(looks_truncated(
            "import numpy as np\np, q = sample_grasp_pose('red cube')\n"
            "goto_pose(p, q, z_approach=0.05)\nclose_gripper()\n"))

    def test_empty_is_not_truncated(self):
        from opencap.repair import looks_truncated
        # An empty program is a timed-out request, a different artifact (#11)
        # with its own handling. Do not relabel it as this one.
        self.assertFalse(looks_truncated(""))
        self.assertFalse(looks_truncated("   \n\n"))

    def test_the_threshold_is_outside_the_real_distribution(self):
        from opencap.repair import TRUNCATION_LINES
        # Corpus programs that parse and end on a statement run 5-83 lines,
        # median 22. A threshold inside that range would flag real programs.
        self.assertGreaterEqual(TRUNCATION_LINES, 90)
