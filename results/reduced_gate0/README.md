# Gate 0 for the reduced API

Does a hand-written program reach the ceiling at S3/S4? Until that is
answered per task, a model's 0/30 at those tiers means nothing -- it could be
the model, the config, the reward, or an API that cannot express the task.

Run with `scripts/reduced_api_gate0.py <task> --code <file> --tier S3`.
No LLM: the program is injected as `oracle_code` and run on the same 30
seeded scenes, exactly as `verify_repairs.py` does for S1.

## Results

| file | task | tier | score | note |
|---|---|---|---|---|
| `cube_lifting_control_qwen35-27b.py` | cube_lifting | S3 | **27/30** | POSITIVE CONTROL -- reproduces the store exactly |
| `cube_stack_v1.py` | cube_stack | S3 | 20/18/17 (55/90) | API can express it; grasp noise dominates |
| `cube_stack_v2.py` | cube_stack | S3 | 23/18/20 (61/90) | NOT better than v1, p=0.436 |
| `spill_wipe_v1.py` | spill_wipe | S3 | 17/30 | 10x10 serpentine |
| **`spill_wipe_REFERENCE.py`** | spill_wipe | S3 | **30/30, 30/30, 30/30** | **AT CEILING -- 90/90 over three draws** |
| `probe_determinism.py` | -- | S3 | -- | isolates plan_grasp as the only stochastic stage |
| `cube_stack_v3.py` | cube_stack | S3 | 16/20/20 (56/90) | grasp-offset carried into placement; no gain over v2 |
| `cube_stack_v4.py` | cube_stack | S3 | 29/29/29 (87/90) | ANALYTIC top-down grasp, red height from the table; scene 24 fails every draw |
| `cube_stack_v5.py` | cube_stack | S3 | 29/29/29 | fingers across the red->green direction; scene 24 still |
| `cube_stack_v6.py` | cube_stack | S3 | 29/29/29 | red mask minus green, table z by mode; scene 24 still |
| **`cube_stack_v7.py`** | cube_stack | S3 | see STATUS | masks eroded 2 px, top face at the 95th percentile -- scene 24 was boundary pixels reading the green cube's depth |
| `nut_assembly_v2.py` | nut_assembly | S3 | 17/19/16 (52/90) | **Gate 0 CLEARED**: peg by molmo "square block", nut yaw from the handle axis, analytic handle grasp, ring squared to the peg |
| `nut_assembly_v3.py` | nut_assembly | S3 | 24/30 | wrist turned in 15 deg steps (solve_ik seeds from its last solution), compact-and-tall peg check |
| `nut_assembly_v4.py` | nut_assembly | S3 | 18/30 | in-hand visual correction -- REJECTED, the arm occludes the ring and biases it |
| `nut_assembly_v5.py` | nut_assembly | S3 | 25/25 (50/60) | IK seed reset to a fixed pose first: failures become deterministic |
| `nut_assembly_v6.py` | nut_assembly | S3 | 27/28 (55/60) | handle not toward the robot either; scenes 14, 21 fail every draw |
| `nut_assembly_v7.py` | nut_assembly | S3 | **28/28/28 (84/90)** | v6 + a post-release SEATED check (fresh obs) and one retry; the check is exact on 90 outcomes, the retry cannot regrasp a nut leaning on the peg -- 14, 21 still fail every draw |
| **`nut_assembly_v8.py`** | nut_assembly | S3 | **29/29/29 (87/90)** | v7 + grasp the handle 5 mm closer to the ring (the closing fingers leave the nut 1-5 mm further out): scene 21 fixed. Scene 14 fails every draw: its place IK lands 0.147 rad from a joint limit and stops 2.8 cm short; solve_ik's hidden seed cannot be reset from the program |

## spill_wipe is the first reduced-API task with a reference at ceiling

    qwen3vl-32b 5x5 comb (best model, pooled)   22/90   24.4% [16.7,34.2]
    reference v1, 10x10 serpentine              17/30   56.7% [39.2,72.6]
    reference v2, 14x14 + perpendicular pass    30/30  100.0% [88.6,100.0]
                                     fisher vs model p = 1.6e-14

The whole spill_wipe column is therefore a MODEL failure, not a capability
limit: the reduced API does the task perfectly while 21 models managed
8/630 at S3 and 0/630 at S4.

The separating defect is nameable. The best model program sweeps 5x5 = 25
waypoints in a comb that flies back across the spill each row; the reference
sweeps 14x14 plus a crossing pass = 392. That is `missing_coverage`, already
23% of the S1 defect mix, reappearing unchanged at S3 -- so part of the mix
carries across tiers rather than being replaced.

v1 -> v2 was diagnosed, not guessed: all 13 of v1's failures sat at reward
0.79-0.99 with four at 0.99, which is a wipe a few percent short of
threshold rather than a broken one.

**spill_wipe never grasps, so it is the one task where the S1 repair loop
applies unchanged** -- 30/30 bar, one re-draw, no pooling. Its 19 failing S3
programs are where the repair pass should start.

### Why the control comes first

`cube_lifting_control_qwen35-27b.py` is not hand-written -- it is the
qwen35-27b program that scored 27/30 in the grid. Injecting it here returned
27/30 with `sandbox_rc 0` on all 30 and 27 trials at reward exactly 1.000.
The instrument reproduces a known S3 result, so a zero it reports elsewhere
is a measurement rather than a broken harness.

### What the reduced API makes hard

`plan_grasp(depth, intrinsics, mask)` deprojects INTERNALLY, so cube_lifting
never touches 3D maths -- which is why that 27/30 program is 60 lines of
straightforward calls.

`get_oriented_bounding_box_from_3d_points(points)` takes `(N,3)` points the
caller must build, and the reduced API ships **no deprojection helper**. So
placing one cube on another requires hand-writing
mask -> pixels -> depth -> unproject -> world. Models reached for a
`get_3d_points_from_mask` that does not exist (artifact #16).

Note `control_reduced_skill_library.py` -- a different API class, NOT bound
at S3/S4 -- does provide `mask_to_world_points` and `pixel_to_world_point`.
The capability gap is real and was known upstream.

A further wrinkle the candidate has to handle: a single view sees only
visible surfaces, so an OBB's centre is biased toward the camera and its
depth-axis extent is wrong. Two quantities ARE well observed from above --
the top face's height and its horizontal size -- so the candidate uses a
robust max-z for the support height and the largest OBB extent for the
cube's side.
