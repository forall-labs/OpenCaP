# Tagging task: 60 robot-control programs and their repairs

You are the rater. This file is the complete brief; work only from it,
`taxonomy_full.md` and `packet.md`.

## What you are looking at

Each of 60 items is a **unified diff** from a Python program that a language
model wrote to control a simulated robot arm, to a corrected version of that
same program which was then verified to work in the simulator.

- Lines starting `-` are what the model wrote.
- Lines starting `+` are what the corrected version has instead.
- Context lines (no marker) are unchanged.

The programs call a provided robot API. **Different items use different API
sets** — some have high-level helpers like `goto_pose(...)`, others only have
perception plus `solve_ik` and `move_to_joints`. That is expected. Judge each
program against the API *it* uses, visible in the code.

Some diffs contain a collapsed marker for shared environment setup. **Never tag
collapsed/shared code.** It is setup that every correct program needs, not a
fault of this program.

## Your job

For each item, list **one tag per distinct cause the correction had to fix.**

A "distinct cause" is one independent thing that was wrong. If the correction
changes five lines to fix one mistake, that is **one** tag. If it fixes a wrong
coordinate frame *and* a missing final motion, that is **two** tags.

Rules:

1. **Only tags from `taxonomy_full.md`.** Do not invent tags. If nothing in the
   list fits, use the closest one and note it (see Output below).
2. **Tag what the correction actually changed**, not everything you believe is
   imperfect about the program. If the diff does not touch it, it is not a tag.
3. **One tag at most per tag name per item.** Two instances of the same mistake
   is still one tag.
4. **A tag may apply even if you would have fixed it differently.** The question
   is what cause the correction addressed, not whether it is the best fix.
5. **No minimum and no maximum.** Some items have one cause; some have six.
   Do not pad to a typical number, and do not compress to a tidy one.
6. **Work in order, one item at a time, and do not revise earlier answers**
   after seeing later items. Decide P01 before you read P02.

## Output

Write a CSV with exactly this header and 60 data rows, in packet order:

```
id,task,tags,notes
P01,cube_lifting,frame_convention;missing_action,
P02,cube_stack,approach_param,
```

- `id` and `task`: copy from the packet heading. Do not change them.
- `tags`: tag names separated by `;`, no spaces around the `;`. Empty if the
  correction fixed nothing attributable (rare — say why in `notes`).
- `notes`: free text, optional. Use it when you were torn between two tags,
  when nothing fit, or when the diff was unreadable. **Notes are read.** A
  recorded hesitation is more useful than a confident guess.

Save it as `answers.csv` in the directory you are working in. Do not go
looking for somewhere else to put it.

## Do not open these

Your tags are compared against an existing set. Reading any of the following
makes the comparison worthless, and a worthless comparison is worse than none
because it will be reported as agreement:

```
artifacts/second_rater/key.json
artifacts/second_rater/retag_session2.json
artifacts/second_rater/README.md
results/                      (any file)
STATUS.md
docs/
artifacts/cross-tier-defects.html
opencap/editclass.py
```

`taxonomy_full.md` and `packet.md` are the only project files you need.

**You have not been told what this measures or what result anyone expects, and
that is deliberate.** If you find yourself inferring a hypothesis from the
material, do not act on it: tag the diff in front of you. If you believe you
have seen the answer key or the study's conclusion, stop and say so — that is a
useful outcome, not a failure.

## Calibration: three worked examples

These three are **not** in the packet. They show the intended grain.

**Example A** — one cause, several lines.

```diff
-target_quat = grasp_world_frame[:4]
+target_quat = np.roll(Rotation.from_matrix(grasp_world_frame[:3, :3]).as_quat(), 1)
```
The program read a rotation out of a 4x4 matrix by slicing the wrong thing, and
the fix converts the rotation block to a quaternion in the API's component
order. That is **one** cause: the value was not in the form the API wants.
→ `wrong_formula`

**Example B** — two independent causes in one diff.

```diff
-points = cam["points"][mask]
-move_to_joints(solve_ik(points.mean(0), quat))
+points = (pose_mat @ to_homog(cam["points"][mask]).T).T[:, :3]
+move_to_joints(solve_ik(points.mean(0), quat))
+close_gripper()
+move_to_joints(solve_ik(points.mean(0) + np.array([0, 0, 0.1]), quat))
```
Two separate things were wrong: the points were used in the camera's frame
without being transformed into world coordinates, **and** the program never
closed the gripper or lifted. Frames and the missing ending are unrelated
mistakes.
→ `frame_convention;missing_action`

**Example C** — a change that is not a cause.

```diff
-pos = grasp[:3]
-quat = grasp[3:]
+grasp_pos = grasp[:3]
+grasp_quat = grasp[3:]
 move_to_joints(solve_ik(grasp_pos, grasp_quat))
```
Only the variable names changed. Nothing was wrong and nothing was fixed.
→ (no tags; note "rename only")

## If you get stuck

- **Can't tell what the API does?** Judge from how the corrected version uses
  it. The `+` lines are known to work.
- **Diff looks truncated / ends mid-token?** Tag `syntax` and move on. Do not
  try to reconstruct what the program would have said.
- **Correction is a near-total rewrite?** Tag the causes you can actually name
  from the diff, and say in `notes` that little of the original survived.
- **Two tags look like the same thing?** Read their entries in
  `taxonomy_full.md`; they are defined apart. Pick one and note the other.

Pace: roughly 2–4 minutes per item.
