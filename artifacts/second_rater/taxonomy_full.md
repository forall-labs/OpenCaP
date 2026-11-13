# Tag taxonomy

18 tags in 6 families. Use the **tag** names in your CSV, not the family names.

Every example below is a real recorded cause from a program that is **not in
your packet**.

---

## FRAME — the arithmetic was right in the wrong coordinate frame

### `frame_convention`
A value is used in a different frame or component order than it is in. The
program's geometry may be perfectly sensible; it is applied to numbers that do
not mean what it assumes.

Covers: camera-frame values used as world coordinates; a quaternion in `wxyz`
handed to something expecting `xyzw` (or the reverse); pixel coordinates used
as metric positions; a transform that was computed and never applied.

- *cube points kept in camera frame and used as world coordinates*
- *grasp pose used in camera frame: pose_mat never applied*

---

## INTERFACE — the API was called wrongly

### `call_signature`
Wrong arity, wrong argument types or positions, duplicate keywords — **and
misreading what a call RETURNS**, such as indexing a list as a dict or
unpacking the wrong number of values.

- *plan_grasp given np.zeros_like(depth) as the segmentation — an empty mask*
- *depth from cam['depth']; SAM3's list indexed ['mask']*

### `object_name`
A wrong identifier for a scene object, including a wrong **text prompt** to a
perception call. The call is well-formed; the string does not name the thing.

- *underscored object strings 'green_cube' / 'red_cube'*

### `unbound_name`
A name used that is never bound: not imported, not defined, and not provided by
the API. Includes a function the program appears to assume exists.

- *time.sleep without importing time*
- *get_object_pose is not bound in this API — the program calls it anyway*

---

## SEMANTIC — it ran and did the wrong thing

### `missing_action`
An action the task requires is absent. The program's last required step was
never written — most often the final lift, placement or insertion.

- *the transit is 'move to a dummy position' (all-zero joints): no insertion*
- *no perception at all and no insertion*

### `missing_coverage`
A task needing an area or a sequence is done with too few points — a sweep
written as one or two moves, where iteration was required.

- *one move to the box centre: no sweep*

### `stale_read`
A pose is read, then something happens that could move the object, and the
**old** value is used afterwards. The fix re-reads it after the motion.

(No example outside your packet; the shape is: read pose → act → use the
pre-action pose.)

### `wrong_formula`
The arithmetic itself is wrong — a miscomputed offset, height, or a value
assembled from the wrong parts of a structure. Distinct from
`frame_convention`: here the numbers are in the right frame and the expression
over them is wrong.

- *9 flattened matrix entries passed as the quaternion*
- *a 3x3 applied to the first 3 ROWS of a 4x4 as 'position'*

### `wrong_target`
Acts on the wrong thing, or on the right thing's wrong property: the other
object's coordinates, an object's own centre where a grasp pose was needed, or
a computed value that is then ignored in favour of a constant.

- *the planned grasp is computed and never used: motion goes to the OBB centre*
- *target pose is a constant, not the planned grasp*

### `wrong_task`
Solves a different task than the one asked for — e.g. places an object when the
task was to lift it. Rare; use only when the whole objective is wrong, not when
one step is.

---

## STRUCTURE — it parsed and could never act

### `api_rebound`
Assigns over a provided API name, so later calls hit the program's own value.

- *redefines every API function as a wrapper that calls ITSELF — infinite
  recursion on first use*

### `gripper_order`
The gripper calls are sequenced so the object is never actually held — closing
where it should open, or releasing before the carry.

- *gripper sequence [close, open] → [open, close]: released after the lift*
- *opens at the 'grasp' and closes at the insertion point*

### `mock_shadowing`
**Defines** stub or placeholder versions of provided API functions, so the real
ones never run. Distinct from `api_rebound`: that is an assignment, this is a
`def`.

- *redefines solve_ik, move_to_joints and both gripper calls as placeholder
  stubs*

### `never_invoked`
Wraps the whole program in a function (or class) and never calls it, so nothing
executes.

- *defines pick_and_stack and never calls it*

---

## SYNTAX — it did not parse

### `syntax`
The program is not valid Python. In this corpus that is nearly always a
generation cut off mid-token. Tag it and move on; do not reconstruct.

- *ends mid-token inside a ~1900-character literal array (line 11)*
- *ends mid-token: '{' was never closed (line 410 of 413)*

---

## TRAJECTORY — the actions were right, the motion between them was not

### `approach_param`
An approach or clearance offset is wrong, missing, or applied to the wrong
motion: approaching from below, no clearance before a grasp, or a standoff
applied to the lift instead of the reach.

- *the grasp and placement 'approaches' sit 0.1 m BELOW their targets*
- *inserts to block_center - 0.05, i.e. down INTO the peg's mid-height*

### `post_release_retreat`
After releasing, the arm moves somewhere that undoes the result — reaching back
through what it just built.

(No example outside your packet; the shape is: open_gripper → a trailing motion
that disturbs the placed object.)

### `spurious_move`
An extra motion that should not be there at all, usually before acting on an
object — driving to the object's own pose, or a stray joint nudge.

- *carries the red cube to the green cube's own surface point before placing —
  straight into it*
- *'move down a bit' subtracts 0.2 rad from joint 0, swinging the arm sideways*

---

## Distinctions that come up repeatedly

| If the fix… | tag |
|---|---|
| transforms a value between frames, or reorders quaternion components | `frame_convention` |
| changes the expression, with frames already correct | `wrong_formula` |
| points the motion at a different object or property | `wrong_target` |
| changes how a call is made, or how its return is read | `call_signature` |
| adds the task's final step | `missing_action` |
| adds iteration over an area | `missing_coverage` |
| adds or corrects a clearance/standoff offset | `approach_param` |
| removes a motion entirely | `spurious_move` (or `post_release_retreat` if after release) |
| adds a call to a function the program defined | `never_invoked` |

An absent insertion or placement *recipe* is `missing_action`, not
`missing_coverage`. A missing pre-grasp approach is `approach_param`, not
`missing_action`.
