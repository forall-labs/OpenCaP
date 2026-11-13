import numpy as np
from scipy.spatial.transform import Rotation
obs = get_observation()
cam = obs["robot0_robotview"]
rgb, depth = cam["images"]["rgb"], cam["images"]["depth"]
K, E = cam["intrinsics"], cam["pose_mat"]
pr = point_prompt_molmo(rgb, "red cube")
pc = pr.get("red cube", (None, None))
if pc is None or tuple(pc) == (None, None):
    raise RuntimeError("molmo could not find the red cube")
res = segment_sam3_point_prompt(rgb, pc)
if not res:
    raise RuntimeError("sam3 returned no mask")
mask = max(res, key=lambda d: d["score"])["mask"]
poses, scores = plan_grasp(depth, K, mask)
if len(poses) == 0:
    raise RuntimeError("no grasp candidates")
T = E @ poses[int(np.argmax(scores))]
gp = T[:3, 3]
gq = np.roll(Rotation.from_matrix(T[:3, :3]).as_quat(), 1)   # xyzw -> wxyz
move_to_joints(solve_ik(gp + np.array([0.0, 0.0, 0.10]), gq))
open_gripper()
move_to_joints(solve_ik(gp, gq))
close_gripper()
move_to_joints(solve_ik(gp + np.array([0.0, 0.0, 0.15]), gq))
