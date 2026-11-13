import numpy as np
from scipy.spatial.transform import Rotation

obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
E = cam["pose_mat"]          # camera -> world


def locate(name):
    """molmo point -> sam3 point prompt -> best mask."""
    pr = point_prompt_molmo(rgb, name)
    pc = pr.get(name, (None, None))
    if pc is None or tuple(pc) == (None, None):
        raise RuntimeError("molmo could not find " + name)
    res = segment_sam3_point_prompt(rgb, pc)
    if not res:
        raise RuntimeError("sam3 returned no mask for " + name)
    return max(res, key=lambda d: d["score"])["mask"]


def mask_to_world(mask):
    """The deprojection the reduced API does NOT provide.

    Same convention as graspnet._depth_to_pointcloud and the skill library's
    mask_to_world_points: x=(u-cx)z/fx, y=(v-cy)z/fy, then extrinsics @ p.
    """
    m = mask[:, :, 0] if mask.ndim == 3 else mask
    ys, xs = np.where(m > 0)
    if len(ys) == 0:
        raise RuntimeError("empty mask")
    z = np.asarray(depth)[ys, xs].astype(np.float64).reshape(-1)
    ok = z > 0
    ys, xs, z = ys[ok], xs[ok], z[ok]
    x = (xs - K[0, 2]) * z / K[0, 0]
    y = (ys - K[1, 2]) * z / K[1, 1]
    pts = np.stack([x, y, z], axis=-1)
    hom = np.hstack([pts, np.ones((len(pts), 1))])
    return (E @ hom.T).T[:, :3]


red_mask = locate("red cube")
green_mask = locate("green cube")

# ---- grasp the red cube (plan_grasp does its own deprojection) ----
poses, scores = plan_grasp(depth, K, red_mask)
if len(poses) == 0:
    raise RuntimeError("no grasp candidates for the red cube")
T = E @ poses[int(np.argmax(scores))]
grasp_pos = T[:3, 3]
grasp_quat = np.roll(Rotation.from_matrix(T[:3, :3]).as_quat(), 1)  # xyzw->wxyz

move_to_joints(solve_ik(grasp_pos + np.array([0.0, 0.0, 0.10]), grasp_quat))
open_gripper()
move_to_joints(solve_ik(grasp_pos, grasp_quat))
close_gripper()
move_to_joints(solve_ik(grasp_pos + np.array([0.0, 0.0, 0.20]), grasp_quat))

# ---- where the red cube has to end up ----
# A single view sees only visible surfaces, so an OBB centre is biased toward
# the camera and its depth-axis extent is wrong. Two quantities ARE well
# observed from above: the top face's height (max z) and the top face's
# horizontal size (for a cube, its side length).
green_pts = mask_to_world(green_mask)
red_pts = mask_to_world(red_mask)

g_obb = get_oriented_bounding_box_from_3d_points(green_pts.astype(np.float64))
g_center = np.asarray(g_obb["center"], dtype=np.float64)
green_top_z = float(np.percentile(green_pts[:, 2], 98))   # robust max

r_obb = get_oriented_bounding_box_from_3d_points(red_pts.astype(np.float64))
r_extent = np.sort(np.asarray(r_obb["extent"], dtype=np.float64))
red_side = float(r_extent[-1])          # largest extent = the cube's side

place_pos = np.array([g_center[0], g_center[1], green_top_z + red_side / 2.0])

move_to_joints(solve_ik(place_pos + np.array([0.0, 0.0, 0.10]), grasp_quat))
move_to_joints(solve_ik(place_pos, grasp_quat))
open_gripper()
move_to_joints(solve_ik(place_pos + np.array([0.0, 0.0, 0.10]), grasp_quat))
