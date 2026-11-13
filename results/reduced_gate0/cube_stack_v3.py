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

def top_face(pts, band=0.006):
    """Points within `band` of the highest z -- the visible top face."""
    top = pts[:, 2] >= pts[:, 2].max() - band
    return pts[top] if top.sum() >= 10 else pts


# v3: measure the red cube's centre from its own top face BEFORE grasping.
# plan_grasp resamples, and the fingertip point it returns sits at a different
# height and x,y on the cube every draw (1.4-15.2 mm). v2 placed that point as
# if it were the cube's centre, so a grasp 1 cm above centre pressed the red
# cube 1 cm into the green one -- which unseats both -- and one below centre
# dropped it. Carrying the grasp's own offset into the placement removes the
# grasp sample from the placement error.
red_pts = mask_to_world(red_mask)
r_top = top_face(red_pts)
red_side = float(max(r_top[:, 0].ptp(), r_top[:, 1].ptp()))
if not (0.01 < red_side < 0.10):          # fall back to the OBB if the face is thin
    r_obb = get_oriented_bounding_box_from_3d_points(red_pts.astype(np.float64))
    red_side = float(np.sort(np.asarray(r_obb["extent"], dtype=np.float64))[-1])
red_centre = np.array([*r_top[:, :2].mean(axis=0),
                       float(np.percentile(r_top[:, 2], 90)) - red_side / 2.0])

# ---- grasp the red cube (plan_grasp does its own deprojection) ----
poses, scores = plan_grasp(depth, K, red_mask)
if len(poses) == 0:
    raise RuntimeError("no grasp candidates for the red cube")
T = E @ poses[int(np.argmax(scores))]
grasp_pos = T[:3, 3]
grasp_quat = np.roll(Rotation.from_matrix(T[:3, :3]).as_quat(), 1)  # xyzw->wxyz
grasp_offset = grasp_pos - red_centre       # where on the cube the fingers are

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

g_top = top_face(green_pts)
green_xy = g_top[:, :2].mean(axis=0)
green_top_z = float(np.percentile(g_top[:, 2], 90))

# Release a little high and let it settle rather than pressing the red cube
# into the green one, which unseats both.
CLEARANCE = 0.004
# where the red cube's CENTRE must go, then the fingers go there plus the offset
centre_goal = np.array([green_xy[0], green_xy[1], green_top_z + red_side / 2.0 + CLEARANCE])
place_pos = centre_goal + grasp_offset

move_to_joints(solve_ik(place_pos + np.array([0.0, 0.0, 0.12]), grasp_quat))
move_to_joints(solve_ik(place_pos, grasp_quat))
open_gripper()
move_to_joints(solve_ik(place_pos + np.array([0.0, 0.0, 0.12]), grasp_quat))
