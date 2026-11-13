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
DOWN = np.diag([1.0, -1.0, -1.0])          # gripper pointing down, fingers open along world y


def top_face(pts, band=0.004):
    top = pts[pts[:, 2] >= pts[:, 2].max() - band]
    return top if len(top) >= 10 else pts


def yaw_of_square(pts):
    """Yaw (mod 90 deg) of a square top face: the minimum-area bounding rectangle."""
    c = pts[:, :2].mean(axis=0)
    best = None
    for th in np.radians(np.arange(0.0, 90.0, 1.0)):
        R = np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]])
        q = (pts[:, :2] - c) @ R.T
        area = np.ptp(q[:, 0]) * np.ptp(q[:, 1])
        if best is None or area < best[0]:
            best = (area, th)
    return best[1]


def table_z_around(mask, rad=12):
    """Height of the table in a ring just outside a mask -- the cube's base."""
    from scipy.ndimage import binary_dilation
    m = np.asarray(mask); m = m[:, :, 0] if m.ndim == 3 else m
    ring = binary_dilation(m, iterations=rad) & ~binary_dilation(m, iterations=rad // 2)
    gm = np.asarray(green_mask); gm = gm[:, :, 0] if gm.ndim == 3 else gm
    return float(np.median(mask_to_world(ring & ~gm & ~m)[:, 2]))


# v4: grasp ANALYTICALLY, not with plan_grasp. Videos of v3 show its failures
# include clean misses -- the sampled grasp comes in tilted beside the cube and
# closes on air. A top-down grasp at the top face's centre, fingers square to
# the cube's edges, is fully determined by perception, which is deterministic,
# so the reference stops inheriting the grasp sampler's noise.
red_pts = mask_to_world(red_mask)
r_top = top_face(red_pts)
red_xy = r_top[:, :2].mean(axis=0)
red_top_z = float(np.percentile(r_top[:, 2], 90))
red_side = red_top_z - table_z_around(red_mask)   # v3 read this off the top face: 4.3-5.6 cm for a 4 cm cube
yaw = yaw_of_square(r_top)
q = np.roll((Rotation.from_euler("z", yaw) * Rotation.from_matrix(DOWN)).as_quat(), 1)
GRIP = 0.015                                      # fingertips this far below the top face
grasp_pos = np.array([red_xy[0], red_xy[1], red_top_z - GRIP])

move_to_joints(solve_ik(grasp_pos + np.array([0.0, 0.0, 0.10]), q))
open_gripper()
move_to_joints(solve_ik(grasp_pos, q))
close_gripper()
move_to_joints(solve_ik(grasp_pos + np.array([0.0, 0.0, 0.20]), q))

# ---- place: the red cube's BASE onto the green cube's top face ----------------
g_top = top_face(mask_to_world(green_mask))
green_xy = g_top[:, :2].mean(axis=0)
green_top_z = float(np.percentile(g_top[:, 2], 90))
CLEARANCE = 0.003
place_pos = np.array([green_xy[0], green_xy[1],
                      green_top_z + CLEARANCE + red_side - GRIP])
print("V4 red_xy", np.round(red_xy, 4), "side", round(red_side, 4), "yaw", round(np.degrees(yaw), 1),
      "green", np.round(green_xy, 4), round(green_top_z, 4), "place", np.round(place_pos, 4))
move_to_joints(solve_ik(place_pos + np.array([0.0, 0.0, 0.10]), q))
move_to_joints(solve_ik(place_pos, q))
open_gripper()
move_to_joints(solve_ik(place_pos + np.array([0.0, 0.0, 0.10]), q))
