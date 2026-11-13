# Code block 0
import numpy as np
from scipy.spatial.transform import Rotation

obs = get_observation()
cam = obs["robot0_robotview"]
rgb = cam["images"]["rgb"]
depth = cam["images"]["depth"]
K = cam["intrinsics"]
E = cam["pose_mat"]          # camera -> world


def locate(name):
    """v7pg_text: the SAM3 TEXT prompt's best mask -- the route most model
    programs take -- instead of molmo point -> SAM3 point prompt."""
    res = segment_sam3_text_prompt(rgb, name)
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
# v6: with the cubes touching (scene 24) the SAM3 mask grown from molmo's red
# point covered BOTH cubes, the red "centre" fell between them, and the grasp
# took the green cube. A pixel cannot belong to both.
_r = np.asarray(red_mask); _r = _r[:, :, 0] if _r.ndim == 3 else _r
_g = np.asarray(green_mask); _g = _g[:, :, 0] if _g.ndim == 3 else _g
red_mask = _r & ~_g
# v7: pixels on a mask's boundary take their depth from whatever is behind or
# in front of them. With the cubes touching (scene 24), red-mask boundary
# pixels read the GREEN cube's top, top_face() keyed on that maximum, and the
# red "top" landed at the green cube's height and on the boundary between the
# two -- so the grasp straddled the green cube. Erode before measuring.
from scipy.ndimage import binary_erosion
red_geom = binary_erosion(red_mask, iterations=2)
green_geom = binary_erosion(_g, iterations=2)
DOWN = np.diag([1.0, -1.0, -1.0])          # gripper pointing down, fingers open along world y


def top_face(pts, band=0.004):
    zref = float(np.percentile(pts[:, 2], 95))     # not the max: one stray pixel moved it
    top = pts[np.abs(pts[:, 2] - zref) <= band]
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


def table_z_around(mask, rad=25):
    """Height of the table in a ring around a mask -- the cube's base.

    v4 used a narrow ring that excluded only the two masks themselves; with the
    cubes touching (scene 24) it picked up the green cube's side walls, read the
    red cube 1 cm too tall, and released it from a height. Excluding a dilated
    band around BOTH cubes leaves only table.
    """
    from scipy.ndimage import binary_dilation
    m = np.asarray(mask); m = m[:, :, 0] if m.ndim == 3 else m
    gm = np.asarray(green_mask); gm = gm[:, :, 0] if gm.ndim == 3 else gm
    ring = binary_dilation(m, iterations=rad) & ~binary_dilation(m | gm, iterations=8)
    z = mask_to_world(ring)[:, 2]
    # the MODE, not the median: near the table's far edge the ring reaches
    # past it and the median slid 1 cm low (scene 24 again)
    hist, edges = np.histogram(z, bins=np.arange(z.min(), z.max() + 0.002, 0.002))
    k = int(np.argmax(hist))
    return float(0.5 * (edges[k] + edges[k + 1]))


# v4: grasp ANALYTICALLY, not with plan_grasp. Videos of v3 show its failures
# include clean misses -- the sampled grasp comes in tilted beside the cube and
# closes on air. A top-down grasp at the top face's centre, fingers square to
# the cube's edges, is fully determined by perception, which is deterministic,
# so the reference stops inheriting the grasp sampler's noise.
red_pts = mask_to_world(red_geom)
r_top = top_face(red_pts)
red_xy = r_top[:, :2].mean(axis=0)
red_top_z = float(np.percentile(r_top[:, 2], 90))
red_side = red_top_z - table_z_around(red_mask)   # v3 read this off the top face: 4.3-5.6 cm for a 4 cm cube
yaw = yaw_of_square(r_top)
# A square can be gripped across either pair of faces. Close the fingers ACROSS
# the red->green direction, never along it: with the cubes touching (scene 24)
# a finger coming down between them lands on the green cube.
g_xy = mask_to_world(green_geom)[:, :2].mean(axis=0)
to_green = np.arctan2(*(g_xy - red_xy)[::-1])
yaw = min((yaw, yaw + np.pi / 2), key=lambda y: abs(np.cos(y + np.pi / 2 - to_green)))
q = np.roll((Rotation.from_euler("z", yaw) * Rotation.from_matrix(DOWN)).as_quat(), 1)
# v7pg: the API docstring's grasp (plan_grasp, best score) in place of the
# analytic one; everything else is v7. Carries the grasp's offset from the red
# cube's measured centre into the placement.
poses, scores = plan_grasp(depth, K, red_mask)
if len(poses) == 0:
    raise RuntimeError("no grasp candidates for the red cube")
T = E @ poses[int(np.argmax(scores))]
grasp_pos = T[:3, 3]
q = np.roll(Rotation.from_matrix(T[:3, :3]).as_quat(), 1)
red_centre = np.array([red_xy[0], red_xy[1], red_top_z - red_side / 2.0])
offset = grasp_pos - red_centre

move_to_joints(solve_ik(grasp_pos + np.array([0.0, 0.0, 0.10]), q))
open_gripper()
move_to_joints(solve_ik(grasp_pos, q))
close_gripper()
move_to_joints(solve_ik(grasp_pos + np.array([0.0, 0.0, 0.20]), q))

# ---- place: the red cube's BASE onto the green cube's top face ----------------
g_top = top_face(mask_to_world(green_geom))
green_xy = g_top[:, :2].mean(axis=0)
green_top_z = float(np.percentile(g_top[:, 2], 90))
CLEARANCE = 0.003
place_pos = np.array([green_xy[0], green_xy[1], green_top_z + CLEARANCE + red_side / 2.0]) + offset
print("V4 red_xy", np.round(red_xy, 4), "side", round(red_side, 4), "yaw", round(np.degrees(yaw), 1),
      "green", np.round(green_xy, 4), round(green_top_z, 4), "place", np.round(place_pos, 4))
move_to_joints(solve_ik(place_pos + np.array([0.0, 0.0, 0.10]), q))
move_to_joints(solve_ik(place_pos, q))
open_gripper()
move_to_joints(solve_ik(place_pos + np.array([0.0, 0.0, 0.10]), q))