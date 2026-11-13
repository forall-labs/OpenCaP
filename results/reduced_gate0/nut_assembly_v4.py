import numpy as np
from scipy.spatial.transform import Rotation

obs = get_observation()
cam = obs["robot0_robotview"]
rgb, depth = cam["images"]["rgb"], cam["images"]["depth"]
K, E = cam["intrinsics"], cam["pose_mat"]

DOWN = np.diag([1.0, -1.0, -1.0])          # gripper pointing down, fingers open along world y


def world_pts(mask, depth=depth):
    m = np.asarray(mask)
    m = m[:, :, 0] if m.ndim == 3 else m
    ys, xs = np.where(m > 0)
    z = np.asarray(depth)[ys, xs].astype(np.float64).reshape(-1)
    ok = z > 0
    ys, xs, z = ys[ok], xs[ok], z[ok]
    if len(z) == 0:
        raise RuntimeError("mask had no valid depth")
    pc = np.stack([(xs - K[0, 2]) * z / K[0, 0], (ys - K[1, 2]) * z / K[1, 1], z], axis=-1)
    return (E @ np.hstack([pc, np.ones((len(pc), 1))]).T).T[:, :3]


def top_face(pts, band=0.006):
    top = pts[pts[:, 2] >= pts[:, 2].max() - band]
    return top if len(top) >= 10 else pts


def yaw_quat(phi):
    """Top-down gripper whose fingers open along world direction angle `phi`."""
    R = Rotation.from_euler("z", phi - np.pi / 2) * Rotation.from_matrix(DOWN)
    return np.roll(R.as_quat(), 1)                      # xyzw -> wxyz


def wrap90(a):
    """An angle modulo 90 degrees, into [-45, 45) degrees -- a square's symmetry."""
    return (a + np.pi / 4) % (np.pi / 2) - np.pi / 4


# ---- 1. the nut: ring centre, handle, and yaw, from its own shape -----------
# The nut is a square ring with a handle sticking out of the middle of one side.
# Its top face, projected on its principal axis, is WIDE across the ring and
# NARROW along the handle; the ring's edges run parallel and perpendicular to
# that axis, so the axis is also the nut's yaw.
res = segment_sam3_text_prompt(rgb, text_prompt="brown square nut")
if not res:
    raise RuntimeError("no mask for the nut")
nut = top_face(world_pts(max(res, key=lambda d: d["score"])["mask"]))
c = nut[:, :2].mean(axis=0)
evals, evecs = np.linalg.eigh(np.cov((nut[:, :2] - c).T))
u = evecs[:, np.argmax(evals)]                          # handle axis
v = np.array([-u[1], u[0]])
a = (nut[:, :2] - c) @ u
b = (nut[:, :2] - c) @ v
edges = np.arange(a.min(), a.max() + 0.004, 0.004)
idx = np.clip(np.digitize(a, edges) - 1, 0, len(edges) - 1)
width = np.array([np.ptp(b[idx == i]) if (idx == i).sum() > 2 else 0.0
                  for i in range(len(edges))])
ring_bins = np.where(width >= 0.6 * width.max())[0]
in_ring = np.isin(idx, ring_bins)
a_lo, a_hi = a[in_ring].min(), a[in_ring].max()
ring_c = c + 0.5 * (a_lo + a_hi) * u + 0.5 * (b[in_ring].min() + b[in_ring].max()) * v
ring_side = float(a_hi - a_lo)                       # outer side of the square ring
handle = ~in_ring & ((a < a_lo - 0.004) | (a > a_hi + 0.004))
if handle.sum() < 5:
    raise RuntimeError("could not separate the handle from the ring")
h_xy = c + np.median(a[handle]) * u + np.median(b[handle]) * v
h_top = float(np.percentile(nut[handle][:, 2], 90))
nut_yaw = np.arctan2(u[1], u[0])
print("NUT ring", np.round(ring_c, 4), "handle", np.round(h_xy, 4), "top", round(h_top, 4),
      "yaw", round(np.degrees(nut_yaw), 1), "n_handle", int(handle.sum()))

# ---- 2. the peg: centre, top and yaw ----------------------------------------
# "the square peg" makes molmo point at the brown nut; the S1 oracle's own name
# for it, "square block", points at the peg. SAM3's best "square peg" mask is
# the fallback.
#
# The peg stands ~11 cm above the nut, so a candidate whose top is not well
# above the nut's is the nut itself -- molmo's "square block" pointed at the
# nut in 2 of 30 scenes, and the program then carried the nut onto itself.
nut_top = float(nut[:, 2].max())


def tall(mask):
    """A peg candidate: its top stands well above the nut AND is compact. v2
    checked only height, and a mask spanning both pegs passed it, putting the
    'peg' centre between them (2 of 30 scenes)."""
    try:
        t = top_face(world_pts(mask), band=0.004)
    except RuntimeError:
        return False
    return (float(t[:, 2].max()) > nut_top + 0.05
            and np.ptp(t[:, 0]) < 0.05 and np.ptp(t[:, 1]) < 0.05)


pr = point_prompt_molmo(rgb, "square block")
pc = pr.get("square block", (None, None))
peg_mask = None
if pc is not None and tuple(pc) != (None, None):
    r2 = [d for d in (segment_sam3_point_prompt(rgb, pc) or []) if tall(d["mask"])]
    if r2:
        peg_mask = max(r2, key=lambda d: d["score"])["mask"]
if peg_mask is None:
    r2 = [d for d in (segment_sam3_text_prompt(rgb, text_prompt="square peg") or []) if tall(d["mask"])]
    if not r2:
        raise RuntimeError("could not locate the peg")
    peg_mask = max(r2, key=lambda d: d["score"])["mask"]
peg = top_face(world_pts(peg_mask), band=0.004)
peg_xy = peg[:, :2].mean(axis=0)
peg_top = float(peg[:, 2].max())
best = None
for th in np.radians(np.arange(0.0, 90.0, 1.0)):    # minimum-area rectangle
    Rz = np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]])
    q = (peg[:, :2] - peg_xy) @ Rz.T
    area = np.ptp(q[:, 0]) * np.ptp(q[:, 1])
    if best is None or area < best[0]:
        best = (area, th)
peg_yaw = best[1]
print("PEG", np.round(peg_xy, 4), "top", round(peg_top, 4), "yaw", round(np.degrees(peg_yaw), 1),
      "n", len(peg))

# ---- 3. grasp the handle, top-down, fingers closing across it ---------------
phi = np.arctan2(v[1], v[0])
phi = (phi + np.pi / 2) % np.pi - np.pi / 2           # a parallel gripper is symmetric
q_grasp = yaw_quat(phi)
grasp = np.array([h_xy[0], h_xy[1], h_top - 0.008])
move_to_joints(solve_ik(grasp + np.array([0.0, 0.0, 0.08]), q_grasp))
open_gripper()
move_to_joints(solve_ik(grasp, q_grasp))
close_gripper()
lift = grasp + np.array([0.0, 0.0, 0.15])
move_to_joints(solve_ik(lift, q_grasp))

# ---- 4. square the ring to the peg, carry the hole over it, insert ----------
# A square ring fits the peg at any multiple of 90 degrees. Of those turns, take
# the smallest one that does not leave the handle pointing AWAY from the robot:
# that puts the fingers past the peg, at the edge of the arm's reach (the one
# failure in the first 6-scene probe was exactly that, fingers at x=0.85).
base = wrap90(peg_yaw - nut_yaw)
h_dir = np.array([grasp[0] - ring_c[0], grasp[1] - ring_c[1], 0.0])
cands = []
for k in (-1, 0, 1, 2, -2):
    r = base + k * np.pi / 2
    if abs(r) > np.radians(135):
        continue
    d = Rotation.from_euler("z", r).apply(h_dir)
    away = d[0] / (np.linalg.norm(d) + 1e-9) > 0.7
    cands.append((away, abs(r), r))
delta = min(cands)[2]
Rd = Rotation.from_euler("z", delta)
q_place = np.roll((Rd * Rotation.from_quat(np.roll(q_grasp, -1))).as_quat(), 1)
off = Rd.apply(np.array([ring_c[0] - grasp[0], ring_c[1] - grasp[1], 0.0]))
xy = peg_xy - off[:2]
print("PLACE delta", round(np.degrees(delta), 1), "tcp_xy", np.round(xy, 4))
# v3: turn the wrist in <=15 degree steps at lift height, then travel through a
# midpoint. solve_ik seeds from its PREVIOUS solution, not from where the arm
# is (move_to_joints(HOME) does not reset it), so one big jump lands in a
# different arm configuration and the swing to it threw the nut -- v2 failed
# every scene that needed a large positive turn, and none that reached the SAME
# final pose by a negative one.
q_grasp_R = Rotation.from_quat(np.roll(q_grasp, -1))
for s in np.linspace(0.0, delta, int(np.ceil(abs(np.degrees(delta)) / 15.0)) + 1)[1:]:
    move_to_joints(solve_ik(lift, np.roll((Rotation.from_euler("z", s) * q_grasp_R).as_quat(), 1)))
above = np.array([xy[0], xy[1], peg_top + 0.05])
move_to_joints(solve_ik(0.5 * (lift + above), q_place))
move_to_joints(solve_ik(above, q_place))

# v4: LOOK before inserting. Hold the nut over the peg, find it in a fresh
# image, and correct the remaining x,y error. v3's misses all ended with the nut
# resting on the peg top ~2 cm short in x: grasp slip and single-view bias add
# up to more than the hole's clearance, and nothing open-loop removes that.
# The handle may be hidden by the fingers, so the ring centre is taken from the
# ring's FAR edge (opposite the handle, whose direction is known) plus half the
# side measured at the start -- which needs no view of the handle.
o3 = get_observation()["robot0_robotview"]
rgb3, depth3 = o3["images"]["rgb"], o3["images"]["depth"]
u2 = Rd.apply(np.array([u[0], u[1], 0.0]))[:2]
v2 = np.array([-u2[1], u2[0]])
h2 = Rd.apply(h_dir)[:2]
s = 1.0 if h2 @ u2 > 0 else -1.0                         # handle side along u2
held = []
for d3 in (segment_sam3_text_prompt(rgb3, text_prompt="brown square nut") or []):
    try:
        p3 = world_pts(d3["mask"], depth3)
    except RuntimeError:
        continue
    t3 = top_face(p3)
    if peg_top + 0.01 < float(t3[:, 2].max()) < peg_top + 0.15 and len(t3) > 30:
        held.append((d3["score"], t3))
corr = np.zeros(2)
if held:
    t3 = max(held, key=lambda x: x[0])[1]
    c3 = t3[:, :2].mean(axis=0)
    a3 = (t3[:, :2] - c3) @ u2
    b3 = (t3[:, :2] - c3) @ v2
    far = a3.min() if s > 0 else a3.max()
    ring_now = c3 + (far + s * ring_side / 2.0) * u2 + 0.5 * (b3.min() + b3.max()) * v2
    corr = peg_xy - ring_now
    if np.linalg.norm(corr) > 0.04:                         # implausible: trust the open-loop estimate
        corr = np.zeros(2)
print("CORR", np.round(corr, 4), "seen", len(held))
xy = xy + corr
above = np.array([xy[0], xy[1], peg_top + 0.05])
move_to_joints(solve_ik(above, q_place))
move_to_joints(solve_ik(np.array([xy[0], xy[1], peg_top + 0.015]), q_place))
move_to_joints(solve_ik(np.array([xy[0], xy[1], peg_top - 0.03]), q_place))
open_gripper()
move_to_joints(solve_ik(above, q_place))
