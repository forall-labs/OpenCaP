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



SEED_POSE = np.array([0.45, 0.0, 0.30])


def analyse_nut(rgb, depth):
    """Ring centre, handle point and yaw of the nut, from its own top face.

    The nut is a square ring with a handle out of the middle of one side: its
    top face is WIDE across the ring and NARROW along the handle on the
    principal axis, and the ring's edges run along/across that axis."""
    res = segment_sam3_text_prompt(rgb, text_prompt="brown square nut")
    if not res:
        raise RuntimeError("no mask for the nut")
    nut = top_face(world_pts(max(res, key=lambda d: d["score"])["mask"], depth))
    c = nut[:, :2].mean(axis=0)
    evals, evecs = np.linalg.eigh(np.cov((nut[:, :2] - c).T))
    u = evecs[:, np.argmax(evals)]
    v = np.array([-u[1], u[0]])
    a = (nut[:, :2] - c) @ u
    b = (nut[:, :2] - c) @ v
    edges = np.arange(a.min(), a.max() + 0.004, 0.004)
    idx = np.clip(np.digitize(a, edges) - 1, 0, len(edges) - 1)
    width = np.array([np.ptp(b[idx == i]) if (idx == i).sum() > 2 else 0.0 for i in range(len(edges))])
    in_ring = np.isin(idx, np.where(width >= 0.6 * width.max())[0])
    a_lo, a_hi = a[in_ring].min(), a[in_ring].max()
    ring_c = c + 0.5 * (a_lo + a_hi) * u + 0.5 * (b[in_ring].min() + b[in_ring].max()) * v
    handle = ~in_ring & ((a < a_lo - 0.004) | (a > a_hi + 0.004))
    if handle.sum() < 5:
        raise RuntimeError("could not separate the handle from the ring")
    h_xy = c + np.median(a[handle]) * u + np.median(b[handle]) * v
    h_top = float(np.percentile(nut[handle][:, 2], 90))
    return {"ring_c": ring_c, "u": u, "v": v, "h_xy": h_xy, "h_top": h_top,
            "yaw": np.arctan2(u[1], u[0]), "top": float(nut[:, 2].max())}


def find_peg(rgb, depth, nut_top):
    """The square peg: molmo's "square block" (the S1 oracle's own name for it;
    "the square peg" points at the brown nut), checked to be tall and compact."""
    def ok(mask):
        try:
            t = top_face(world_pts(mask, depth), band=0.004)
        except RuntimeError:
            return False
        return (float(t[:, 2].max()) > nut_top + 0.05
                and np.ptp(t[:, 0]) < 0.05 and np.ptp(t[:, 1]) < 0.05)
    pr = point_prompt_molmo(rgb, "square block")
    pc = pr.get("square block", (None, None))
    peg_mask = None
    if pc is not None and tuple(pc) != (None, None):
        r2 = [d for d in (segment_sam3_point_prompt(rgb, pc) or []) if ok(d["mask"])]
        if r2:
            peg_mask = max(r2, key=lambda d: d["score"])["mask"]
    if peg_mask is None:
        r2 = [d for d in (segment_sam3_text_prompt(rgb, text_prompt="square peg") or []) if ok(d["mask"])]
        if not r2:
            raise RuntimeError("could not locate the peg")
        peg_mask = max(r2, key=lambda d: d["score"])["mask"]
    peg = top_face(world_pts(peg_mask, depth), band=0.004)
    xy = peg[:, :2].mean(axis=0)
    best = None
    for th in np.radians(np.arange(0.0, 90.0, 1.0)):    # minimum-area rectangle
        Rz = np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]])
        q = (peg[:, :2] - xy) @ Rz.T
        area = np.ptp(q[:, 0]) * np.ptp(q[:, 1])
        if best is None or area < best[0]:
            best = (area, th)
    return {"xy": xy, "top": float(peg[:, 2].max()), "yaw": best[1]}


def attempt(nut, peg, avoid_toward):
    """Grasp the handle, square the ring to the peg, insert, release."""
    # one fixed IK solve first: solve_ik seeds from its previous solution, which
    # persists across trials on a worker (v5)
    move_to_joints(solve_ik(SEED_POSE, yaw_quat(0.0)))
    phi = np.arctan2(nut["v"][1], nut["v"][0])
    phi = (phi + np.pi / 2) % np.pi - np.pi / 2       # a parallel gripper is symmetric
    q_grasp = yaw_quat(phi)
    # v8: grasp the handle 5 mm closer to the ring than its median point. The
    # closing fingers leave the nut further OUT along the handle than planned
    # (+1 to +5 mm over 30 scenes, measured against the true pose for diagnosis
    # only); at +4-5 mm the ring lands on the peg's edge (scene 21). Same result
    # at 8 mm; 5 mm outward makes it worse (26/30).
    _u = np.asarray(nut["ring_c"]) - np.asarray(nut["h_xy"]); _u = _u / (np.linalg.norm(_u) + 1e-9)
    _h = np.asarray(nut["h_xy"]) + 0.005 * _u
    grasp = np.array([_h[0], _h[1], nut["h_top"] - 0.008])
    move_to_joints(solve_ik(grasp + np.array([0.0, 0.0, 0.08]), q_grasp))
    open_gripper()
    move_to_joints(solve_ik(grasp, q_grasp))
    close_gripper()
    lift = grasp + np.array([0.0, 0.0, 0.15])
    move_to_joints(solve_ik(lift, q_grasp))
    # the smallest 90-degree-equivalent turn that keeps the handle from pointing
    # AWAY from the robot (reach), and -- in the first attempt -- TOWARD it
    base = wrap90(peg["yaw"] - nut["yaw"])
    h_dir = np.array([grasp[0] - nut["ring_c"][0], grasp[1] - nut["ring_c"][1], 0.0])
    cands = []
    for k in (-1, 0, 1, 2, -2):
        r = base + k * np.pi / 2
        if abs(r) > np.radians(135):
            continue
        d = Rotation.from_euler("z", r).apply(h_dir)
        cos_x = d[0] / (np.linalg.norm(d) + 1e-9)
        cands.append((cos_x > 0.7, avoid_toward and cos_x < -0.7, abs(r), r))
    delta = min(cands)[3]
    Rd = Rotation.from_euler("z", delta)
    q_grasp_R = Rotation.from_quat(np.roll(q_grasp, -1))
    q_place = np.roll((Rd * q_grasp_R).as_quat(), 1)
    off = Rd.apply(np.array([nut["ring_c"][0] - grasp[0], nut["ring_c"][1] - grasp[1], 0.0]))
    xy = peg["xy"] - off[:2]
    print("PLACE avoid_toward", avoid_toward, "delta", round(np.degrees(delta), 1), "tcp_xy", np.round(xy, 4))
    # turn in <=15 degree steps so each IK solve is seeded by a nearby pose (v3)
    for s in np.linspace(0.0, delta, int(np.ceil(abs(np.degrees(delta)) / 15.0)) + 1)[1:]:
        move_to_joints(solve_ik(lift, np.roll((Rotation.from_euler("z", s) * q_grasp_R).as_quat(), 1)))
    above = np.array([xy[0], xy[1], peg["top"] + 0.05])
    move_to_joints(solve_ik(0.5 * (lift + above), q_place))
    move_to_joints(solve_ik(above, q_place))
    move_to_joints(solve_ik(np.array([xy[0], xy[1], peg["top"] + 0.015]), q_place))
    move_to_joints(solve_ik(np.array([xy[0], xy[1], peg["top"] - 0.03]), q_place))
    open_gripper()
    move_to_joints(solve_ik(above, q_place))


def seated(peg, table_top):
    """Look, with the arm out of the way: is there a nut LOW and AROUND the peg?

    The nut env's observations are fresh (force_update), unlike the cubes env's.
    The peg's own brown top also answers "brown square nut", so candidates are
    taken only near table height, where a seated nut sits and the peg top
    cannot."""
    move_to_joints(solve_ik(SEED_POSE, yaw_quat(0.0)))
    o = get_observation()["robot0_robotview"]
    rgb2, depth2 = o["images"]["rgb"], o["images"]["depth"]
    # A SEATED nut is a ring AROUND the peg at table height: low points spread
    # over more than 4 cm in x AND y, centred on the peg. The first version asked
    # only for low points near the peg -- and the peg's own lower side wall
    # answered "brown square nut" and passed it, so a failed insertion read as
    # seated in every scene.
    for d in (segment_sam3_text_prompt(rgb2, text_prompt="brown square nut") or []):
        try:
            p = world_pts(d["mask"], depth2)
        except RuntimeError:
            continue
        near = np.linalg.norm(p[:, :2] - peg["xy"], axis=1) < 0.07
        low = p[near & (p[:, 2] < table_top + 0.02)]
        if len(low) < 100:
            continue
        spread = np.ptp(low[:, :2], axis=0)
        off = float(np.linalg.norm(low[:, :2].mean(axis=0) - peg["xy"]))
        print("SEATCHECK n", len(low), "spread", np.round(spread, 3), "off", round(off, 3))
        # Thresholds from 60 measured outcomes (v7b): every seated nut gave
        # n >= 1329, spread >= 0.086 m on both axes, off <= 0.018 m; every
        # failure gave n <= 844 -- a loose (n>50, spread>0.04) version passed
        # three failures as seated.
        if len(low) > 1000 and spread.min() > 0.07 and off < 0.025:
            return True
    return False


# v7: v6 failed scenes 14 and 21 in EVERY draw (a 63-69 deg turn leaves the nut
# leaning on the peg), v5's turn preference passed them but failed others, and
# no single rule did both. The program can LOOK: if the nut is not seated after
# the first attempt, pick it up from wherever it ended and try the other rule.
nut = analyse_nut(rgb, depth)
peg = find_peg(rgb, depth, nut["top"])
print("NUT", np.round(nut["ring_c"], 4), "yaw", round(np.degrees(nut["yaw"]), 1),
      "PEG", np.round(peg["xy"], 4), round(peg["top"], 4))
attempt(nut, peg, avoid_toward=True)
if not seated(peg, nut["top"]):
    print("RETRY: nut not seated after the first attempt")
    o = get_observation()["robot0_robotview"]
    attempt(analyse_nut(o["images"]["rgb"], o["images"]["depth"]), peg, avoid_toward=False)
    print("SEATED after retry:", seated(peg, nut["top"]))
else:
    print("SEATED on the first attempt")
