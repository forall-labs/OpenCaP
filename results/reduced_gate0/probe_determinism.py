import numpy as np

obs = get_observation()
cam = obs["robot0_robotview"]
rgb, depth, K = cam["images"]["rgb"], cam["images"]["depth"], cam["intrinsics"]

# Same text prompt twice -> is molmo deterministic within a trial?
m1 = point_prompt_molmo(rgb, "red cube").get("red cube", (None, None))
m2 = point_prompt_molmo(rgb, "red cube").get("red cube", (None, None))
print("MOLMO_1", tuple(m1), "MOLMO_2", tuple(m2), "SAME", tuple(m1) == tuple(m2))

# Same point twice -> is SAM3 deterministic?
r1 = segment_sam3_point_prompt(rgb, m1)
r2 = segment_sam3_point_prompt(rgb, m1)
k1 = max(r1, key=lambda d: d["score"]); k2 = max(r2, key=lambda d: d["score"])
a1 = np.asarray(k1["mask"]); a2 = np.asarray(k2["mask"])
print("SAM3_px", int(a1.sum()), int(a2.sum()), "score", round(float(k1["score"]), 6),
      round(float(k2["score"]), 6), "SAME", bool(a1.sum() == a2.sum()))

# Same mask twice -> is the GRASP PLANNER deterministic?
p1, s1 = plan_grasp(depth, K, a1)
p2, s2 = plan_grasp(depth, K, a1)
print("GRASP_n", len(s1), len(s2))
if len(s1) and len(s2):
    b1 = p1[int(np.argmax(s1))][:3, 3]
    b2 = p2[int(np.argmax(s2))][:3, 3]
    print("GRASP_best1", np.round(b1, 5).tolist(), "score", round(float(np.max(s1)), 6))
    print("GRASP_best2", np.round(b2, 5).tolist(), "score", round(float(np.max(s2)), 6))
    print("GRASP_SAME", bool(np.allclose(b1, b2, atol=1e-6)),
          "shift_mm", round(float(np.linalg.norm(b1 - b2) * 1000), 3))
raise RuntimeError("probe only -- do not actuate")
