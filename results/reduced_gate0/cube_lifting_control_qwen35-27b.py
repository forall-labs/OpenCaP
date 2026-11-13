import numpy as np
from scipy.spatial.transform import Rotation

# Get current observation
obs = get_observation()
rgb = obs["robot0_robotview"]["images"]["rgb"]
depth = obs["robot0_robotview"]["images"]["depth"]
intrinsics = obs["robot0_robotview"]["intrinsics"]
camera_extrinsics = obs["robot0_robotview"]["pose_mat"]

# Use Molmo to find the red cube
point_result = point_prompt_molmo(rgb, "red cube")
point_coords = point_result.get("red cube", (None, None))

if point_coords == (None, None):
    raise RuntimeError("Failed to locate red cube using Molmo")

# Use SAM3 with point prompt to segment the red cube
sam3_results = segment_sam3_point_prompt(rgb, point_coords)
if not sam3_results:
    raise RuntimeError("SAM3 failed to segment the red cube")

# Get the best mask
best_mask = max(sam3_results, key=lambda d: d["score"])["mask"]

# Plan grasp
grasp_poses, grasp_scores = plan_grasp(depth, intrinsics, best_mask)
if len(grasp_poses) == 0:
    raise RuntimeError("No grasp candidates found")

# Select best grasp
best_idx = np.argmax(grasp_scores)
best_grasp_pose_cam = grasp_poses[best_idx]

# Transform grasp pose to world frame
best_grasp_pose_world = camera_extrinsics @ best_grasp_pose_cam

# Extract position and orientation from the grasp pose
grasp_pos = best_grasp_pose_world[:3, 3]
grasp_quat = Rotation.from_matrix(best_grasp_pose_world[:3, :3]).as_quat()
grasp_quat_wxyz = np.roll(grasp_quat, 1)  # Convert xyzw to wxyz

# Move to approach position (above the grasp point)
approach_offset = np.array([0.0, 0.0, 0.1])
approach_pos = grasp_pos + approach_offset
approach_joints = solve_ik(approach_pos, grasp_quat_wxyz)
move_to_joints(approach_joints)

# Open gripper
open_gripper()

# Move to grasp position
grasp_joints = solve_ik(grasp_pos, grasp_quat_wxyz)
move_to_joints(grasp_joints)

# Close gripper to pick up the cube
close_gripper()

# Lift the cube
lift_offset = np.array([0.0, 0.0, 0.15])
lift_pos = grasp_pos + lift_offset
lift_joints = solve_ik(lift_pos, grasp_quat_wxyz)
move_to_joints(lift_joints)