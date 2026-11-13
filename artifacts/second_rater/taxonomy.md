# Tag taxonomy (S1, opencap/editclass.FAMILY)

- **FRAME**: `frame_convention`
- **INTERFACE**: `call_signature`, `object_name`, `unbound_name`
- **SEMANTIC**: `missing_action`, `missing_coverage`, `stale_read`, `wrong_formula`, `wrong_target`, `wrong_task`
- **STRUCTURE**: `api_rebound`, `gripper_order`, `mock_shadowing`, `never_invoked`
- **SYNTAX**: `syntax`
- **TRAJECTORY**: `approach_param`, `post_release_retreat`, `spurious_move`

Conventions: call_signature also covers misreading what an API RETURNS; frame_convention covers camera-frame values used as world, wxyz/xyzw, pixel-as-world; object_name covers text prompts; approach_param covers approach/lift offsets; an absent insertion recipe is missing_action; a missing pre-grasp approach is approach_param.
