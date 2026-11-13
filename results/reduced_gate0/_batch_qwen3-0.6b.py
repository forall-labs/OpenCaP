import numpy as np
_obs = get_observation()
_rgb = _obs["robot0_robotview"]["images"]["rgb"]
_depth = _obs["robot0_robotview"]["images"]["depth"]
_K = _obs["robot0_robotview"]["intrinsics"]
_E = _obs["robot0_robotview"]["pose_mat"]
_res = segment_sam3_text_prompt(_rgb, text_prompt='spill')
if not _res:
    _pr = point_prompt_molmo(_rgb, 'spill')
    _pc = _pr.get('spill', (None, None))
    if _pc is None or tuple(_pc) == (None, None):
        raise RuntimeError("could not locate the spill")
    _res = segment_sam3_point_prompt(_rgb, _pc)
if not _res: raise RuntimeError("no mask for the spill")
_mask = max(_res, key=lambda d: d["score"])["mask"]

_m = np.asarray(_mask); _m = _m[:, :, 0] if _m.ndim == 3 else _m
_ys, _xs = np.where(_m > 0)
_z = np.asarray(_depth)[_ys, _xs].astype(np.float64).reshape(-1)
_ok = _z > 0; _ys, _xs, _z = _ys[_ok], _xs[_ok], _z[_ok]
if len(_z) == 0: raise RuntimeError("mask had no valid depth")
_cam = np.stack([(_xs - _K[0, 2]) * _z / _K[0, 0],
                 (_ys - _K[1, 2]) * _z / _K[1, 1], _z], axis=-1)
_pts = (_E @ np.hstack([_cam, np.ones((len(_cam), 1))]).T).T[:, :3]
_x0, _x1 = np.percentile(_pts[:, 0], [2, 98]); _y0, _y1 = np.percentile(_pts[:, 1], [2, 98])
_P = 0.018; _x0, _x1, _y0, _y1 = _x0-_P, _x1+_P, _y0-_P, _y1+_P
_zt = float(np.median(_pts[:, 2])); _down = np.array([0.0, 0.0, 1.0, 0.0])
_gx = np.linspace(_x0, _x1, 14); _gy = np.linspace(_y0, _y1, 14)
_path = []
for _i, _y in enumerate(_gy):
    for _x in (_gx if _i % 2 == 0 else _gx[::-1]): _path.append(np.array([_x, _y, _zt]))
for _i, _x in enumerate(_gx):
    for _y in (_gy if _i % 2 == 0 else _gy[::-1]): _path.append(np.array([_x, _y, _zt]))
for _p in _path:
    try: _j = solve_ik(_p, _down)
    except Exception: continue
    move_to_joints(_j)
