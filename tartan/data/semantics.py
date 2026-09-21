"""Semantic surface extraction from TartanGround packed-RGB PCD labels."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np

from tartan.data.terrain import ElevationMap, _pcd_binary_layout


GROUP_NAMES = ("ground", "sidewalk", "vegetation", "building", "steps", "vehicle", "obstacle")
GROUP_COLORS = ("#92979f", "#d8c9a7", "#55a84f", "#8b6553", "#ed8b32", "#3979c9", "#c94b45")
UNKNOWN = 255

# AirSim's fixed segmentation palette, packed as R | (G << 8) | (B << 16).
# Source: https://microsoft.github.io/AirSim/seg_rgbs.txt
AIRSIM_PACKED_RGB = (
    0, 421017, 12544368, 4749657, 4252094, 3915470, 2362705, 12824691, 1813409, 11839879, 13048349, 15667302, 9595890, 1558172, 10508593, 7658052,
    650251, 532164, 1852281, 4273408, 4600978, 9410018, 11239063, 468930, 10582221, 3945428, 13652179, 12355517, 13453366, 10353767, 8066428, 4555795,
    8711619, 11533662, 5766070, 15901274, 73159, 15011070, 16040995, 3253212, 14089814, 8455064, 6954844, 5957071, 3165053, 4863842, 15630718, 7182814,
    2267221, 2049453, 8224805, 2167610, 7813510, 7568602, 13107320, 6063073, 1071862, 15833907, 10183114, 11964856, 2943072, 8778885, 1949620, 12639745,
    6877795, 14395483, 14235201, 8536724, 13395659, 4935384, 16389354, 1625709, 1163940, 15472541, 5796510, 7214837, 2298179, 6149557, 2798506, 9747252,
    7325943, 11419161, 15735140, 9487295, 4400380, 9784817, 9249261, 5629559, 7086620, 16671310, 2007410, 15938123, 16638530, 5007406, 14215688, 6746383,
    4656733, 12713920, 10758653, 7909144, 15201209, 6416809, 9558003, 1411400, 6648224, 875734, 9669799, 11890021, 8287797, 2142467, 6504232, 10062778,
    6606680, 14914119, 12265196, 14091479, 4379410, 8794481, 4139567, 8349659, 9039929, 13862371, 13191057, 12037593, 7416058, 4488656, 16366304, 3052613,
    1332719, 14709868, 1758776, 2855859, 11320368, 3101661, 14329499, 12441918, 8041670, 11112649, 918148, 7519616, 7398307, 11640109, 9328192, 10731894,
    5185550, 11152840, 151882, 13903163, 14754633, 2613343, 14461524, 11352735, 15555345, 5529375, 16304418, 13715775, 7072641, 2651111, 6244900, 10151150,
    3593277, 10837517, 44685, 16754572, 5987701, 12192439, 4005029, 12775056, 2727436, 15363660, 7932310, 16122254, 13011174, 15285253, 5307112, 3698831,
    10241723, 4110594, 14868362, 14595962, 259494, 9176751, 13777904, 666872, 3428947, 11008223, 9834327, 7713391, 1463493, 8179947, 2968585, 3283120,
    16490394, 13594517, 1042344, 13301713, 10014032, 14015922, 2492443, 3372532, 12469355, 9160471, 11032747, 3852936, 5647878, 11566953, 12974510, 9088172,
    5344996, 12176391, 16203030, 5137641, 6898047, 10376993, 16555147, 8914730, 11756308, 14653007, 12105347, 2456430, 6326844, 6185170, 1192059, 10667401,
    332476, 9952039, 8884172, 4804601, 11681869, 5109289, 301584, 1279604, 15432196, 15100593, 816917, 6423912, 3506738, 1687582, 10921242, 5414922,
    8596330, 6805548, 14509567, 12883744, 5889237, 14995782, 5486689, 15265618, 8430843, 16059329, 10427174, 13340133, 3618946, 774803, 7785378, 16777215,
)


def _group_for_name(name: str) -> int:
    if name in {"road", "ground", "floor", "tramlines", "storesignsground"}:
        return 0
    if name == "sidewalk":
        return 1
    if name in {"tree", "plant", "bush", "leaves", "planter"}:
        return 2
    if name in {"building", "undergroundbuilding", "wall", "roof", "roofcaps", "ceiling", "awning"}:
        return 3
    if name in {"steps"}:
        return 4
    if name in {"car", "truck"}:
        return 5
    if name in {"sky", "z"}:
        return UNKNOWN
    return 6


def build_semantic_surface(
    data_root: Path,
    terrain: ElevationMap,
    cache_dir: Path,
    chunk_points: int = 2_000_000,
) -> np.ndarray:
    environment = terrain.environment
    signature = hashlib.sha1(
        np.asarray([*terrain.elevation.shape, *terrain.origin_xy, terrain.resolution], dtype=np.float64).tobytes()
    ).hexdigest()[:10]
    cache_path = cache_dir / f"{environment}_semantic_surface_v4_r{terrain.resolution:.2f}_{signature}.npz"
    if cache_path.exists():
        return np.load(cache_path)["semantic"]
    label_path = data_root / environment / "seg_label_map.json"
    name_map = json.loads(label_path.read_text(encoding="utf-8"))["name_map"]
    packed_to_group = {
        AIRSIM_PACKED_RGB[int(label_id)]: _group_for_name(name)
        for name, label_id in name_map.items()
    }

    pcd_path = data_root / environment / f"{environment}_sem.pcd"
    offset, point_count = _pcd_binary_layout(pcd_path)
    cloud = np.memmap(pcd_path, dtype=np.float32, mode="r", offset=offset, shape=(point_count, 4))
    semantic = np.full(terrain.elevation.shape, UNKNOWN, dtype=np.uint8)
    height, width = terrain.elevation.shape
    for start in range(0, point_count, chunk_points):
        points = np.asarray(cloud[start : min(start + chunk_points, point_count)])
        grid = np.floor((points[:, :2] - terrain.origin_xy[None]) / terrain.resolution).astype(np.int64)
        inside = (grid[:, 0] >= 0) & (grid[:, 1] >= 0) & (grid[:, 0] < width) & (grid[:, 1] < height)
        points, grid = points[inside], grid[inside]
        target_z = terrain.elevation[grid[:, 1], grid[:, 0]]
        surface = np.isfinite(target_z) & (np.abs(points[:, 2] - target_z) < 1e-4)
        if surface.any():
            packed = points[surface, 3].view(np.uint32)
            groups = np.asarray([packed_to_group.get(int(value), UNKNOWN) for value in packed], dtype=np.uint8)
            selected_grid = grid[surface]
            semantic[selected_grid[:, 1], selected_grid[:, 0]] = groups
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, semantic=semantic, groups=np.asarray(GROUP_NAMES))
    return semantic
