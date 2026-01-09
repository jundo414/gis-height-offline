# main_global.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Tuple

import numpy as np
import rasterio
from pyproj import CRS, Transformer


# ============================================================
# DTMの高さ基準
# ============================================================
VerticalDatum = Literal["DTM_IS_MSL", "DTM_IS_ELLIPSOID"]


# ============================================================
# 出力（地面 + ジオイドのみ）
# ============================================================
@dataclass
class HeightsResult:
    lat: float
    lon: float
    H_ground_msl: float
    h_ground_ellipsoid: float
    N_geoid: float


# ============================================================
# DTM（GeoTIFF）から地面高さ取得（バイリニア補間固定）
# ============================================================
def sample_dtm_height(dtm_path: str, lat: float, lon: float) -> float:
    with rasterio.open(dtm_path) as ds:
        if ds.crs is None:
            raise ValueError("DTM GeoTIFF に CRS が定義されていません。")

        transformer = Transformer.from_crs(CRS.from_epsg(4326), ds.crs, always_xy=True)
        x, y = transformer.transform(lon, lat)

        row, col = ds.index(x, y)
        if not (0 <= row < ds.height and 0 <= col < ds.width):
            raise ValueError("指定した地点は DTM の範囲外です。")

        r0 = max(row - 1, 0)
        c0 = max(col - 1, 0)
        r1 = min(r0 + 2, ds.height)
        c1 = min(c0 + 2, ds.width)

        data = ds.read(1, window=((r0, r1), (c0, c1))).astype(np.float64)

        if data.shape != (2, 2):
            return float(data[0, 0])

        def pixel_center(rr: int, cc: int) -> Tuple[float, float]:
            px, py = rasterio.transform.xy(ds.transform, rr, cc, offset="center")
            return float(px), float(py)

        x00, y00 = pixel_center(r0, c0)
        x11, y11 = pixel_center(r0 + 1, c0 + 1)

        fx = (x - x00) / (x11 - x00 if x11 != x00 else 1.0)
        fy = (y - y00) / (y11 - y00 if y11 != y00 else 1.0)
        fx = float(np.clip(fx, 0.0, 1.0))
        fy = float(np.clip(fy, 0.0, 1.0))

        v00, v10 = data[0, 0], data[0, 1]
        v01, v11 = data[1, 0], data[1, 1]

        v0 = v00 * (1 - fx) + v10 * fx
        v1 = v01 * (1 - fx) + v11 * fx
        return float(v0 * (1 - fy) + v1 * fy)


# ============================================================
# EGM2008 (.pgm) からジオイド高 N を計算（memmap）
# ============================================================
_GEOID_CACHE: dict[str, dict] = {}

_RE_SCALE = re.compile(r"Scale\s+([+-]?\d+(?:\.\d+)?)", re.IGNORECASE)
_RE_OFFSET = re.compile(r"Offset\s+([+-]?\d+(?:\.\d+)?)", re.IGNORECASE)


def _read_geoid_pgm(pgm_path: str) -> dict:
    path = str(Path(pgm_path).resolve())
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"ジオイドPGMが見つかりません: {path}")

    with p.open("rb") as f:
        magic = f.readline().strip()
        if magic != b"P5":
            raise ValueError(f"対応していない PGM 形式です: {magic!r}（P5のみ対応）")

        comments: list[str] = []
        tokens: list[bytes] = []

        while len(tokens) < 3:
            line = f.readline()
            if not line:
                raise ValueError("PGMヘッダが途中で終わりました（EOF）。")
            if line.startswith(b"#"):
                comments.append(line.decode("ascii", "ignore").strip())
            else:
                tokens.extend(line.split())

        nlon = int(tokens[0])
        nlat = int(tokens[1])
        maxval = int(tokens[2])
        data_offset = f.tell()

    scale = None
    offset = None
    for c in comments:
        m = _RE_SCALE.search(c)
        if m:
            scale = float(m.group(1))
        m = _RE_OFFSET.search(c)
        if m:
            offset = float(m.group(1))

    if scale is None or offset is None:
        raise ValueError("PGMコメントから Scale / Offset を取得できませんでした。")

    dtype = np.dtype(np.uint8) if maxval <= 255 else np.dtype(">u2")

    raw = np.memmap(
        path,
        dtype=dtype,
        mode="r",
        offset=data_offset,
        shape=(nlat, nlon),
    )

    dlat = 180.0 / (nlat - 1)
    dlon = 360.0 / nlon

    return {
        "raw": raw,
        "scale": scale,
        "offset": offset,
        "nlat": nlat,
        "nlon": nlon,
        "dlat": dlat,
        "dlon": dlon,
    }


def geoid_undulation_N_pgm(lat: float, lon: float, *, geoid_pgm_path: str) -> float:
    key = str(Path(geoid_pgm_path).resolve())
    if key not in _GEOID_CACHE:
        _GEOID_CACHE[key] = _read_geoid_pgm(key)

    g = _GEOID_CACHE[key]

    lon = lon % 360.0
    lat = max(-90.0, min(90.0, lat))

    i = (90.0 - lat) / g["dlat"]
    j = lon / g["dlon"]

    i0 = min(max(int(math.floor(i)), 0), g["nlat"] - 2)
    i1 = i0 + 1
    j0 = int(math.floor(j)) % g["nlon"]
    j1 = (j0 + 1) % g["nlon"]

    fi = i - i0
    fj = j - math.floor(j)

    scale = g["scale"]
    offset = g["offset"]
    raw = g["raw"]

    z00 = raw[i0, j0] * scale + offset
    z10 = raw[i0, j1] * scale + offset
    z01 = raw[i1, j0] * scale + offset
    z11 = raw[i1, j1] * scale + offset

    return float(
        z00 * (1 - fi) * (1 - fj)
        + z10 * (1 - fi) * fj
        + z01 * fi * (1 - fj)
        + z11 * fi * fj
    )


# ============================================================
# 統合計算（地面のみ）
# ============================================================
def compute_heights(
    lat: float,
    lon: float,
    *,
    dtm_path: str,
    vertical_datum: VerticalDatum,
    geoid_pgm_path: str,
) -> HeightsResult:
    dtm_value = sample_dtm_height(dtm_path, lat, lon)
    N = geoid_undulation_N_pgm(lat, lon, geoid_pgm_path=geoid_pgm_path)

    if vertical_datum == "DTM_IS_MSL":
        H_ground = dtm_value
        h_ground = H_ground + N
    elif vertical_datum == "DTM_IS_ELLIPSOID":
        h_ground = dtm_value
        H_ground = h_ground - N
    else:
        raise ValueError(f"未知の vertical_datum です: {vertical_datum}")

    return HeightsResult(
        lat=lat,
        lon=lon,
        H_ground_msl=float(H_ground),
        h_ground_ellipsoid=float(h_ground),
        N_geoid=float(N),
    )


def print_result(res: HeightsResult) -> None:
    print("===== 計算結果（グローバル：EGM2008 + DTM） =====")
    print(f"緯度・経度                 : {res.lat}, {res.lon}")
    print(f"ジオイド高 N [m]          : {res.N_geoid:.4f}")
    print(f"地面 正高(MSL) H [m]      : {res.H_ground_msl:.3f}")
    print(f"地面 楕円体高 h [m]       : {res.h_ground_ellipsoid:.3f}")


if __name__ == "__main__":
    dtm_path = "../opt-route/data_v2/DTM.tiff"
    geoid_pgm_path = "./data/egm2008-1.pgm"
    vertical_datum: VerticalDatum = "DTM_IS_MSL"

    lat = 24.77236672963945
    lon = 125.3419969747311

    res = compute_heights(
        lat, lon,
        dtm_path=dtm_path,
        vertical_datum=vertical_datum,
        geoid_pgm_path=geoid_pgm_path,
    )
    print_result(res)

