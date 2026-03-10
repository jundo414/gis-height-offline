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
# DTM（GeoTIFF）から地面高さ取得（標準的バイリニア補間）
#   - 位置(x,y)を「浮動小数のピクセル座標(col_f,row_f)」に変換
#   - floorを左上(c0,r0)として2x2を取り、小数部(fx,fy)で補間
#   - nodataが混ざる場合は近傍の有効値へフォールバック
# ============================================================
def sample_dtm_height(dtm_path: str, lat: float, lon: float) -> float:
    with rasterio.open(dtm_path) as ds:
        if ds.crs is None:
            raise ValueError("DTM GeoTIFF に CRS が定義されていません。")

        # WGS84 -> DTM CRS
        transformer = Transformer.from_crs(CRS.from_epsg(4326), ds.crs, always_xy=True)
        x, y = transformer.transform(lon, lat)

        # (x,y) -> (col,row) 浮動小数ピクセル座標
        # 注意: rasterio transform は (col,row)->(x,y) の写像。逆変換で (col,row) を得る。
        col_f, row_f = (~ds.transform) * (x, y)

        # 範囲チェック（厳密：外はエラー）
        if not (0.0 <= col_f <= (ds.width - 1) and 0.0 <= row_f <= (ds.height - 1)):
            raise ValueError("指定した地点は DTM の範囲外です。")

        # 2x2を必ず確保するため、端は内側に寄せる
        # c0 in [0, width-2], r0 in [0, height-2]
        c0 = int(math.floor(col_f))
        r0 = int(math.floor(row_f))
        c0 = min(max(c0, 0), ds.width - 2) if ds.width >= 2 else 0
        r0 = min(max(r0, 0), ds.height - 2) if ds.height >= 2 else 0
        c1 = min(c0 + 1, ds.width - 1)
        r1 = min(r0 + 1, ds.height - 1)

        # 2x2を読み込む（Windowは [start, stop) なので +2）
        # ただし幅/高さが1のラスタにも一応対応
        w_c1 = c0 + 2 if ds.width >= 2 else c0 + 1
        w_r1 = r0 + 2 if ds.height >= 2 else r0 + 1

        data = ds.read(1, window=((r0, w_r1), (c0, w_c1))).astype(np.float64)

        # 1x1しか取れないなど、2x2にならない場合は最近傍
        if data.shape != (2, 2):
            # 最近傍：col_f,row_f を丸める
            rr = int(round(row_f))
            cc = int(round(col_f))
            rr = min(max(rr, 0), ds.height - 1)
            cc = min(max(cc, 0), ds.width - 1)
            v = float(ds.read(1, window=((rr, rr + 1), (cc, cc + 1)))[0, 0])
            return v

        # 重み（小数部）
        fx = float(np.clip(col_f - c0, 0.0, 1.0))
        fy = float(np.clip(row_f - r0, 0.0, 1.0))

        # nodata処理（あれば NaN 扱いにする）
        nodata = ds.nodata
        if nodata is not None:
            data = np.where(data == float(nodata), np.nan, data)

        # 4点の値（row方向が下、col方向が右）
        v00, v10 = data[0, 0], data[0, 1]
        v01, v11 = data[1, 0], data[1, 1]

        # nodata混入時：近傍の有効値にフォールバック
        if np.isnan([v00, v10, v01, v11]).any():
            candidates = []
            # 各コーナーのピクセル座標
            corners = [
                (r0, c0, v00),
                (r0, c1, v10),
                (r1, c0, v01),
                (r1, c1, v11),
            ]
            for rr, cc, vv in corners:
                if not np.isnan(vv):
                    # ピクセル座標空間で距離
                    d2 = (row_f - rr) ** 2 + (col_f - cc) ** 2
                    candidates.append((d2, float(vv)))
            if not candidates:
                raise ValueError("DTMの2x2近傍がすべて nodata でした。")
            candidates.sort(key=lambda t: t[0])
            return candidates[0][1]

        # 標準的バイリニア補間
        v0 = v00 * (1.0 - fx) + v10 * fx
        v1 = v01 * (1.0 - fx) + v11 * fx
        return float(v0 * (1.0 - fy) + v1 * fy)


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
    dtm_path = "./data/DTM_Naha.tif"
    geoid_pgm_path = "./data/egm2008-1.pgm"
    vertical_datum: VerticalDatum = "DTM_IS_MSL"

    lat = 26.186898142655217
    lon = 127.8151512771811

    res = compute_heights(
        lat, lon,
        dtm_path=dtm_path,
        vertical_datum=vertical_datum,
        geoid_pgm_path=geoid_pgm_path,
    )
    print_result(res)
