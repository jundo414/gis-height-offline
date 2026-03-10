# main_japan.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from functools import lru_cache
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

        transformer = Transformer.from_crs(CRS.from_epsg(4326), ds.crs, always_xy=True)
        x, y = transformer.transform(lon, lat)

        col_f, row_f = (~ds.transform) * (x, y)

        if not (0.0 <= col_f <= (ds.width - 1) and 0.0 <= row_f <= (ds.height - 1)):
            raise ValueError("指定した地点は DTM の範囲外です。")

        c0 = int(math.floor(col_f))
        r0 = int(math.floor(row_f))
        c0 = min(max(c0, 0), ds.width - 2) if ds.width >= 2 else 0
        r0 = min(max(r0, 0), ds.height - 2) if ds.height >= 2 else 0
        c1 = min(c0 + 1, ds.width - 1)
        r1 = min(r0 + 1, ds.height - 1)

        w_c1 = c0 + 2 if ds.width >= 2 else c0 + 1
        w_r1 = r0 + 2 if ds.height >= 2 else r0 + 1

        data = ds.read(1, window=((r0, w_r1), (c0, w_c1))).astype(np.float64)

        if data.shape != (2, 2):
            rr = int(round(row_f))
            cc = int(round(col_f))
            rr = min(max(rr, 0), ds.height - 1)
            cc = min(max(cc, 0), ds.width - 1)
            v = float(ds.read(1, window=((rr, rr + 1), (cc, cc + 1)))[0, 0])
            return v

        fx = float(np.clip(col_f - c0, 0.0, 1.0))
        fy = float(np.clip(row_f - r0, 0.0, 1.0))

        nodata = ds.nodata
        if nodata is not None:
            data = np.where(data == float(nodata), np.nan, data)

        v00, v10 = data[0, 0], data[0, 1]
        v01, v11 = data[1, 0], data[1, 1]

        if np.isnan([v00, v10, v01, v11]).any():
            candidates = []
            corners = [
                (r0, c0, v00),
                (r0, c1, v10),
                (r1, c0, v01),
                (r1, c1, v11),
            ]
            for rr, cc, vv in corners:
                if not np.isnan(vv):
                    d2 = (row_f - rr) ** 2 + (col_f - cc) ** 2
                    candidates.append((d2, float(vv)))
            if not candidates:
                raise ValueError("DTMの2x2近傍がすべて nodata でした。")
            candidates.sort(key=lambda t: t[0])
            return candidates[0][1]

        v0 = v00 * (1.0 - fx) + v10 * fx
        v1 = v01 * (1.0 - fx) + v11 * fx
        return float(v0 * (1.0 - fy) + v1 * fy)


# ============================================================
# ISG（JPGEO2024.isg）
# ============================================================
@dataclass(frozen=True)
class IsgHeader:
    data_format: str
    coord_type: str
    coord_units: str

    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    delta_lat: float
    delta_lon: float

    nrows: int
    ncols: int
    nodata: float


@dataclass
class IsgGrid:
    header: IsgHeader
    grid: np.ndarray

    def height(self, lat: float, lon: float) -> float:
        h = self.header

        if not (min(h.lat_min, h.lat_max) <= lat <= max(h.lat_min, h.lat_max)):
            raise ValueError(f"緯度がISG範囲外です: lat={lat} (範囲 {h.lat_min}..{h.lat_max})")
        if not (h.lon_min <= lon <= h.lon_max):
            raise ValueError(f"経度がISG範囲外です: lon={lon} (範囲 {h.lon_min}..{h.lon_max})")

        # 多くのISGは「行：北→南、列：西→東」
        row_f = (h.lat_max - lat) / h.delta_lat
        col_f = (lon - h.lon_min) / h.delta_lon

        r0 = int(math.floor(row_f))
        c0 = int(math.floor(col_f))
        r1 = r0 + 1
        c1 = c0 + 1

        # 端は内側へ
        if r0 < 0:
            r0, r1 = 0, 1
        if c0 < 0:
            c0, c1 = 0, 1
        if r1 >= h.nrows:
            r1 = h.nrows - 1
            r0 = r1 - 1
        if c1 >= h.ncols:
            c1 = h.ncols - 1
            c0 = c1 - 1

        fr = row_f - r0
        fc = col_f - c0

        z00 = float(self.grid[r0, c0])
        z01 = float(self.grid[r0, c1])
        z10 = float(self.grid[r1, c0])
        z11 = float(self.grid[r1, c1])

        nd = h.nodata
        if (z00 == nd) or (z01 == nd) or (z10 == nd) or (z11 == nd):
            return float("nan")

        z0 = z00 * (1.0 - fc) + z01 * fc
        z1 = z10 * (1.0 - fc) + z11 * fc
        return float(z0 * (1.0 - fr) + z1 * fr)


_RE_MULTI_SPACE = re.compile(r"\s+")
_RE_DMS = re.compile(
    r"""(?P<sign>[-+])?\s*(?P<deg>\d+)\s*°\s*(?P<min>\d+)\s*'\s*(?P<sec>\d+(?:\.\d+)?)\s*" """,
    re.VERBOSE
)


def _norm_key(k: str) -> str:
    return _RE_MULTI_SPACE.sub(" ", k.strip().lower())


def _parse_dms(s: str) -> float:
    m = _RE_DMS.match(s.strip())
    if not m:
        raise ValueError(f"dms形式として解釈できません: {s!r}")
    sign = -1.0 if m.group("sign") == "-" else 1.0
    deg = float(m.group("deg"))
    minute = float(m.group("min"))
    sec = float(m.group("sec"))
    return sign * (deg + minute / 60.0 + sec / 3600.0)


def _parse_coord_value(s: str, coord_units: str) -> float:
    u = coord_units.strip().lower()
    if u == "deg":
        return float(s)
    if u == "dms":
        return _parse_dms(s)
    raise ValueError(f"未対応の coord units です: {coord_units!r}")


def _parse_isg_header(lines: list[str]) -> IsgHeader:
    raw: dict[str, str] = {}

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if ":" in line:
            left, right = line.split(":", 1)
            raw[_norm_key(left)] = right.strip()
        elif "=" in line:
            left, right = line.split("=", 1)
            raw[_norm_key(left)] = right.strip()

    coord_units = raw.get("coord units", "deg").strip()

    def get_str(k: str, default: str | None = None) -> str:
        kk = _norm_key(k)
        if kk in raw:
            return raw[kk]
        if default is not None:
            return default
        raise ValueError(f"ISGヘッダに {k!r} が見つかりません。")

    def get_float(k: str) -> float:
        return _parse_coord_value(get_str(k), coord_units)

    def get_int(k: str) -> int:
        return int(float(get_str(k)))

    def get_num(k: str, default: float) -> float:
        kk = _norm_key(k)
        return float(raw[kk]) if kk in raw else default

    header = IsgHeader(
        data_format=get_str("data format"),
        coord_type=get_str("coord type", default="geodetic"),
        coord_units=coord_units,
        lat_min=get_float("lat min"),
        lat_max=get_float("lat max"),
        lon_min=get_float("lon min"),
        lon_max=get_float("lon max"),
        delta_lat=get_float("delta lat"),
        delta_lon=get_float("delta lon"),
        nrows=get_int("nrows"),
        ncols=get_int("ncols"),
        nodata=get_num("nodata", default=-9999.0),
    )

    if header.data_format.strip().lower() != "grid":
        raise ValueError(f"このコードは grid のISGを想定しています: data format={header.data_format!r}")
    if header.coord_type.strip().lower() != "geodetic":
        raise ValueError(f"このコードは geodetic(緯度経度) のISGを想定しています: coord type={header.coord_type!r}")

    return header


def load_isg_grid(isg_path: str | Path, *, dtype=np.float64) -> IsgGrid:
    isg_path = Path(isg_path)
    if not isg_path.exists():
        raise FileNotFoundError(f"ISGファイルが見つかりません: {isg_path}")

    all_lines = isg_path.read_text(encoding="utf-8", errors="replace").splitlines()

    begin_idx = None
    end_idx = None
    for i, line in enumerate(all_lines):
        s = line.strip().lower()
        if s.startswith("begin_of_head"):
            begin_idx = i
        elif s.startswith("end_of_head"):
            end_idx = i
            break

    if begin_idx is None or end_idx is None or end_idx <= begin_idx:
        raise ValueError("ISGヘッダ(begin_of_head〜end_of_head)を検出できませんでした。")

    header = _parse_isg_header(all_lines[begin_idx + 1 : end_idx])

    nrows, ncols = header.nrows, header.ncols
    grid = np.empty((nrows, ncols), dtype=dtype)

    row = 0
    col = 0
    for line in all_lines[end_idx + 1 :]:
        s = line.strip()
        if not s or s.startswith("#"):
            continue

        vals = np.fromstring(s, sep=" ")
        if vals.size == 0:
            continue

        for v in vals:
            grid[row, col] = v
            col += 1
            if col >= ncols:
                col = 0
                row += 1
                if row >= nrows:
                    break
        if row >= nrows:
            break

    if row < nrows:
        raise ValueError(f"ISGデータ部の値が不足しています。期待行数={nrows}, 読み込み完了行={row}")

    return IsgGrid(header=header, grid=grid)


@lru_cache(maxsize=4)
def _cached_isg(isg_path_str: str) -> IsgGrid:
    return load_isg_grid(isg_path_str, dtype=np.float64)


def geoid_undulation_N_isg(lat: float, lon: float, *, isg_path: str | Path) -> float:
    return _cached_isg(str(isg_path)).height(lat, lon)


# ============================================================
# 統合計算（地面のみ）
# ============================================================
def compute_heights(
    lat: float,
    lon: float,
    *,
    dtm_path: str,
    vertical_datum: VerticalDatum,
    isg_path: str,
) -> HeightsResult:
    dtm_value = sample_dtm_height(dtm_path, lat, lon)

    N = geoid_undulation_N_isg(lat, lon, isg_path=isg_path)
    if math.isnan(N):
        raise ValueError("ジオイド高NがNaNになりました（範囲外 or nodata の可能性）。")

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
    print("===== 計算結果（日本：JPGEO2024 + DTM） =====")
    print(f"緯度・経度                 : {res.lat}, {res.lon}")
    print(f"ジオイド高 N [m]          : {res.N_geoid:.4f}")
    print(f"地面 正高(MSL) H [m]      : {res.H_ground_msl:.3f}")
    print(f"地面 楕円体高 h [m]       : {res.h_ground_ellipsoid:.3f}")


if __name__ == "__main__":
    dtm_path = "./data/DTM_Naha.tif"
    isg_path = "./data/JPGEO2024.isg"
    vertical_datum: VerticalDatum = "DTM_IS_MSL"

    lat = 26.186898142655217
    lon = 127.8151512771811

    res = compute_heights(
        lat, lon,
        dtm_path=dtm_path,
        vertical_datum=vertical_datum,
        isg_path=isg_path,
    )
    print_result(res)
