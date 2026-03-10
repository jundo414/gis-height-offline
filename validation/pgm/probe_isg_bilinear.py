# probe_isg_bilinear.py
# -*- coding: utf-8 -*-

"""
JPGEO2024.isg（ISG / grid）から、緯度経度（lat, lon）における
ジオイド高 N [m] をオフラインで取得する最小実装。

- ISG ヘッダ（begin_of_head〜end_of_head）をパース
  - coord units = deg / dms の両方に対応（15°00'00" など）
- データ部を 2次元格子（nrows×ncols）として読み込む
- 緯度経度 -> 格子上の連続インデックスへ変換し、バイリニア補間で N を返す

注意：
- これは「ジオイド高 N」を出すコードです。
  標高（DTM）を読む処理は含みません。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np


# ============================================================
# ISG ヘッダ構造
# ============================================================

@dataclass(frozen=True)
class IsgHeader:
    model_name: str
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

    raw: Dict[str, str]


@dataclass
class IsgGrid:
    header: IsgHeader
    grid: np.ndarray  # shape=(nrows, ncols), dtype=float64

    def debug_N(self, lat: float, lon: float) -> float:
        h = self.header

        print("----- ISG デバッグ -----")
        print(f"入力 lat, lon          : {lat}, {lon}")

        row_f = (h.lat_max - lat) / h.delta_lat
        col_f = (lon - h.lon_min) / h.delta_lon

        r0 = int(math.floor(row_f))
        c0 = int(math.floor(col_f))
        r1 = r0 + 1
        c1 = c0 + 1

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

        print(f"row_f, col_f           : {row_f}, {col_f}")
        print(f"r0,r1 / c0,c1          : {r0},{r1} / {c0},{c1}")
        print(f"fr, fc                : {fr}, {fc}")
        print("")
        print(f"z00 (左上) [m]         : {z00}")
        print(f"z01 (右上) [m]         : {z01}")
        print(f"z10 (左下) [m]         : {z10}")
        print(f"z11 (右下) [m]         : {z11}")

        z0 = z00 * (1.0 - fc) + z01 * fc
        z1 = z10 * (1.0 - fc) + z11 * fc
        N = z0 * (1.0 - fr) + z1 * fr

        print("")
        print(f"z_top                 : {z0}")
        print(f"z_bottom              : {z1}")
        print(f"補間ジオイド高 N [m]   : {N}")
        print("------------------------")

        return N

    def N(self, lat: float, lon: float) -> float:
        """
        緯度経度 (lat, lon) のジオイド高 N[m] をバイリニア補間して返す。
        """
        h = self.header

        # 範囲チェック（lat_min <= lat <= lat_max の順とは限らない想定で両対応）
        lat_lo = min(h.lat_min, h.lat_max)
        lat_hi = max(h.lat_min, h.lat_max)
        if not (lat_lo <= lat <= lat_hi):
            raise ValueError(f"緯度がISG範囲外です: lat={lat} (範囲 {lat_lo}..{lat_hi})")

        lon_lo = min(h.lon_min, h.lon_max)
        lon_hi = max(h.lon_min, h.lon_max)
        if not (lon_lo <= lon <= lon_hi):
            raise ValueError(f"経度がISG範囲外です: lon={lon} (範囲 {lon_lo}..{lon_hi})")

        # ISGは典型的に「row=0 が北端(lat_max)、row増で南へ」
        row_f = (h.lat_max - lat) / h.delta_lat
        col_f = (lon - h.lon_min) / h.delta_lon

        r0 = int(math.floor(row_f))
        c0 = int(math.floor(col_f))
        r1 = r0 + 1
        c1 = c0 + 1

        # 端の調整：必ず2×2が取れるよう内側に寄せる
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

        fr = row_f - r0  # 0..1 目安
        fc = col_f - c0  # 0..1 目安

        z00 = float(self.grid[r0, c0])  # 左上
        z01 = float(self.grid[r0, c1])  # 右上
        z10 = float(self.grid[r1, c0])  # 左下
        z11 = float(self.grid[r1, c1])  # 右下

        nd = h.nodata
        if (z00 == nd) or (z01 == nd) or (z10 == nd) or (z11 == nd):
            return float("nan")

        # バイリニア補間
        z0 = z00 * (1.0 - fc) + z01 * fc
        z1 = z10 * (1.0 - fc) + z11 * fc
        return float(z0 * (1.0 - fr) + z1 * fr)


# ============================================================
# ISG ヘッダパース（deg / dms 両対応）
# ============================================================

_RE_MULTI_SPACE = re.compile(r"\s+")
_RE_DMS = re.compile(
    r"""
    (?P<sign>[-+])?
    \s*(?P<deg>\d+)\s*°\s*
    (?P<min>\d+)\s*'\s*
    (?P<sec>\d+(?:\.\d+)?)\s*"
    """,
    re.VERBOSE
)

def _norm_key(k: str) -> str:
    k = k.strip().lower()
    k = _RE_MULTI_SPACE.sub(" ", k)
    return k

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
    raw: Dict[str, str] = {}

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

    def get_str(k: str, default: Optional[str] = None) -> str:
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
        if kk not in raw:
            return default
        return float(raw[kk])

    header = IsgHeader(
        model_name=get_str("model name", default=""),
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
        raw=raw,
    )

    if header.data_format.strip().lower() != "grid":
        raise ValueError(f"このコードは grid のISGを想定しています: data format={header.data_format!r}")
    if header.coord_type.strip().lower() != "geodetic":
        raise ValueError(f"このコードは geodetic(緯度経度) のISGを想定しています: coord type={header.coord_type!r}")

    return header


# ============================================================
# ISG 読み込み（巨大テキスト）→格子配列
# ============================================================

def load_isg(isg_path: str | Path, *, dtype=np.float64) -> IsgGrid:
    """
    ISG（JPGEO2024.isg 等）を読み込んで IsgGrid を返す。
    """
    isg_path = Path(isg_path)
    if not isg_path.exists():
        raise FileNotFoundError(f"ISGファイルが見つかりません: {isg_path}")

    lines = isg_path.read_text(encoding="utf-8", errors="replace").splitlines()

    begin = end = None
    for i, line in enumerate(lines):
        s = line.strip().lower()
        if s.startswith("begin_of_head"):
            begin = i
        elif s.startswith("end_of_head"):
            end = i
            break

    if begin is None or end is None or end <= begin:
        raise ValueError("ISGヘッダ(begin_of_head〜end_of_head)を検出できませんでした。")

    header = _parse_isg_header(lines[begin + 1 : end])

    nrows, ncols = header.nrows, header.ncols
    grid = np.empty((nrows, ncols), dtype=dtype)

    row = 0
    col = 0

    # データ部（end_of_head の次行以降）
    for line in lines[end + 1 :]:
        s = line.strip()
        if not s:
            continue
        if s.startswith("#"):
            continue

        vals = np.fromstring(s, sep=" ", dtype=np.float64)
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
        raise ValueError(
            f"ISGデータ部の値が不足しています。期待行数={nrows}, 読み込み完了行={row}"
        )

    return IsgGrid(header=header, grid=grid)


@lru_cache(maxsize=2)
def _cached_isg(isg_abs_path: str) -> IsgGrid:
    """
    同一プロセス内で ISG の読み込みを1回にするためのキャッシュ。
    """
    return load_isg(isg_abs_path, dtype=np.float64)


# ============================================================
# 外部APIっぽい関数：lat/lon -> N[m]
# ============================================================

def geoid_N_from_isg(lat: float, lon: float, *, isg_path: str | Path) -> float:
    isg_abs = str(Path(isg_path).resolve())
    g = _cached_isg(isg_abs)
    return g.N(lat, lon)


# ============================================================
# 実行例
# ============================================================

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    isg_path = project_root / "data" / "JPGEO2024.isg"

    lat = 26.186898142655217
    lon = 127.8151512771811

    grid = load_isg(isg_path)

    N = grid.debug_N(lat, lon)
