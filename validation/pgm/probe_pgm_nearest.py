import math
import re
import numpy as np
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
pgm_path = project_root / "data" / "egm2008-1.pgm"

# ===== ユーザー入力（ここだけ変える）=====
lat = 26.186898142655217
lon = 127.8151512771811
# =========================================

# ---- PGMヘッダを読み取る（既に成功してる手順）----
with open(pgm_path, "rb") as f:
    magic = f.readline().strip()
    if magic != b"P5":
        raise RuntimeError("P5形式ではありません")

    comments = []
    tokens = []
    while len(tokens) < 3:
        line = f.readline()
        if line.startswith(b"#"):
            comments.append(line.decode("ascii", "ignore").strip())
        else:
            tokens.extend(line.split())

    nlon = int(tokens[0])
    nlat = int(tokens[1])
    maxval = int(tokens[2])
    data_offset = f.tell()

# ---- Scale/Offsetをコメントから抜く ----
scale = None
offset = None
for c in comments:
    m = re.search(r"Scale\s+([+-]?\d+(?:\.\d+)?)", c, re.IGNORECASE)
    if m:
        scale = float(m.group(1))
    m = re.search(r"Offset\s+([+-]?\d+(?:\.\d+)?)", c, re.IGNORECASE)
    if m:
        offset = float(m.group(1))

if scale is None or offset is None:
    raise RuntimeError("Scale/Offset が見つかりません")

# ---- memmap ----
dtype = ">u2" if maxval > 255 else "u1"
raw = np.memmap(pgm_path, dtype=dtype, mode="r", offset=data_offset, shape=(nlat, nlon))

# ---- 格子間隔 ----
dlat = 180.0 / (nlat - 1)
dlon = 360.0 / nlon

# ---- 連続インデックス ----
lon0_360 = lon % 360.0
i = (90.0 - lat) / dlat
j = lon0_360 / dlon

# ---- 最近傍（手動で分かりやすい）----
ii = int(round(i))
jj = int(round(j)) % nlon

raw_val = int(raw[ii, jj])
N = raw_val * scale + offset

print("===== PGM 手動プローブ（最近傍）=====")
print(f"入力 lat,lon           : {lat}, {lon}")
print(f"nlat,nlon              : {nlat}, {nlon}")
print(f"dlat,dlon [deg]        : {dlat}, {dlon}")
print(f"連続インデックス i,j   : {i}, {j}")
print(f"最近傍 index ii,jj      : {ii}, {jj}")
print(f"raw[ii,jj]             : {raw_val}")
print(f"Scale,Offset           : {scale}, {offset}")
print(f"ジオイド高 N [m]       : {N}")

