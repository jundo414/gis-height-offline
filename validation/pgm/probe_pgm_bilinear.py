import math
import re
import numpy as np
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
pgm_path = project_root / "data" / "egm2008-1.pgm"

lat = 26.186898142655217
lon = 127.8151512771811

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

scale = offset = None
for c in comments:
    m = re.search(r"Scale\s+([+-]?\d+(?:\.\d+)?)", c, re.IGNORECASE)
    if m: scale = float(m.group(1))
    m = re.search(r"Offset\s+([+-]?\d+(?:\.\d+)?)", c, re.IGNORECASE)
    if m: offset = float(m.group(1))
if scale is None or offset is None:
    raise RuntimeError("Scale/Offset が見つかりません")

dtype = ">u2" if maxval > 255 else "u1"
raw = np.memmap(pgm_path, dtype=dtype, mode="r", offset=data_offset, shape=(nlat, nlon))

dlat = 180.0 / (nlat - 1)
dlon = 360.0 / nlon

lon0_360 = lon % 360.0
i = (90.0 - lat) / dlat
j = lon0_360 / dlon

i0 = min(max(int(math.floor(i)), 0), nlat - 2)
j0 = int(math.floor(j)) % nlon
i1 = i0 + 1
j1 = (j0 + 1) % nlon

fi = i - i0
fj = j - math.floor(j)

z00 = raw[i0, j0] * scale + offset
z10 = raw[i0, j1] * scale + offset
z01 = raw[i1, j0] * scale + offset
z11 = raw[i1, j1] * scale + offset

N = (
    z00 * (1 - fi) * (1 - fj)
    + z10 * (1 - fi) * fj
    + z01 * fi * (1 - fj)
    + z11 * fi * fj
)

print("===== PGM 手動プローブ（バイリニア補間）=====")
print(f"入力 lat,lon           : {lat}, {lon} (lon正規化={lon0_360})")
print(f"dlat,dlon [deg]        : {dlat}, {dlon}")
print(f"連続インデックス i,j   : {i}, {j}")
print(f"セル i0,i1 / j0,j1      : {i0},{i1} / {j0},{j1}")
print(f"係数 fi,fj             : {fi}, {fj}")
print("")
print(f"z00(左上) [m]           : {float(z00)}")
print(f"z10(右上) [m]           : {float(z10)}")
print(f"z01(左下) [m]           : {float(z01)}")
print(f"z11(右下) [m]           : {float(z11)}")
print("")
print(f"補間ジオイド高 N [m]     : {float(N)}")

