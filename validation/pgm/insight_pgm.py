import numpy as np
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
pgm_path = project_root / "data" / "egm2008-1.pgm"

with open(pgm_path, "rb") as f:
    # 1) マジックナンバー
    magic = f.readline().strip()
    if magic != b"P5":
        raise RuntimeError("P5形式ではありません")

    comments = []
    tokens = []

    # 2) width height maxval を取得
    while len(tokens) < 3:
        line = f.readline()
        if line.startswith(b"#"):
            comments.append(line.decode("ascii", "ignore"))
        else:
            tokens.extend(line.split())

    nlon = int(tokens[0])
    nlat = int(tokens[1])
    maxval = int(tokens[2])

    # 3) ここが「本当の」バイナリ開始位置
    data_offset = f.tell()

print("nlat =", nlat)
print("nlon =", nlon)
print("maxval =", maxval)
print("data_offset =", data_offset)

# 4) dtype を決める
dtype = ">u2" if maxval > 255 else "u1"

# 5) memmap
raw = np.memmap(
    pgm_path,
    dtype=dtype,
    mode="r",
    offset=data_offset,
    shape=(nlat, nlon)
)

print("raw shape:", raw.shape)
print("sample raw value:", raw[nlat // 2, nlon // 2])

