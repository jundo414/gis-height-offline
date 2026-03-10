# test_dtm_nodata.py
# DTMデータのnodata値を確認するスクリプト

import rasterio
from pathlib import Path

# 問題のある座標
test_points = [
    (26.191573, 127.663669, "那覇_4"),
    (26.184410, 127.659293, "那覇_5"),
]

project_root = Path(__file__).resolve().parents[2]
dtm_path = str(project_root / "data" / "DTM_Naha.tif")

with rasterio.open(dtm_path) as ds:
    print(f"DTM情報:")
    print(f"  ファイル: {dtm_path}")
    print(f"  CRS: {ds.crs}")
    print(f"  サイズ: {ds.width} x {ds.height}")
    print(f"  nodata値: {ds.nodata}")
    print(f"  データ型: {ds.dtypes[0]}")
    print(f"  バウンディングボックス: {ds.bounds}")
    print()
    
    from pyproj import CRS, Transformer
    transformer = Transformer.from_crs(CRS.from_epsg(4326), ds.crs, always_xy=True)
    
    for lat, lon, name in test_points:
        print(f"{name} (緯度: {lat}, 経度: {lon})")
        x, y = transformer.transform(lon, lat)
        row, col = ds.index(x, y)
        
        print(f"  変換後座標: ({x:.6f}, {y:.6f})")
        print(f"  ピクセル位置: row={row}, col={col}")
        
        if 0 <= row < ds.height and 0 <= col < ds.width:
            # 2x2ウィンドウを取得
            r0 = max(row - 1, 0)
            c0 = max(col - 1, 0)
            r1 = min(r0 + 2, ds.height)
            c1 = min(c0 + 2, ds.width)
            
            data = ds.read(1, window=((r0, r1), (c0, c1))).astype(float)
            
            print(f"  ウィンドウ: ({r0}, {r1}) x ({c0}, {c1})")
            print(f"  データ形状: {data.shape}")
            print(f"  データ値:")
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    val = data[i, j]
                    is_nodata = (ds.nodata is not None and val == ds.nodata) or (ds.nodata is None and val == 0)
                    nodata_mark = " [nodata]" if is_nodata else ""
                    print(f"    [{i}, {j}]: {val:.3f}{nodata_mark}")
            
            # 中心ピクセルの値
            center_val = data[row - r0, col - c0]
            print(f"  中心ピクセル値: {center_val:.3f}")
            if ds.nodata is not None:
                print(f"  nodata値との比較: {center_val == ds.nodata}")
        else:
            print(f"  範囲外です！")
        print()

