# check_dtm_bounds.py
# DTMデータの有効範囲を確認するスクリプト

import rasterio
import numpy as np
from pyproj import CRS, Transformer
from pathlib import Path

def check_dtm_bounds(dtm_path: str):
    """DTMデータの有効範囲を確認"""
    
    with rasterio.open(dtm_path) as ds:
        print("=" * 70)
        print(f"DTM情報: {dtm_path}")
        print("=" * 70)
        
        # 基本情報
        print(f"\n【基本情報】")
        print(f"  CRS: {ds.crs}")
        print(f"  サイズ: {ds.width} x {ds.height}")
        print(f"  データ型: {ds.dtypes[0]}")
        print(f"  nodata値: {ds.nodata}")
        
        # バウンディングボックス（DTMのCRS座標系）
        bounds = ds.bounds
        print(f"\n【バウンディングボックス（DTM座標系）】")
        print(f"  左（West）: {bounds.left:.6f}")
        print(f"  下（South）: {bounds.bottom:.6f}")
        print(f"  右（East）: {bounds.right:.6f}")
        print(f"  上（North）: {bounds.top:.6f}")
        
        # WGS84に変換
        if ds.crs is not None and ds.crs != CRS.from_epsg(4326):
            transformer = Transformer.from_crs(ds.crs, CRS.from_epsg(4326), always_xy=True)
            
            # 4隅の座標を変換
            corners = [
                (bounds.left, bounds.bottom),  # 左下
                (bounds.right, bounds.bottom),  # 右下
                (bounds.right, bounds.top),    # 右上
                (bounds.left, bounds.top),     # 左上
            ]
            
            print(f"\n【バウンディングボックス（WGS84）】")
            wgs84_corners = []
            for x, y in corners:
                lon, lat = transformer.transform(x, y)
                wgs84_corners.append((lon, lat))
                print(f"  ({x:.6f}, {y:.6f}) -> (経度: {lon:.6f}, 緯度: {lat:.6f})")
            
            # 範囲を計算
            lons = [c[0] for c in wgs84_corners]
            lats = [c[1] for c in wgs84_corners]
            
            print(f"\n【有効範囲（WGS84）】")
            print(f"  経度範囲: {min(lons):.6f} ～ {max(lons):.6f}")
            print(f"  緯度範囲: {min(lats):.6f} ～ {max(lats):.6f}")
        else:
            # 既にWGS84の場合
            print(f"\n【有効範囲（WGS84）】")
            print(f"  経度範囲: {bounds.left:.6f} ～ {bounds.right:.6f}")
            print(f"  緯度範囲: {bounds.bottom:.6f} ～ {bounds.top:.6f}")
        
        # nodata値の分布を確認
        print(f"\n【データ品質チェック】")
        if ds.nodata is not None:
            # サンプリングしてnodata値の割合を確認
            sample_size = min(10000, ds.width * ds.height)
            step = max(1, (ds.width * ds.height) // sample_size)
            
            nodata_count = 0
            total_count = 0
            valid_values = []
            
            for i in range(0, ds.height, step):
                for j in range(0, ds.width, step):
                    val = ds.read(1, window=((i, i+1), (j, j+1)))[0, 0]
                    total_count += 1
                    if val == ds.nodata or np.isnan(val):
                        nodata_count += 1
                    else:
                        valid_values.append(float(val))
            
            nodata_ratio = nodata_count / total_count * 100
            print(f"  サンプリング数: {total_count}")
            print(f"  nodata値の割合: {nodata_ratio:.2f}%")
            
            if valid_values:
                print(f"  有効値の範囲: {min(valid_values):.3f} ～ {max(valid_values):.3f} m")
                print(f"  有効値の平均: {np.mean(valid_values):.3f} m")
        else:
            print(f"  nodata値が定義されていません")
        
        # 解像度
        transform = ds.transform
        print(f"\n【解像度】")
        print(f"  ピクセルサイズ（X方向）: {abs(transform[0]):.6f}")
        print(f"  ピクセルサイズ（Y方向）: {abs(transform[4]):.6f}")
        
        # テスト地点の確認
        print(f"\n【テスト地点の確認】")
        test_points = [
            (26.186898142655217, 127.8151512771811, "那覇_1"),
            (26.20, 127.70, "那覇_2"),
            (26.22, 127.80, "那覇_3"),
            (26.191573, 127.663669, "那覇_4（問題あり）"),
            (26.184410, 127.659293, "那覇_5（問題あり）"),
        ]
        
        if ds.crs is not None:
            transformer = Transformer.from_crs(CRS.from_epsg(4326), ds.crs, always_xy=True)
            
            for lat, lon, name in test_points:
                x, y = transformer.transform(lon, lat)
                row, col = ds.index(x, y)
                
                in_bounds = (0 <= row < ds.height and 0 <= col < ds.width)
                
                if in_bounds:
                    # 周辺の値を確認
                    r0 = max(row - 1, 0)
                    c0 = max(col - 1, 0)
                    r1 = min(r0 + 2, ds.height)
                    c1 = min(c0 + 2, ds.width)
                    
                    data = ds.read(1, window=((r0, r1), (c0, c1))).astype(float)
                    
                    has_nodata = False
                    if ds.nodata is not None:
                        has_nodata = np.any(data == ds.nodata)
                    
                    center_val = data[row - r0, col - c0]
                    
                    status = "OK"
                    if has_nodata:
                        status = "nodata値あり"
                    elif center_val == 0.0 and ds.nodata != 0.0:
                        status = "値が0.0（要確認）"
                    
                    print(f"  {name}: 範囲内, 値={center_val:.3f} m, ステータス={status}")
                else:
                    print(f"  {name}: 範囲外 (row={row}, col={col})")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    check_dtm_bounds(str(project_root / "data" / "DTM_Naha.tif"))

