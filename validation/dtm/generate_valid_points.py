# generate_valid_points.py
# DTMデータの有効範囲内で、有効なデータが存在する地点をランダムに生成

import rasterio
import numpy as np
import random
from pyproj import CRS, Transformer
from pathlib import Path

def find_valid_points(dtm_path: str, num_points: int = 10, min_elevation: float = 0.1):
    """
    DTMデータの有効範囲内で、有効なデータが存在する地点をランダムに生成
    
    Args:
        dtm_path: DTMファイルのパス
        num_points: 生成する地点数
        min_elevation: 有効とみなす最小標高値（m）
    """
    
    with rasterio.open(dtm_path) as ds:
        print(f"DTM情報: {dtm_path}")
        print(f"  サイズ: {ds.width} x {ds.height}")
        print(f"  有効範囲: 経度 {ds.bounds.left:.6f} ～ {ds.bounds.right:.6f}, "
              f"緯度 {ds.bounds.bottom:.6f} ～ {ds.bounds.top:.6f}")
        print()
        
        # 有効なデータポイントを探す
        print("有効なデータポイントを探索中...")
        valid_pixels = []
        
        # サンプリングして有効なピクセルを探す
        sample_step = max(1, min(ds.width, ds.height) // 100)  # 適度なサンプリング
        
        for row in range(0, ds.height, sample_step):
            for col in range(0, ds.width, sample_step):
                val = ds.read(1, window=((row, row+1), (col, col+1)))[0, 0]
                
                # 有効な値かチェック
                is_valid = True
                if ds.nodata is not None and val == ds.nodata:
                    is_valid = False
                elif abs(val) < min_elevation:  # 0.0に近い値は除外
                    is_valid = False
                elif np.isnan(val):
                    is_valid = False
                
                if is_valid:
                    # ピクセル中心の座標を取得
                    x, y = rasterio.transform.xy(ds.transform, row, col, offset="center")
                    valid_pixels.append((row, col, x, y, float(val)))
        
        print(f"  有効なピクセル数: {len(valid_pixels)}")
        
        if len(valid_pixels) == 0:
            print("警告: 有効なデータポイントが見つかりませんでした。")
            return []
        
        # ランダムに地点を選択
        selected = random.sample(valid_pixels, min(num_points, len(valid_pixels)))
        
        # WGS84に変換
        if ds.crs is not None and ds.crs != CRS.from_epsg(4326):
            transformer = Transformer.from_crs(ds.crs, CRS.from_epsg(4326), always_xy=True)
        else:
            transformer = None
        
        results = []
        print(f"\n選択された地点（{len(selected)}地点）:")
        print("-" * 70)
        print(f"{'地点名':<12} {'緯度':<15} {'経度':<15} {'標高[m]':<10}")
        print("-" * 70)
        
        for i, (row, col, x, y, elevation) in enumerate(selected, 1):
            if transformer:
                lon, lat = transformer.transform(x, y)
            else:
                lon, lat = x, y
            
            location_name = f"那覇_{i}"
            results.append((lat, lon, location_name))
            
            print(f"{location_name:<12} {lat:>14.6f} {lon:>14.6f} {elevation:>9.3f}")
        
        return results


def generate_test_locations(dtm_path: str, num_points: int = 5):
    """比較スクリプト用のテスト地点を生成"""
    
    valid_points = find_valid_points(dtm_path, num_points=num_points)
    
    if not valid_points:
        return []
    
    # 比較スクリプト用の形式で出力
    print("\n" + "=" * 70)
    print("比較スクリプト用のテスト地点（コピーして使用）:")
    print("=" * 70)
    print()
    print("test_locations = [")
    for lat, lon, name in valid_points:
        print(f"    ({lat:.9f}, {lon:.9f}, \"{name}\", \"{dtm_path}\"),")
    print("]")
    print()
    
    return valid_points


if __name__ == "__main__":
    import sys
    
    # シードを設定（再現性のため）
    random.seed(42)
    np.random.seed(42)
    
    num_points = 5
    if len(sys.argv) > 1:
        num_points = int(sys.argv[1])
    
    project_root = Path(__file__).resolve().parents[2]
    dtm_path = str(project_root / "data" / "DTM_Naha.tif")
    generate_test_locations(dtm_path, num_points=num_points)

