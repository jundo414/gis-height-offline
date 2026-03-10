# compare_gsi_api.py
# -*- coding: utf-8 -*-
"""
国土地理院APIと本実装の精度比較スクリプト
"""

import requests
import sys
from pathlib import Path
from typing import Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main_japan import compute_heights as compute_heights_japan
from main_global import compute_heights as compute_heights_global


def get_elevation_from_gsi_api(lat: float, lon: float) -> Tuple[Optional[float], Optional[str]]:
    """
    国土地理院標高APIから標高とモデル名を取得
    
    APIエンドポイント: https://cyberjapandata2.gsi.go.jp/general/dem/scripts/getelevation.php
    
    Returns:
        (標高, モデル名) のタプル
    """
    url = "https://cyberjapandata2.gsi.go.jp/general/dem/scripts/getelevation.php"
    params = {
        "lon": lon,
        "lat": lat,
        "outtype": "JSON"
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # 標高を取得
        elevation = None
        if "elevation" in data:
            elevation = float(data["elevation"])
        elif "height" in data:
            elevation = float(data["height"])
        elif isinstance(data, (int, float)):
            elevation = float(data)
        else:
            print(f"警告: APIレスポンスの形式が予期しないものです: {data}")
            return None, None
        
        # モデル名（hsrc）を取得
        hsrc = data.get("hsrc", None)
        
        return elevation, hsrc
    except Exception as e:
        print(f"API呼び出しエラー: {e}")
        return None, None


def compare_with_gsi_api(lat: float, lon: float):
    """
    本実装と国土地理院APIの結果を比較
    """
    print("=" * 60)
    print("国土地理院APIとの精度比較")
    print("=" * 60)
    print(f"テスト地点: 緯度 {lat}, 経度 {lon}\n")
    
    # 国土地理院APIから標高とモデル名を取得
    print("1. 国土地理院標高APIから標高を取得中...")
    gsi_elevation, gsi_hsrc = get_elevation_from_gsi_api(lat, lon)
    
    if gsi_elevation is None:
        print("警告: 国土地理院APIから標高を取得できませんでした。")
        print("      APIが利用できない、または座標が範囲外の可能性があります。")
        return
    
    hsrc_str = f" ({gsi_hsrc})" if gsi_hsrc else ""
    print(f"   国土地理院API 標高: {gsi_elevation:.3f} m{hsrc_str}\n")
    
    # 本実装の結果
    print("2. 本実装による計算...")
    
    dtm_path = str(PROJECT_ROOT / "data" / "DTM_Naha.tif")
    isg_path = str(PROJECT_ROOT / "data" / "JPGEO2024.isg")
    geoid_pgm_path = str(PROJECT_ROOT / "data" / "egm2008-1.pgm")
    
    # 日本版（JPGEO2024）
    try:
        result_japan = compute_heights_japan(
            lat, lon,
            dtm_path=dtm_path,
            vertical_datum="DTM_IS_MSL",
            isg_path=isg_path,
        )
        print(f"   日本版（JPGEO2024）正高: {result_japan.H_ground_msl:.3f} m")
    except Exception as e:
        print(f"   日本版の計算エラー: {e}")
        result_japan = None
    
    # グローバル版（EGM2008）
    try:
        result_global = compute_heights_global(
            lat, lon,
            dtm_path=dtm_path,
            vertical_datum="DTM_IS_MSL",
            geoid_pgm_path=geoid_pgm_path,
        )
        print(f"   グローバル版（EGM2008）正高: {result_global.H_ground_msl:.3f} m")
    except Exception as e:
        print(f"   グローバル版の計算エラー: {e}")
        result_global = None
    
    # 比較結果
    print("\n3. 比較結果:")
    print("-" * 80)
    hsrc_str = f" ({gsi_hsrc})" if gsi_hsrc else ""
    print(f"{'方法':<25} {'標高 [m]':<15} {'モデル':<20} {'差異 [m]':<15}")
    print("-" * 80)
    print(f"{'国土地理院API':<25} {gsi_elevation:>14.3f} {hsrc_str:<20} {'基準':<15}")
    
    if result_japan:
        diff_japan = result_japan.H_ground_msl - gsi_elevation
        print(f"{'本実装（JPGEO2024）':<25} {result_japan.H_ground_msl:>14.3f} {'JPGEO2024':<20} {diff_japan:>14.3f}")
    
    if result_global:
        diff_global = result_global.H_ground_msl - gsi_elevation
        print(f"{'本実装（EGM2008）':<25} {result_global.H_ground_msl:>14.3f} {'EGM2008':<20} {diff_global:>14.3f}")
    
    print("-" * 80)
    
    # 精度評価
    print("\n4. 精度評価:")
    if result_japan:
        abs_diff_japan = abs(diff_japan)
        if abs_diff_japan < 0.1:
            print(f"   日本版（JPGEO2024）: 非常に高精度（差異: {abs_diff_japan:.3f} m）")
        elif abs_diff_japan < 0.5:
            print(f"   日本版（JPGEO2024）: 高精度（差異: {abs_diff_japan:.3f} m）")
        elif abs_diff_japan < 1.0:
            print(f"   日本版（JPGEO2024）: 実用精度（差異: {abs_diff_japan:.3f} m）")
        else:
            print(f"   日本版（JPGEO2024）: 要検討（差異: {abs_diff_japan:.3f} m）")
    
    if result_global:
        abs_diff_global = abs(diff_global)
        if abs_diff_global < 0.5:
            print(f"   グローバル版（EGM2008）: 高精度（差異: {abs_diff_global:.3f} m）")
        elif abs_diff_global < 1.0:
            print(f"   グローバル版（EGM2008）: 実用精度（差異: {abs_diff_global:.3f} m）")
        else:
            print(f"   グローバル版（EGM2008）: 要検討（差異: {abs_diff_global:.3f} m）")
    
    print("\n注意: 差異の原因として以下が考えられます:")
    print("  - DTMデータソースの違い")
    print("  - 補間アルゴリズムの違い")
    print("  - 座標変換の精度")
    print("  - データの更新時期の違い")


if __name__ == "__main__":
    # テスト地点: 沖縄県那覇市
    lat = 26.186898142655217
    lon = 127.8151512771811
    
    compare_with_gsi_api(lat, lon)

