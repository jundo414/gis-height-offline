# compare_multiple_points.py
# -*- coding: utf-8 -*-
"""
複数地点での国土地理院APIと本実装の精度比較スクリプト
統計的な評価を含む
"""

import requests
import time
import numpy as np
from typing import Optional, List, Tuple
from dataclasses import dataclass
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main_japan import compute_heights as compute_heights_japan
from main_global import compute_heights as compute_heights_global

API_SLEEP_SEC = 1.2
API_MAX_RETRIES = 3
API_BACKOFF_SEC = 1.5


@dataclass
class ComparisonResult:
    """比較結果を格納するデータクラス"""
    lat: float
    lon: float
    location_name: str
    gsi_elevation: Optional[float]  # 国土地理院APIの標高（正高MSL）
    gsi_hsrc: Optional[str]  # モデル名
    gsi_geoid: Optional[float]  # 国土地理院APIのジオイド高N
    gsi_h_ellipsoid: Optional[float]  # 国土地理院APIの楕円体高h
    japan_h_msl: Optional[float]  # 日本版の正高（MSL）
    global_h_msl: Optional[float]  # グローバル版の正高（MSL）
    japan_h_ellipsoid: Optional[float]  # 日本版の楕円体高
    global_h_ellipsoid: Optional[float]  # グローバル版の楕円体高
    japan_n: Optional[float]  # 日本版のジオイド高
    global_n: Optional[float]  # グローバル版のジオイド高
    japan_diff_msl: Optional[float]  # 正高（MSL）の差異
    global_diff_msl: Optional[float]  # 正高（MSL）の差異
    japan_diff_geoid: Optional[float]  # ジオイド高Nの差異
    global_diff_geoid: Optional[float]  # ジオイド高Nの差異
    japan_diff_ellipsoid: Optional[float]  # 楕円体高hの差異
    global_diff_ellipsoid: Optional[float]  # 楕円体高hの差異


def _sleep_for_api() -> None:
    """API負荷軽減のため、呼び出し間隔を確保する。"""
    time.sleep(API_SLEEP_SEC)


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
    
    for attempt in range(1, API_MAX_RETRIES + 1):
        try:
            _sleep_for_api()
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
            if attempt == API_MAX_RETRIES:
                print(f"API呼び出しエラー ({lat}, {lon}): {e}")
                return None, None
            time.sleep(API_BACKOFF_SEC * attempt)
    return None, None


def get_geoid_from_gsi_api(lat: float, lon: float) -> Optional[float]:
    """
    国土地理院APIからジオイド高Nを取得

    APIエンドポイント:
    https://vldb.gsi.go.jp/sokuchi/surveycalc/geoid/calcgh/cgi/geoidcalc.pl
    """
    url = "https://vldb.gsi.go.jp/sokuchi/surveycalc/geoid/calcgh/cgi/geoidcalc.pl"
    params = {
        "latitude": lat,
        "longitude": lon,
        "outputType": "json",
    }

    for attempt in range(1, API_MAX_RETRIES + 1):
        try:
            _sleep_for_api()
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            # サーバ混雑時: {"ExportData":{"ErrMsg":"001:..."}}
            export = data.get("ExportData", {})
            err_msg = export.get("ErrMsg")
            if err_msg:
                if "001:" in err_msg and attempt < API_MAX_RETRIES:
                    time.sleep(API_BACKOFF_SEC * attempt)
                    continue
                print(f"警告: ジオイドAPIエラー ({lat}, {lon}): {err_msg}")
                return None

            output = data.get("OutputData", {})
            geoid_height = output.get("geoidHeight", None)
            if geoid_height is None:
                print(f"警告: ジオイドAPIレスポンスの形式が予期しないものです: {data}")
                return None
            return float(geoid_height)
        except Exception as e:
            if attempt == API_MAX_RETRIES:
                print(f"ジオイドAPI呼び出しエラー ({lat}, {lon}): {e}")
                return None
            time.sleep(API_BACKOFF_SEC * attempt)
    return None


def compare_single_point(
    lat: float,
    lon: float,
    location_name: str,
    dtm_path: str,
    isg_path: str,
    geoid_pgm_path: str,
) -> ComparisonResult:
    """1地点の比較を実行"""
    
    # 国土地理院APIから標高とモデル名を取得
    gsi_elevation, gsi_hsrc = get_elevation_from_gsi_api(lat, lon)
    gsi_geoid = get_geoid_from_gsi_api(lat, lon)
    gsi_h_ellipsoid = None
    if gsi_elevation is not None and gsi_geoid is not None:
        gsi_h_ellipsoid = gsi_elevation + gsi_geoid
    time.sleep(API_SLEEP_SEC)  # API負荷軽減のため待機
    
    # 本実装の結果
    result_japan = None
    result_global = None
    
    try:
        result_japan = compute_heights_japan(
            lat, lon,
            dtm_path=dtm_path,
            vertical_datum="DTM_IS_MSL",
            isg_path=isg_path,
        )
    except Exception as e:
        print(f"  日本版の計算エラー ({location_name}): {e}")
    
    try:
        result_global = compute_heights_global(
            lat, lon,
            dtm_path=dtm_path,
            vertical_datum="DTM_IS_MSL",
            geoid_pgm_path=geoid_pgm_path,
        )
    except Exception as e:
        print(f"  グローバル版の計算エラー ({location_name}): {e}")
    
    # 差異を計算（正高MSL）
    japan_diff_msl = None
    global_diff_msl = None
    japan_diff_geoid = None
    global_diff_geoid = None
    japan_diff_ellipsoid = None
    global_diff_ellipsoid = None
    
    if gsi_elevation is not None:
        if result_japan:
            japan_diff_msl = result_japan.H_ground_msl - gsi_elevation
        if result_global:
            global_diff_msl = result_global.H_ground_msl - gsi_elevation

    if gsi_geoid is not None:
        if result_japan:
            japan_diff_geoid = result_japan.N_geoid - gsi_geoid
        if result_global:
            global_diff_geoid = result_global.N_geoid - gsi_geoid

    if gsi_h_ellipsoid is not None:
        if result_japan:
            japan_diff_ellipsoid = result_japan.h_ground_ellipsoid - gsi_h_ellipsoid
        if result_global:
            global_diff_ellipsoid = result_global.h_ground_ellipsoid - gsi_h_ellipsoid
    
    return ComparisonResult(
        lat=lat,
        lon=lon,
        location_name=location_name,
        gsi_elevation=gsi_elevation,
        gsi_hsrc=gsi_hsrc,
        gsi_geoid=gsi_geoid,
        gsi_h_ellipsoid=gsi_h_ellipsoid,
        japan_h_msl=result_japan.H_ground_msl if result_japan else None,
        global_h_msl=result_global.H_ground_msl if result_global else None,
        japan_h_ellipsoid=result_japan.h_ground_ellipsoid if result_japan else None,
        global_h_ellipsoid=result_global.h_ground_ellipsoid if result_global else None,
        japan_n=result_japan.N_geoid if result_japan else None,
        global_n=result_global.N_geoid if result_global else None,
        japan_diff_msl=japan_diff_msl,
        global_diff_msl=global_diff_msl,
        japan_diff_geoid=japan_diff_geoid,
        global_diff_geoid=global_diff_geoid,
        japan_diff_ellipsoid=japan_diff_ellipsoid,
        global_diff_ellipsoid=global_diff_ellipsoid,
    )


def generate_test_points(dtm_path: str, num_points: int = 10) -> List[Tuple[float, float, str]]:
    """
    DTMの範囲内からテスト地点を生成
    簡易実装: 実際にはDTMの範囲を読み取って生成すべき
    """
    # 那覇と新宿の代表的な座標範囲
    if "Naha" in dtm_path:
        # 那覇周辺の座標範囲
        lat_min, lat_max = 26.15, 26.25
        lon_min, lon_max = 127.60, 127.90
        location_prefix = "那覇"
    elif "Shinjuku" in dtm_path:
        # 新宿周辺の座標範囲
        lat_min, lat_max = 35.65, 35.75
        lon_min, lon_max = 139.65, 139.75
        location_prefix = "新宿"
    else:
        raise ValueError(f"不明なDTMパス: {dtm_path}")
    
    # 格子状に地点を生成
    points = []
    lat_step = (lat_max - lat_min) / (num_points ** 0.5)
    lon_step = (lon_max - lon_min) / (num_points ** 0.5)
    
    count = 0
    for i in range(int(num_points ** 0.5)):
        for j in range(int(num_points ** 0.5)):
            if count >= num_points:
                break
            lat = lat_min + i * lat_step
            lon = lon_min + j * lon_step
            points.append((lat, lon, f"{location_prefix}_{count+1}"))
            count += 1
    
    return points[:num_points]


def print_statistics(results: List[ComparisonResult]):
    """統計情報を表示"""
    
    # 有効な結果のみを抽出
    valid_results = [r for r in results if r.gsi_elevation is not None]
    
    if len(valid_results) == 0:
        print("有効な比較結果がありません。")
        return
    
    # 地域別に分類
    naha_results = [r for r in valid_results if r.location_name.startswith("那覇")]
    shinjuku_results = [r for r in valid_results if r.location_name.startswith("新宿")]
    
    # 正高（MSL）の統計（全体）
    japan_diffs_msl = [abs(r.japan_diff_msl) for r in valid_results if r.japan_diff_msl is not None]
    global_diffs_msl = [abs(r.global_diff_msl) for r in valid_results if r.global_diff_msl is not None]
    
    # ジオイド高Nの統計（API基準）
    japan_diffs_geoid = [abs(r.japan_diff_geoid) for r in valid_results if r.japan_diff_geoid is not None]
    global_diffs_geoid = [abs(r.global_diff_geoid) for r in valid_results if r.global_diff_geoid is not None]

    # 楕円体高hの統計（API基準）
    japan_diffs_ellipsoid = [abs(r.japan_diff_ellipsoid) for r in valid_results if r.japan_diff_ellipsoid is not None]
    global_diffs_ellipsoid = [abs(r.global_diff_ellipsoid) for r in valid_results if r.global_diff_ellipsoid is not None]

    # ジオイドモデル同士の差分（参照用）
    n_model_diffs = [r.japan_n - r.global_n for r in valid_results if r.japan_n is not None and r.global_n is not None]
    
    print("\n" + "=" * 70)
    print("統計的評価（全体）")
    print("=" * 70)
    
    print(f"\n有効な比較地点数: {len(valid_results)} (那覇: {len(naha_results)}, 新宿: {len(shinjuku_results)})")
    
    # 正高（MSL）の精度統計
    print(f"\n【正高（MSL）の精度統計（全体）】")
    if japan_diffs_msl:
        print(f"  日本版（JPGEO2024）:")
        print(f"    平均絶対誤差 (MAE): {np.mean(japan_diffs_msl):.4f} m")
        print(f"    最大絶対誤差 (Max): {np.max(japan_diffs_msl):.4f} m")
        print(f"    最小絶対誤差 (Min): {np.min(japan_diffs_msl):.4f} m")
        print(f"    標準偏差 (Std): {np.std(japan_diffs_msl):.4f} m")
        print(f"    RMSE: {np.sqrt(np.mean([d**2 for d in japan_diffs_msl])):.4f} m")
    
    if global_diffs_msl:
        print(f"  グローバル版（EGM2008）:")
        print(f"    平均絶対誤差 (MAE): {np.mean(global_diffs_msl):.4f} m")
        print(f"    最大絶対誤差 (Max): {np.max(global_diffs_msl):.4f} m")
        print(f"    最小絶対誤差 (Min): {np.min(global_diffs_msl):.4f} m")
        print(f"    標準偏差 (Std): {np.std(global_diffs_msl):.4f} m")
        print(f"    RMSE: {np.sqrt(np.mean([d**2 for d in global_diffs_msl])):.4f} m")
    
    # ジオイド高Nの精度統計（API基準）
    print(f"\n【ジオイド高Nの精度統計（API基準）】")
    if japan_diffs_geoid:
        print(f"  日本版（JPGEO2024）:")
        print(f"    MAE: {np.mean(japan_diffs_geoid):.4f} m")
        print(f"    RMSE: {np.sqrt(np.mean([d**2 for d in japan_diffs_geoid])):.4f} m")
    if global_diffs_geoid:
        print(f"  グローバル版（EGM2008）:")
        print(f"    MAE: {np.mean(global_diffs_geoid):.4f} m")
        print(f"    RMSE: {np.sqrt(np.mean([d**2 for d in global_diffs_geoid])):.4f} m")

    # 楕円体高hの精度統計（API基準）
    print(f"\n【楕円体高hの精度統計（API基準）】")
    if japan_diffs_ellipsoid:
        print(f"  日本版（JPGEO2024）:")
        print(f"    MAE: {np.mean(japan_diffs_ellipsoid):.4f} m")
        print(f"    RMSE: {np.sqrt(np.mean([d**2 for d in japan_diffs_ellipsoid])):.4f} m")
    if global_diffs_ellipsoid:
        print(f"  グローバル版（EGM2008）:")
        print(f"    MAE: {np.mean(global_diffs_ellipsoid):.4f} m")
        print(f"    RMSE: {np.sqrt(np.mean([d**2 for d in global_diffs_ellipsoid])):.4f} m")

    # ジオイドモデル同士の比較統計（参照）
    if n_model_diffs:
        print(f"\n【ジオイドモデル差分（JPGEO2024 - EGM2008）】")
        print(f"  平均差異: {np.mean(n_model_diffs):.4f} m")
        print(f"  最大差異: {np.max([abs(d) for d in n_model_diffs]):.4f} m")
        print(f"  最小差異: {np.min([abs(d) for d in n_model_diffs]):.4f} m")
        print(f"  標準偏差: {np.std(n_model_diffs):.4f} m")
    
    # 地域別統計（正高MSL）
    if naha_results:
        naha_japan_diffs = [abs(r.japan_diff_msl) for r in naha_results if r.japan_diff_msl is not None]
        naha_global_diffs = [abs(r.global_diff_msl) for r in naha_results if r.global_diff_msl is not None]
        
        print(f"\n【那覇の精度統計（正高MSL）】")
        if naha_japan_diffs:
            print(f"  日本版 MAE: {np.mean(naha_japan_diffs):.4f} m, RMSE: {np.sqrt(np.mean([d**2 for d in naha_japan_diffs])):.4f} m")
        if naha_global_diffs:
            print(f"  グローバル版 MAE: {np.mean(naha_global_diffs):.4f} m, RMSE: {np.sqrt(np.mean([d**2 for d in naha_global_diffs])):.4f} m")
    
    if shinjuku_results:
        shinjuku_japan_diffs = [abs(r.japan_diff_msl) for r in shinjuku_results if r.japan_diff_msl is not None]
        shinjuku_global_diffs = [abs(r.global_diff_msl) for r in shinjuku_results if r.global_diff_msl is not None]
        
        print(f"\n【新宿の精度統計（正高MSL）】")
        if shinjuku_japan_diffs:
            print(f"  日本版 MAE: {np.mean(shinjuku_japan_diffs):.4f} m, RMSE: {np.sqrt(np.mean([d**2 for d in shinjuku_japan_diffs])):.4f} m")
        if shinjuku_global_diffs:
            print(f"  グローバル版 MAE: {np.mean(shinjuku_global_diffs):.4f} m, RMSE: {np.sqrt(np.mean([d**2 for d in shinjuku_global_diffs])):.4f} m")
    
    # 精度評価（正高MSL）
    print(f"\n【精度評価（正高MSL、全体）】")
    if japan_diffs_msl:
        mae_japan = np.mean(japan_diffs_msl)
        if mae_japan < 0.1:
            print(f"  日本版: 非常に高精度 (MAE: {mae_japan:.4f} m)")
        elif mae_japan < 0.5:
            print(f"  日本版: 高精度 (MAE: {mae_japan:.4f} m)")
        elif mae_japan < 1.0:
            print(f"  日本版: 実用精度 (MAE: {mae_japan:.4f} m)")
        else:
            print(f"  日本版: 要検討 (MAE: {mae_japan:.4f} m)")
    
    if global_diffs_msl:
        mae_global = np.mean(global_diffs_msl)
        if mae_global < 0.5:
            print(f"  グローバル版: 高精度 (MAE: {mae_global:.4f} m)")
        elif mae_global < 1.0:
            print(f"  グローバル版: 実用精度 (MAE: {mae_global:.4f} m)")
        else:
            print(f"  グローバル版: 要検討 (MAE: {mae_global:.4f} m)")
    
    # 注意事項
    print(f"\n【注意事項】")
    print(f"  - 正高（MSL）の比較では、同じDTMデータを使用しているため、")
    print(f"    日本版とグローバル版の差異は同じになります。")
    print(f"  - ジオイドモデルの精度の違いは、ジオイド高Nの比較で確認できます。")
    print(f"\n【楕円体高の差異について】")
    print(f"  同じDTMを使用しているため、正高（MSL）Hは同じ値になります。")
    print(f"  しかし、楕円体高hは以下の式で計算されます：")
    print(f"    h = H + N")
    print(f"  ここで、N（ジオイド高）は使用するジオイドモデルによって異なります：")
    print(f"    - 日本版: h_japan = H + N_japan (JPGEO2024)")
    print(f"    - グローバル版: h_global = H + N_global (EGM2008)")
    print(f"  したがって、楕円体高の差異は：")
    print(f"    h_japan - h_global = (H + N_japan) - (H + N_global) = N_japan - N_global")
    print(f"  つまり、楕円体高の差異は、ジオイド高の差異と完全に一致します。")
    print(f"  これは、異なるジオイドモデルを使用しているため、正常な動作です。")


def compare_multiple_locations():
    """複数地点での比較を実行"""
    
    print("=" * 70)
    print("複数地点での国土地理院APIとの精度比較")
    print("=" * 70)
    
    isg_path = str(PROJECT_ROOT / "data" / "JPGEO2024.isg")
    geoid_pgm_path = str(PROJECT_ROOT / "data" / "egm2008-1.pgm")
    
    # テスト地点の定義（那覇と新宿）
    test_locations = [
        # 那覇（有効なDTMデータが存在する固定地点）
        (26.21439,127.68054, "那覇_1", str(PROJECT_ROOT / "data" / "DTM_Naha.tif")),
        (26.20648,127.71700, "那覇_2", str(PROJECT_ROOT / "data" / "DTM_Naha.tif")),
        (26.19362,127.66460, "那覇_3", str(PROJECT_ROOT / "data" / "DTM_Naha.tif")),
        (26.19710,127.69421, "那覇_4", str(PROJECT_ROOT / "data" / "DTM_Naha.tif")),
        (26.23959,127.68470, "那覇_5", str(PROJECT_ROOT / "data" / "DTM_Naha.tif")),
        
        # 新宿
        (35.68471676509442, 139.7135696509406, "新宿_1", str(PROJECT_ROOT / "data" / "DTM_Shinjuku.tif")),
        (35.720266907415024, 139.68344661635518, "新宿_2", str(PROJECT_ROOT / "data" / "DTM_Shinjuku.tif")),
        (35.692665481383365, 139.68748204076343, "新宿_3", str(PROJECT_ROOT / "data" / "DTM_Shinjuku.tif")),
        (35.70527851120032, 139.71460595476708, "新宿_4", str(PROJECT_ROOT / "data" / "DTM_Shinjuku.tif")),
        (35.70575955179639, 139.73726841628337, "新宿_5", str(PROJECT_ROOT / "data" / "DTM_Shinjuku.tif")),
    ]
    
    results = []
    
    print(f"\n合計 {len(test_locations)} 地点での比較を開始します...\n")
    
    for i, (lat, lon, name, dtm_path) in enumerate(test_locations, 1):
        print(f"[{i}/{len(test_locations)}] {name} (緯度: {lat:.6f}, 経度: {lon:.6f})")
        
        result = compare_single_point(
            lat, lon, name, dtm_path, isg_path, geoid_pgm_path
        )
        results.append(result)
        
        # 結果を表示
        if result.gsi_elevation is not None:
            hsrc_str = f" ({result.gsi_hsrc})" if result.gsi_hsrc else ""
            print(f"  国土地理院API: {result.gsi_elevation:.3f} m{hsrc_str}")
            if result.gsi_geoid is not None and result.gsi_h_ellipsoid is not None:
                print(f"    APIジオイド高N: {result.gsi_geoid:.4f} m, API楕円体高: {result.gsi_h_ellipsoid:.3f} m")
            if result.japan_h_msl is not None:
                print(f"  日本版（MSL）: {result.japan_h_msl:.3f} m (差異: {result.japan_diff_msl:+.3f} m)")
                if result.japan_h_ellipsoid is not None:
                    geoid_diff_str = f", N差異: {result.japan_diff_geoid:+.4f} m" if result.japan_diff_geoid is not None else ""
                    ellip_diff_str = f", h差異: {result.japan_diff_ellipsoid:+.3f} m" if result.japan_diff_ellipsoid is not None else ""
                    print(f"    楕円体高: {result.japan_h_ellipsoid:.3f} m, ジオイド高N: {result.japan_n:.4f} m{geoid_diff_str}{ellip_diff_str}")
            if result.global_h_msl is not None:
                print(f"  グローバル版（MSL）: {result.global_h_msl:.3f} m (差異: {result.global_diff_msl:+.3f} m)")
                if result.global_h_ellipsoid is not None:
                    geoid_diff_str = f", N差異: {result.global_diff_geoid:+.4f} m" if result.global_diff_geoid is not None else ""
                    ellip_diff_str = f", h差異: {result.global_diff_ellipsoid:+.3f} m" if result.global_diff_ellipsoid is not None else ""
                    print(f"    楕円体高: {result.global_h_ellipsoid:.3f} m, ジオイド高N: {result.global_n:.4f} m{geoid_diff_str}{ellip_diff_str}")
        else:
            print(f"  警告: 国土地理院APIから標高を取得できませんでした")
        print()
        time.sleep(1)
    
    # 詳細結果テーブル（正高MSL）
    print("\n" + "=" * 70)
    print("詳細比較結果（正高MSL）")
    print("=" * 70)
    print(f"{'地点':<12} {'GSI API':<10} {'モデル':<20} {'日本版':<10} {'差異':<10} {'グローバル版':<12} {'差異':<10}")
    print("-" * 70)
    
    for r in results:
        if r.gsi_elevation is not None:
            gsi_str = f"{r.gsi_elevation:.3f}"
            hsrc_str = r.gsi_hsrc if r.gsi_hsrc else "N/A"
            japan_str = f"{r.japan_h_msl:.3f}" if r.japan_h_msl is not None else "N/A"
            japan_diff_str = f"{r.japan_diff_msl:+.3f}" if r.japan_diff_msl is not None else "N/A"
            global_str = f"{r.global_h_msl:.3f}" if r.global_h_msl is not None else "N/A"
            global_diff_str = f"{r.global_diff_msl:+.3f}" if r.global_diff_msl is not None else "N/A"
            
            print(f"{r.location_name:<12} {gsi_str:<10} {hsrc_str:<20} {japan_str:<10} {japan_diff_str:<10} {global_str:<12} {global_diff_str:<10}")
    
    # ジオイド高の比較テーブル（API基準）
    print("\n" + "=" * 70)
    print("ジオイド高の比較（API基準）")
    print("=" * 70)
    print(f"{'地点':<12} {'API N':<10} {'日本版N':<12} {'差異':<10} {'グローバル版N':<15} {'差異':<10}")
    print("-" * 70)
    
    for r in results:
        if r.japan_n is not None and r.global_n is not None:
            api_n = f"{r.gsi_geoid:.4f}" if r.gsi_geoid is not None else "N/A"
            japan_n_diff = f"{r.japan_diff_geoid:+.4f}" if r.japan_diff_geoid is not None else "N/A"
            global_n_diff = f"{r.global_diff_geoid:+.4f}" if r.global_diff_geoid is not None else "N/A"
            print(f"{r.location_name:<12} {api_n:<10} {r.japan_n:>11.4f} {japan_n_diff:>10} {r.global_n:>14.4f} {global_n_diff:>10}")
    
    # 統計情報
    print_statistics(results)
    
    return results


if __name__ == "__main__":
    results = compare_multiple_locations()

