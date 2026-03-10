# GIS Height Offline

指定した緯度経度に対して、オフラインで高さ関連値を計算するプロジェクトです。  
主に以下の値を求めます。

- 地面の正高（MSL）`H`
- ジオイド高 `N`
- 地面の楕円体高 `h`（`h = H + N`）

用途に応じて２系統を使い分けます。

- 日本向け: `JPGEO2024.isg` を使う `main_japan.py`
- グローバル向け: `egm2008-1.pgm` を使う `main_global.py`

## プロジェクト構成

大きく **本体** と **検証** の２種類で構成されています。

```text
gis-height-offline/
├─ main_global.py      # 本体（グローバル計算）
├─ main_japan.py       # 本体（日本向け計算）
└─ validation/         # 検証・比較・診断スクリプト
```

- 本体コード（計算ロジック）: `main_global.py`, `main_japan.py`
- 検証コード（比較・診断）: `validation/*`

## validation 配下の構成

- `compare_gsi_api.py`: 1地点で国土地理院APIと本実装を比較
- `compare_multiple_points.py`: 複数地点でAPI比較し、統計を出力
- `dtm/check_dtm_bounds.py`: DTMの範囲・品質（nodata割合など）を確認
- `dtm/generate_valid_points.py`: DTM内で有効標高点を抽出
- `dtm/test_dtm_nodata.py`: nodata周辺のピクセル値を確認
- `pgm/insight_pgm.py`: PGMヘッダと生データ参照の基本確認
- `pgm/probe_pgm_nearest.py`: PGMの最近傍サンプル確認
- `pgm/probe_pgm_bilinear.py`: PGMバイリニア補間の手計算確認
- `pgm/probe_isg_bilinear.py`: ISG読込とバイリニア補間確認

## 本体コードの役割

- `main_japan.py`
  - 日本域向け
  - ISG（`JPGEO2024.isg`）から `N` を計算
  - `compute_heights(...)` が主要API
- `main_global.py`
  - グローバル向け
  - PGM（`egm2008-1.pgm`）から `N` を計算
  - `compute_heights(...)` が主要API

## 処理フロー（本体）

1. 入力: `lat`, `lon`
2. DTM（GeoTIFF）から地面高を取得（2x2 + バイリニア補間）
3. ジオイドモデルから `N` を取得（ISGまたはPGM）
4. `vertical_datum` に応じて `H` と `h` を変換
5. `HeightsResult` を出力

## セットアップ手順

1. Python仮想環境を作成
2. 仮想環境を有効化
3. 依存関係をインストール
   - `pip install -r requirements.txt`
4. データファイルを確認
   - `data/DTM_Naha.tif`
   - `data/DTM_Shinjuku.tif`
   - `data/JPGEO2024.isg`
   - `data/egm2008-1.pgm`

## 実行方法

### A. 本体コード単体で実行

API比較なしで、指定地点の高さ計算だけを確認します。

実行:
- `python main_japan.py`
- `python main_global.py`

設定箇所（各ファイル末尾）:
- `lat`, `lon`: 計算対象座標
- `dtm_path`: 使用DTM
- `isg_path` または `geoid_pgm_path`: 使用ジオイドモデル
- `vertical_datum`: `DTM_IS_MSL` / `DTM_IS_ELLIPSOID`

### B. 検証スクリプトを実行

API比較、統計評価、データ品質診断、補間デバッグ用のスクリプトです。

実行例:
- `python validation/compare_gsi_api.py`
- `python validation/compare_multiple_points.py`
- `python validation/dtm/check_dtm_bounds.py`
- `python validation/dtm/generate_valid_points.py`
- `python validation/dtm/test_dtm_nodata.py`
- `python validation/pgm/insight_pgm.py`
- `python validation/pgm/probe_pgm_nearest.py`
- `python validation/pgm/probe_pgm_bilinear.py`
- `python validation/pgm/probe_isg_bilinear.py`

設定箇所:
- 比較系: 対象座標、比較地点リスト
- DTM診断系: `dtm_path`、サンプリング条件
- PGM/ISG診断系: `lat`, `lon`、対象モデルファイル
