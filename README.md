# 心のアーキテクチャ — Cognitive Architecture

Pythonで実装した、心の忠実な再現。LLM・世界モデル・自己モデル・感情・エンボディドAI・グローバルワークスペースを統合した認知アーキテクチャです。

## アーキテクチャ概要

```
┌─────────────────────────────────────────────────────────┐
│           Global Workspace — 意識の劇場 (Layer 6)        │
│                                                         │
│  ┌─────┐  ┌────────┐  ┌──────┐  ┌──────┐  ┌─────────┐  │
│  │ LLM │  │ World  │  │ Self │  │感情  │  │Embodied │  │
│  │ (1) │  │ Model  │  │Model │  │ (4)  │  │  AI     │  │
│  │     │  │  (2)   │  │ (3)  │  │      │  │  (5)    │  │
│  └─────┘  └────────┘  └──────┘  └──────┘  └─────────┘  │
└─────────────────────────────────────────────────────────┘
```

### 6つの層

| Layer | モジュール | 役割 |
|-------|-----------|------|
| 1 | **LLM** (Language Module) | 言語パターンの処理・理解・生成・内言 |
| 2 | **World Model** | 世界の内部シミュレーション・予測・異常検出 |
| 3 | **Self Model** | 「世界の中の自分」の定義・目標管理・自伝的記憶・メタ認知 |
| 4 | **Emotion** | 感情による優先順位と価値の付与（評価理論ベース） |
| 5 | **Embodied AI** | 身体性・運動制御・固有受容覚・行為のフィードバック |
| 6 | **Global Workspace** | Baarsのグローバルワークスペース理論に基づく意識の統合 |

## 認知サイクル

`Mind.step()` を呼ぶたびに、一つの認知サイクルが実行されます：

1. 各モジュールが候補となる **Thought**（思考）を生成
2. 感情モジュールが目標の優先順位を調整
3. グローバルワークスペースで思考が **競合** し、最も顕著な思考が選出
4. 勝者が全モジュールに **ブロードキャスト**（意識に上る）
5. 身体モジュールが目標に基づいて **行動** を実行
6. 行動の結果が世界モデルにフィードバック

## インストール

```bash
# 外部依存なし — Python 3.10+ のみ必要
git clone https://github.com/jundo414/cognitive-architecture.git
cd cognitive-architecture
pip install -e ".[dev]"
```

## 使い方

### 基本的な使い方

```python
from cognitive_architecture import Mind

# 心を作る
mind = Mind(agent_name="Kokoro")

# 目標を設定
mind.add_goal("explore the environment", priority=0.7)

# 言語入力を受け取る
mind.receive_input("There is a forest ahead with tall trees.")

# 認知サイクルを実行
for _ in range(10):
    report = mind.step()
    print(f"意識: {report['conscious_thought']}")
    print(f"感情: {report['emotional_state']}")
    print()
```

### デモの実行

```bash
python examples/demo.py
```

### テスト

```bash
pytest tests/
```

## プロジェクト構成

```
cognitive_architecture/
├── core/
│   ├── types.py          # 共通型定義（Percept, Belief, Emotion, etc.）
│   └── event_bus.py      # モジュール間通信のイベントバス
├── llm/
│   └── language_module.py  # Layer 1: 言語処理
├── world_model/
│   └── world_model.py     # Layer 2: 世界モデル
├── self_model/
│   └── self_model.py      # Layer 3: 自己モデル
├── emotion/
│   └── emotion_module.py  # Layer 4: 感情モジュール
├── embodied/
│   └── embodied_agent.py  # Layer 5: 身体性AI
├── global_workspace/
│   └── workspace.py       # Layer 6: グローバルワークスペース
└── mind.py                # 統合Mind クラス
```

## 理論的背景

- **Global Workspace Theory** (Baars, 1988): 意識は情報を全モジュールにブロードキャストする「劇場」
- **Predictive Processing** (Clark, 2013): 世界モデルは予測と予測誤差で駆動
- **Appraisal Theory** (Scherer, 2001): 感情は事象の評価から生まれる
- **PAD Model** (Mehrabian, 1996): 感情を快・覚醒・支配の3次元で表現
- **Embodied Cognition** (Varela et al., 1991): 認知は身体と環境に埋め込まれている

## ライセンス

MIT License
