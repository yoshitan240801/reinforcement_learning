# このフォルダのプログラムについて

このフォルダのプログラムは、在庫に対して最適な価格を付けるタスクを題材として、環境およびPPO(仮)モデルを勉強も兼ねてスクラッチで組んでみたものになります。<br>


# プログラムの目的

在庫を持つ環境において、最適な価格設定を学習する強化学習システム

- 環境: 在庫管理シミュレーション
- 学習手法: Actor-Critic (PPO風のクリップ機構を含む)
- 目標: 総報酬(売上)の最大化

---

## 環境設定

**パラメータ**
- 最大在庫: 100個
- 最大ステップ数: 31ステップ
- 価格選択肢: [100, 120, 140, 160, 180]

**状態表現**
```
状態 = [残り在庫の割合, 残りステップの割合]
```

2次元の連続値で環境の状態を表現

---

## 需要モデルの仕組み

```mermaid
graph LR
    A[価格選択] --> B[需要期待値計算]
    B --> C[ポアソン分布]
    C --> D[実際の需要]
    D --> E[販売数決定]
    E --> F[報酬計算]
    
    style A fill:#e1f5ff
    style F fill:#ffe1e1
```

**計算式**
```
需要期待値 = (最高価格 / 選択価格) × (在庫 × 0.05)
実際の需要 ~ Poisson(需要期待値)
販売数 = min(在庫, 需要)
報酬 = 価格 × 販売数
```

---

## ニューラルネットワーク構造

**Actor ネットワーク**
```
入力(状態: 2次元) 
  ↓
全結合層(64ユニット) + ReLU
  ↓
全結合層(5ユニット) + Softmax
  ↓
出力(行動確率分布: 5次元)
```

**Critic ネットワーク**
```
入力(状態: 2次元)
  ↓
全結合層(64ユニット) + ReLU
  ↓
全結合層(1ユニット)
  ↓
出力(状態価値: スカラー)
```

---

## 学習アルゴリズムの流れ

```mermaid
graph TD
    A[状態 St] --> B[Actor: 行動確率分布]
    B --> C[行動選択 At]
    C --> D[環境実行]
    D --> E[報酬 Rt, 次状態 St+1]
    E --> F[Critic: 状態価値推定]
    F --> G[TD誤差 δ 計算]
    G --> H[Actor損失計算<br/>PPO風クリップ]
    G --> I[Critic損失計算]
    H --> J[Actor更新]
    I --> K[Critic更新]
    
    style A fill:#e1f5ff
    style E fill:#ffe1e1
    style J fill:#e1ffe1
    style K fill:#e1ffe1
```

---

## 学習ループの詳細

**エピソードごとの処理**
1. 環境をリセット
2. 終了条件まで以下を繰り返し:
   - Actorで行動確率を計算(新方策 πθ)
   - 確率分布から行動を選択
   - 環境でステップ実行
   - 報酬と次状態を取得
   - Criticで状態価値を推定
   - 損失を計算して両モデルを更新

**学習パラメータ**
- エピソード数: 2500
- 割引率 γ: 0.98
- クリップ範囲 ε: 0.2
- 学習率: 0.001 (Adam)

---

## 損失関数の計算

**Critic損失(MSE)**
```
target = Rt + γ × V(St+1)
loss_critic = MSE(V(St), target)
```

**Actor損失(PPO風クリップ)**
```
ratio = πθ(At|St) / πold_θ(At|St)
ratio_clip = clip(ratio, 1-ε, 1+ε)
δ = Rt + γ×V(St+1) - V(St)
loss_actor = -min(ratio × δ, ratio_clip × δ)
```

---

## PPO風の確率比率クリップ

```mermaid
graph LR
    A[旧方策 πold] --> C[確率比率]
    B[新方策 πnew] --> C
    C --> D{クリップ処理}
    D --> E[1-ε ≤ ratio ≤ 1+ε]
    E --> F[損失計算]
    
    style D fill:#fff4e1
    style E fill:#e1ffe1
```

方策の急激な変化を防ぐ機構
※TD法ベースのため厳密なPPOではない

---

## 学習の実行フロー

```mermaid
sequenceDiagram
    participant E as Environment
    participant A as Actor
    participant C as Critic
    
    loop 2500 episodes
        E->>E: reset()
        loop until done
            E->>A: state
            A->>A: 行動確率計算
            A->>E: action
            E->>E: step()
            E->>C: next_state
            C->>C: 状態価値推定
            C->>C: 損失計算・更新
            A->>A: 損失計算・更新
        end
    end
```
