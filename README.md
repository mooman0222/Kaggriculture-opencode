# Kaggriculture ベースシステム

[Kaggle コンペ: Kaggriculture](https://www.kaggle.com/competitions/kaggriculture) 用のベースエージェント。
参考ノートブック [Kaggriculture: Getting Started](https://www.kaggle.com/code/bovard/kaggriculture-getting-started)
の melon_maxxer を土台に、ローカル A/B テストで検証した改良を加えたもの。

## ゲーム概要

- 2人対戦の農業シミュレーション。30日 × 24ターン = 720ターン
- シーズン終了時点の所持金が多い方が勝利
- 各プレイヤーは 10×10 の農場 (NW 4象限のみ最初から解放) で作物を育て市場で売る
- 市場価格は需給で変動: 売ると価格が下がり、町の店が消費すると上がる
- 作物: WHEAT / CARROT / TOMATO / STRAWBERRY / MELON、動物: GOOSE / COW / SHEEP
- 詳細ルールは `kaggle-environments` パッケージ内の `kaggriculture/README.md` を参照

## 構成

```
main.py          提出用エージェント (main.py の agent 関数をそのまま提出)
agents/base.py   ノートブックの melon_maxxer 原版 (ベースライン比較用)
tests/run_match.py    1試合の実行と状態表示
tests/evaluate.py     複数試合の勝率・平均所持金統計
requirements.txt
```

## セットアップ

```bash
pip install -r requirements.txt
```

## テスト

```bash
# 1試合 (シード固定・対戦相手指定)
python tests/run_match.py --seed 1 --opponent random
python tests/run_match.py --seed 1 --opponent base
python tests/run_match.py --seed 1 --opponent starter --replay replay.json

# 統計評価 (勝率・平均所持金)
python tests/evaluate.py --games 20 --opponent random
python tests/evaluate.py --games 20 --opponent starter
python tests/evaluate.py --games 20 --opponent base
```

対戦相手は `random` (ランダム), `starter` (組み込み carrot ループ), `base` (melon_maxxer 原版)。

## 提出

Kaggle CLI をセットアップ後 (`pip install kaggle`, API トークン設定):

```bash
kaggle competitions submit kaggriculture -f main.py -m "melon v1"
```

またはノートブック上で main.py をアップロードして提出。