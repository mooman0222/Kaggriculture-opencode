# Kaggriculture ベースシステム

[Kaggle コンペ: Kaggriculture](https://www.kaggle.com/competitions/kaggriculture) 用のエージェント。
**現行の提出は `agents/frontier_k4.py` (E020b)** — 公開ノートブック
prvsiyan/kaggriculture-frontier-the-soil-remembers-rain (ショップ抽選適応型アンサンブル、
day3 の初ショップで農場計画を分岐) の内蔵 V174 に、K=4 デット式 front-run
(クローン戦エッジ) を移植したもの。`main.py` は前世代 E019b (HarvestForge-X + K4、
[salemali7/kaggriculture-2900](https://www.kaggle.com/code/salemali7/kaggriculture-2900) ベース)。
経緯・検証結果は `AGENTS.md` と `.opencode/knowledge/` を参照。

## ゲーム概要

- 2人対戦の農業シミュレーション。30日 × 24ターン = 720ターン
- シーズン終了時点の所持金が多い方が勝利
- 各プレイヤーは 10×10 の農場 (NW 4象限のみ最初から解放) で作物を育て市場で売る
- 市場価格は需給で変動: 売ると価格が下がり、町の店が消費すると上がる
- 作物: WHEAT / CARROT / TOMATO / STRAWBERRY / MELON、動物: GOOSE / COW / SHEEP
- 詳細ルールは `kaggle-environments` パッケージ内の `kaggriculture/README.md` を参照

## 構成

```
agents/frontier_k4.py   現行提出 (E020b: frontier + K4 デット式 front-run 移植)
agents/frontier_prvsiyan.py  frontier 原本 (公開NBペイロード展開・無改変、メタヘッジ提出)
main.py                 前世代 E019b (HarvestForge-X + K4。A/B の強相手用)
agents/base.py          melon_maxxer 原版 (ベースライン比較用)
agents/e018_route.py    旧主力 E018 (ルート再生型、A/B の中堅相手用)
agents/adaptive_route.py  E018 系ツールのエージェント工場 (make_adaptive_agent)
tests/run_match.py      1試合の実行と状態表示
tests/evaluate.py       複数試合の勝率・平均所持金統計
tests/ab.py             同一シード・同一相手での A/B 比較
tests/route_gen.py ほか  E018 系のルート抽出・評価ツール (遺産)
.opencode/knowledge/    実験ログ・知見・メタ情報のナレッジベース
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
kaggle competitions submit kaggriculture -f main.py -m "E019b ..."
```

またはノートブック上で main.py をアップロードして提出。