# Kaggriculture ベースシステム

[Kaggle コンペ: Kaggriculture](https://www.kaggle.com/competitions/kaggriculture) 用のエージェント。
**現行の提出は `agents/kaito_v56_orak16.py` (E029)** — 公開ノートブック
kaitofukami/137-161-replay-9-12-final-v56-shop-hybrid (v56) をコアに、
マルチホライズン・セルフオラクル front-run (K=16)、終局スイープ、ヤーンルート遅延切替
(2-3軒目が YARN_STORE なら step144/216 で切替) を載せたもの。
旧世代: `agents/frontier_*` (E020)、`main.py` (E019b HarvestForge-X + K4)。
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
agents/kaito_v56_orak16.py  現行 (E034: v56 + 小麦先買い30 + K16 オラクル + テープオラクル×2 + step216 C9S8 移植 + ヤーン切替 + sweep)
agents/kaito_v56_orak12_yarn.py  E029b (同上の K=12 版、2軸提出)
agents/kaito_v56_*.py   v56 系の各段階 (素/sweep/oracle/orak12)
agents/frontier_k4.py   旧提出 (E020b: frontier + K4 デット式 front-run 移植)
agents/frontier_prvsiyan.py  frontier 原本 (公開NBペイロード展開・無改変、メタヘッジ提出)
main.py                 前世代 E019b (HarvestForge-X + K4。A/B の強相手用)
agents/base.py          melon_maxxer 原版 (ベースライン比較用)
agents/e018_route.py    旧主力 E018 (ルート再生型、A/B の中堅相手用)
agents/adaptive_route.py  E018 系ツールのエージェント工場 (make_adaptive_agent)
tests/fetch_battles.py  実戦リプレイを相手系統 (24手署名/動物構成) 別に集計
tests/h2h.py            2エージェントの同シード両席直接対決 (ペア margin + ショップ列)
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