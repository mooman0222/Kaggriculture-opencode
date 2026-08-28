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
- 詳細ルールは `.venv/lib/python3.14/site-packages/kaggle_environments/envs/kaggriculture/README.md` を参照

## 構成

```
main.py          提出用エージェント (main.py の agent 関数をそのまま提出)
agents/base.py   ノートブックの melon_maxxer 原版 (ベースライン比較用)
tests/run_match.py    1試合の実行と状態表示
tests/evaluate.py     複数試合の勝率・平均所持金統計
requirements.txt
```

## セットアップ (WSL / Linux)

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Python 3.14 では pygame のビルドに失敗するため、代わりに pygame-ce を先に入れる:

```bash
.venv/bin/pip install pygame-ce
.venv/bin/pip install --no-deps "kaggle-environments>=1.32.2"
.venv/bin/pip install jsonschema requests websocket-client numpy termcolor
```

## テスト

```bash
# 1試合 (シード固定・対戦相手指定)
.venv/bin/python tests/run_match.py --seed 1 --opponent random
.venv/bin/python tests/run_match.py --seed 1 --opponent base
.venv/bin/python tests/run_match.py --seed 1 --opponent starter --replay replay.json

# 統計評価 (勝率・平均所持金)
.venv/bin/python tests/evaluate.py --games 20 --opponent random
.venv/bin/python tests/evaluate.py --games 20 --opponent starter
.venv/bin/python tests/evaluate.py --games 20 --opponent base
```

対戦相手は `random` (ランダム), `starter` (組み込み carrot ループ), `base` (melon_maxxer 原版)。

## エージェントの戦略 (main.py)

- メロン専業: 種購入 → 植え付け → 毎日水やり → 収穫
- **即収穫**: first_yield_day (10日目) で収量が十分なら即収穫し、3サイクル目を狙う
- **分割売却**: 1ターン最大10個。閾値 (200) 以上の価格のときだけ売る
- **供給自己規制**: 価格が閾値を下回ったら植え付けを停止 (市場の消化量を超えない)
- **溢れ対策**: 納屋が90個を超えそうなら安くても売却、day>=28 は強制換金
- **ファームハンド**: 資金に余裕があれば朝に雇用 (1日最大2人、水やり・収穫を分担)

### ローカル A/B テストの知見

| 設定 | 平均所持金 (vs random, 5試合) |
|---|---|
| デフォルト (hands ON / land OFF) | $26,257 |
| 土地購入 ON | $16,307 |
| ハンド OFF | $20,751 |
| 両方 OFF | $23,680 |

- **土地購入はメロン専業では逆効果**: メロン市場は約70個の供給で価格が閾値を割り、
  タイル数を増やしても売上は増えない (購入費 $7k が丸損になる)
- **ファームハンドは有効**: 水やり/収穫のカバー率が上がり約 +$3k
- **肥料はメロンには無意味**: 水やりを毎日していれば max_yield=6 に到達するため

## 提出

Kaggle CLI をセットアップ後 (`pip install kaggle`, API トークン設定):

```bash
kaggle competitions submit kaggriculture -f main.py -m "melon v1"
```

またはノートブック上で main.py をアップロードして提出。

## 次の改善アイデア

1. **多品種化**: メロン市場は頭打ち (~70個) なので、小麦 (価格が下がりにくい) や
   店が要求する作物 (STRAWBERRY/MILK/EGG 等) への分散が有効
2. **店の需要予測**: `town.unlocked_shops` を読んで需要が伸びる商品を生産
3. **土地購入の再検討**: 多品種戦略ならタイル不足がボトルネックになるため有効化
4. **肥料**: トマト・イチゴなど継続作物 (2倍収穫) を育てるなら ON
5. **相手エージェントの観察**: `farms` から相手の戦略を推定して市場投入量を調整