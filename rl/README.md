# rl/ — 学習エージェント (Transformer 方策、模倣学習 → PPO)

## 環境の再構築 (別マシン / .so が無いとき)

```
uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt
uv pip install --python .venv/bin/python -e third_party/kaggriculture-cppsim   # kagsim (C++ エンジン + Game.encode) をその場でビルド、約 1 分
.venv/bin/python -c "import kagsim; print(kagsim.Game(1).encode(0)['tiles'].shape)"  # (2, 10, 10, 18) なら OK
```
`encode.hpp` 等を変えたら `cd third_party/kaggriculture-cppsim && ../../.venv/bin/python setup.py build_ext --inplace`。
重み・shard は git 管理外: 以下の手順で再生成するか、Kaggle dataset `mmn0222/kaggriculture-rl-bc6` (code + bc6data 2,492 shard + bc5_ep3/bc6_ep0) を落とす。

パイプライン (すべて `.venv/bin/python` で実行、作業データは `tmp/rl/`、消えても再生成可):
1. データ: `rl/fetch_episodes.py --sub <submission id> --team <name> --out tmp/rl/<tag>` (Kaggle API から取得して即 shard 化、JSON は捨てる)、
   `rl/extract_parquet.py --parquet tmp/data/replays_2026-09.parquet --top 40 --out tmp/rl/pq` (公開データセット georgymamarin/kaggriculture-episodes の月次 parquet から上位チームの席)。
   shard = `<episode>.npz` (tiles/units/items/glob 特徴、dest/dop/dqty = 目的タイルとそこでの作業のラベル、dmask、mkt)。
2. 模倣学習: `rl/train_bc2.py --data 'tmp/rl/**/*.npz' --out tmp/rl/bcN.pt --epochs 4 --bs 512 --lr 5e-4`。
   **途中終了しても `--resume` を付けて同じコマンドを打てば `<out>.state` から続く** (200 batch ごと・epoch ごとに保存)。epoch ごとの重みは `<out>_epK.pt`。
3. 評価: `rl/play2.py tmp/rl/bcN.pt --games 4 --debug [--opening 144]` (kagsim、対 v41。--opening で E058 の開幕を使う)。
4. PPO: `rl/ppo2.py --init tmp/rl/bcN.pt --out tmp/rl/ppoN.pt --iters 200 --games 32` (相手プール既定 = v41 / E060)。各 iter で `<out>.state`、`--resume` で再開。10 iter ごとに `<out>_itK.pt`。
5. 配備: `rl/export_agent.py tmp/rl/X.pt` → `agents/rl_agent/` (numpy 推論、weights.npz + features/actions/np_policy2 のコピー)。
   `tests/kag_eval.py agents/rl_agent/main.py --vs ...` で確認、tar は main.py + weights.npz + features.py actions.py np_policy.py np_policy2.py。

主要ファイル: `features.py` (観測→token、合法手マスク)、`actions.py` (行動の分解と復元)、`labels.py` (目的タイルのラベル)、`model2.py` (方策 v2/v3: 目的タイル attention + 作業ヘッド + 市場ヘッド + 価値)、
`act2.py` (推論: ユニット逐次に目的タイルを選び claim、到着時に作業)、`rollout2.py` (ベクトル化自己対戦、logp を保存)、`np_policy2.py` (numpy 推論)。
旧: `model.py` / `train_bc.py` / `bc_play.py` / `rollout.py` / `ppo.py` (v1: 方向を直接予測、崩壊した)。

経緯と結果は `.opencode/knowledge/experiments.md` の RL-BC 行。


## 自己対戦 PPO (`rl/sp/`、2026-09-15) — 基盤は完成、学習は規模不足で凍結

- `legal_all.py` 到着合法性のベクトル化 / `vec_env.py` 共有メモリ並列環境 (W ワーカー × K 局、両席とも方策、`--tapes` で記録相手) / `policy_batch.py` バッチ推論 (上位 8 候補 + 到着 PASS マスク + 目的地固定) /
  `train.py` PPO (決定単位の比率、凍結 teacher への解析 KL、critic 暖機、昇格、`--resume`) / `bench.py` スループット / `build_tapes.py`・`build_tapes_pq.py` 記録相手プール。
- 実行例: `caffeinate -i .venv/bin/python rl/sp/train.py --init tmp/rl/bc5_ep3.pt --out tmp/rl/spN.pt --workers 10 --games 48 --T 96 --lr 1e-5 --warmup 200 --critic-warmup 60 --temp 0.5 --kl 0.005 --tapes tmp/rl/tapes_top.pkl --tape-frac 0.25`
- Mac (M3 Pro, MPS): 1,178 game-steps/s (2,356 決定/s)。ボトルネックは推論。結果: 3 走とも 1〜3k 局で発散 (experiments.md SP1/SP3)。

## C++ 観測エンコーダ

- `kagsim.Game.encode(seat)` → tiles/units/items/glob/legal/pos/money/prices/seeds (`python/encode.hpp`)。features.encode + legal_all と完全一致。
- ビルド: `uv pip install pybind11 --python .venv/bin/python; cd third_party/kaggriculture-cppsim && ../../.venv/bin/python setup.py build_ext --inplace` (.so は git 管理外)。

## BC データ v4 (勝者側) と Kaggle 実行

- `rl/extract_winners.py --out tmp/rl/bcw --min-rating 2500 --parquet tmp/data/replays_2026-09.parquet --json 'tmp/top0915/**/episode-*.json' ...` → 勝者側 shard。
- 学習フォルダ `tmp/rl/bc6data/` (bcw + majkel + mmpq のシンボリックリンク)。Mac: `train_bc2.py --data 'tmp/rl/bc6data/*.npz' --out tmp/rl/bc6.pt --epochs 6 --bs 512 --lr 5e-4` (40 分/epoch)。
- Kaggle: `rl/kaggle/` (dataset-metadata.json → `kaggle datasets create -p <staging> -r zip`、kernel-metadata.json + run_bc6.py → `kaggle kernels push -p rl/kaggle`)。
  staging は tmp/kaggle_ds/{code/rl,code/kagsim,code/v41_main.py,data/bc6data,ckpt}。結果: `kaggle kernels output mmn0222/kaggriculture-bc6 -p tmp/kaggle_out` (bc6k_epK.pt, eval.txt, bc6k.pt.state)。
  再開は出力を dataset にして dataset_sources に追加 (スクリプトが bc6k.pt.state を見つけて --resume)。

## BC データ v5 (勝者+敗者の混合) と Kaggle 実行 (bc7、現行本線)

- 動機: 勝者限定 bc6 は val 改善と閉ループが逆相関 (ep1 31.6k が天井、基準 59.6k に届かず) のため、劣勢からの立て直しデモを補給する。
- `rl/extract_losers.py --out tmp/rl/bcl --parquet tmp/data/replays_2026-09.parquet --json 'tmp/e058_live/episode-*.json' 'tmp/e060_live/episode-*.json'`
  → 敗者側 shard (parquet の rating≥2500・敗北・bank≥80k = pql + 実戦の相手席 jsl/jsw、計 1,014)。
- 学習フォルダ `tmp/rl/bc7data/` (bc6data 2,492 + bcl 1,014 = 3,506 のシンボリックリンク)。
  パイロット: `train_bc2.py --data 'tmp/rl/bc7data/*.npz' --out tmp/rl/bc7.pt --epochs 2 --bs 512 --lr 5e-4 --max-games 1200` (RTX3060Ti、約 5 分/epoch、ep0 40.7k/−98k)。
- Kaggle: `rl/kaggle7/` (dataset `mmn0222/kaggriculture-rl-bc7` = code + bc7data 実コピー 674MB、kernel `mmn0222/kaggriculture-bc7` + run_bc7.py → `kaggle kernels push -p rl/kaggle7`)。
  staging は tmp/kaggle_bc7/ds/{code/rl,code/kagsim,code/v41_main.py,data/bc7data}。bs512・6 epoch・各 epoch 32 戦評価、約 7h。
  結果: `kaggle kernels output mmn0222/kaggriculture-bc7 -p /tmp/kaggle_out_bc7 --force` (bc7k_epK.pt, eval.txt)。判定基準は AGENTS.md「RL 路線」節。
- **メモリ注意 (2026-09-16)**: `load_shards` は省メモリ化済み (2 パス・事前確保、bc7 全量ピーク 13.4GB)。旧実装 (リスト＋concatenate、2 倍ピーク約 25GB) は
  15GB 機で OOM-killer → WSL フリーズの連鎖を起こすため使わない。パイロットは `--max-games 1200` (ピーク約 10GB)。

## 別環境での復元 (本線データのみ)

- git: コード (`rl/`、`rl/kaggle7/`、`tests/`) + 知識 (`.opencode/knowledge/`) + ドキュメント。
- Kaggle datasets (git 管理外の重み・shard): `kaggle datasets download mmn0222/kaggriculture-rl-bc7 -p /tmp/kaggle_ds --force` → 解凍 → `data/bc7data/` を `tmp/rl/bc7data/` へ (3,506 局)。
  bc6 系が必要なら `mmn0222/kaggriculture-rl-bc6` (bc6data 2,492 + `ckpt/bc5_ep3.pt` + `ckpt/bc6_ep0.pt`)。
- torch はシステム側 (`/usr/bin/python3`、2.13+cu130) + `PYTHONPATH=<repo>/.venv/lib/python3.14/site-packages` (kagsim 等)。venv を作り直す場合は先頭の手順。
