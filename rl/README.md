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

## Policy3: 一段 option 方策 (現行本線) — 現在地 2026-09-16

- `model3.py`: 各ユニットの `(目的タイル, 到着後の作業)` を 100 × 44 の joint option として直接スコアリング (op 別 rank-16 bilinear)。Policy2 の teacher-forcing と `prev` 入力を廃止。
  `bc_loss3` の `PASS_IDLE_WEIGHT` (学習器 `--pass-idle-weight`、既定 1.0) は未給水/未給餌が残る時の PASS ラベルの重み (bc11 で 0.1 を試し効果なし)。
- `train_bc3.py`: 既存 shard の `dest`/`dop` を joint label に使う。`--init` で互換重みを引き継ぐ (Policy2 の encoder・市場・数量、Policy3 checkpoint の全部)。
- `act3.py`: 現在地は実行可能手、遠隔地は地形上成立する将来作業を候補にし、選んだ option を到着まで保持。**推論に補完層・班分け・復号規則を足すと崩れる** (下の診断)。
- `play3.py`: kagsim 対 v41 の閉ループ評価。`rollout3.py`: 自分の軌跡を shard 化 (DAgger 的、bc12 で不成立)。

### 結果 (対 v41、Kaggle 16 戦 = seed 5000〜5007 両席 / local 8 戦 = 5000〜5003)

| 名 | 学習 | 16 戦 own | 備考 |
|---|---|---|---|
| bc9k_ep3 | bc7 混合 1,200 局、bc5_ep3 から 4 epoch (`rl/kaggle9/`) | 58.0k | local 8 戦 68.2k。家畜は demo 並みに回復、Policy2 の DIG/BUILD 荒らしは消えた |
| bc10k_ep3 | bc9 から継続 4 epoch lr 2e-4 (`rl/kaggle10/`) | 53.1k | 頭打ち |
| bc11k_ep3 | 同 + PASS_IDLE_WEIGHT 0.1 (`rl/kaggle11/`) | 57.8k | 待機は demo 並みに減るが浮いた手は MOVE/DIG へ |
| bc12k_ep3 | 同 + 自己軌跡 200 局混合 (`rl/kaggle12/`, `rollout3.py`) | 39.8k | 自分の欠点を強化して悪化 |
| **bc13_ep3** | **bc9k_ep3 から 1 位 Majkel1337 の直近 200 局のみ 4 epoch lr 2e-4 (Mac 20 分)** | **32 戦 88.5k / margin −27.8k** | local 8 戦 103.6k。WATER 1203・畑 57 株・2 日放置 4.0%。**基準 (bc5_ep3 59.6k/−69k) を初めて大きく超えた** |
| bc14_ep3 | bc13_ep3 から継続 4 epoch lr 1e-4 | 32 戦 87.7k / −37.3k | 飽和。最良は bc13_ep3 |

### 診断で分かったこと (scratchpad の診断スクリプトは会話ログ、結論は experiments.md 09-16 行)

- 得点差の正体は d9 以降の収入差で、畑は同数、**家畜と給水**が半分。open-loop 精度 (option 92%) は閉ループ得点と無相関。
- 1 位 Majkel1337 の実戦 16 局 (`tmp/top1_0916/`): PASS 54/局 (demo 529)、WATER 1387、渇死 17、2 日放置 1.9%、班分けなし。demo の平均像 (PASS 529) を写すことが BC の天井だった。
- 推論側の補完 (欠品なら倉庫へ・待機時に強制) と班分けマスクはいずれも悪化〜崩壊。方策は塞がれた option の代わりに PASS を選ぶ。
- 物差し: 得点のほかに **PASS ≦ 100 / WATER ≧ 1,300 / 2 日放置 ≦ 2% / 家畜逃走 ≈ 0** (1 位の値)。

### 方針 (2026-09-16 夜、ユーザー合意)

1. **本命 = 1 位 Majkel1337 の実戦だけを教師にした BC のスケール**。200 局で +30k (bc13) の事実が根拠。次は別提出のエピソードも辿って 400〜600 局にし、bc9k_ep3 から同じレシピ (4 epoch、lr 2e-4) で bc15。
   判定は `play3.py --games 32` (bc13_ep3 88.5k / −27.8k を超えるか) と 1 位の物差し (PASS / WATER / 2 日放置 / 渇死)。継続 epoch (bc14) は飽和済みなので、伸ばすのはデータ量。
2. **PPO は検証扱い** (成功見込み 3〜4 割): 前 3 走の敗因のうち決定数と信用配分には対応 (保持 option、渇死/逃走の密報酬、初期値 bc13)、規模と KL 錨は未解決。
   Kaggle で 2.5 時間の塊ごとに回し、**停止基準 = it25〜50 で渇死・逃走が減らない、または 32 戦が bc13_ep3 を下回る**。満たさなければ 1 塊で凍結に戻す。
3. 推論側の細工 (補完層・班分け・復号規則・待機ラベル・自己軌跡) は全部効かなかったので再挑戦しない。
4. 配備には `export_agent.py` の Policy3 対応 (numpy 推論) が要る。bc13 級が安定したら着手。

### データと再開 (別 PC)

- **Majkel 200 局 (取得済み)**: Kaggle dataset **`mmn0222/kaggriculture-rl-majkel0916`** (data/majkel_0916 200 shard 35MB、ckpt/bc9k_ep3.pt・bc13_ep3.pt、tapes/tapes_top.pkl 190MB)。
  `kaggle datasets download mmn0222/kaggriculture-rl-majkel0916 -p /tmp/kaggle_ds --force` → 解凍 → `tmp/rl/majkel_0916/`, `tmp/rl/*.pt`, `tmp/rl/tapes_top.pkl` へ。
- **追加取得**: LB csv は `kaggle competitions leaderboard kaggriculture --download -p DIR`。`tests/fetch_top.py --lb <csv> --top 1 --per 200 --out tmp/top1_<date>` はベスト提出 1 本の直近 200 局まで。
  他提出も取るには `fetch_top.py` の `subs[:1]` のループを広げる (submission id ごとに `ApiListSubmissionEpisodesRequest`、episode id で重複排除)。
  shard 化: `rl/extract.py --glob 'tmp/top1_*/*/episode-*.json' --team Majkel1337 --out tmp/rl/majkel_all`。旧 Majkel shard (`tmp/rl/majkel`, 8 月、306 局) は旧方策なので混ぜない。
- **学習 (Mac MPS 4 分/epoch、RTX3060Ti なら数分)**: `.venv/bin/python rl/train_bc3.py --data 'tmp/rl/majkel_all/*.npz' --out tmp/rl/bc15.pt --init tmp/rl/bc9k_ep3.pt --epochs 4 --bs 128 --lr 2e-4`
- **評価**: `.venv/bin/python rl/play3.py tmp/rl/bc15_ep3.pt --games 32` と診断 (op 数・渇死・2 日放置は会話ログの scratchpad スクリプト相当; `play3.py --debug` 相当は未実装、必要なら bc13 の診断値を README 上の表と比べる)。
- **Kaggle で回す場合**: コードは dataset `mmn0222/kaggriculture-rl-code` (rl/ を平置き、`kaggle datasets version -p tmp/kaggle_code -r zip` で更新)、bc7 データは `mmn0222/kaggriculture-rl-bc7`、
  重みは dataset か kernel 出力 (`kernel_sources`)。投入前に `run_*.py` を偽の `/kaggle/input` で乾式実行する (bc12 は経路探索と変数定義で 3 回落ちた)。カーネルは完了時にしか出力を保存しない。

## 自己対戦 PPO (`rl/sp/`、2026-09-15) — 基盤は完成、学習は規模不足で凍結

- `legal_all.py` 到着合法性のベクトル化 / `vec_env.py` 共有メモリ並列環境 (W ワーカー × K 局、両席とも方策、`--tapes` で記録相手) / `policy_batch.py` バッチ推論 (上位 8 候補 + 到着 PASS マスク + 目的地固定) /
  `train.py` PPO (決定単位の比率、凍結 teacher への解析 KL、critic 暖機、昇格、`--resume`) / `bench.py` スループット / `build_tapes.py`・`build_tapes_pq.py` 記録相手プール。
- 実行例: `caffeinate -i .venv/bin/python rl/sp/train.py --init tmp/rl/bc5_ep3.pt --out tmp/rl/spN.pt --workers 10 --games 48 --T 96 --lr 1e-5 --warmup 200 --critic-warmup 60 --temp 0.5 --kl 0.005 --tapes tmp/rl/tapes_top.pkl --tape-frac 0.25`
- Mac (M3 Pro, MPS): 1,178 game-steps/s (2,356 決定/s)。ボトルネックは推論。結果: 3 走とも 1〜3k 局で発散 (experiments.md SP1/SP3)。
- **Policy3 版 (2026-09-16、煙試験済み・本走は未着手)**: `policy_batch3.py` (joint option を到着まで保持、到着時の正確な合法性をマスク、上位 64 候補で比率と KL) + `train3.py`
  (密報酬に **渇死 1 株 −0.05・逃走 1 頭 −0.2** を追加、他は train.py 同様)。log-prob 再計算誤差 0、決定は 1 手 2.8〜3.9/席。
  実行例: `caffeinate -i .venv/bin/python rl/sp/train3.py --init tmp/rl/bc13_ep3.pt --out tmp/rl/sp4.pt --workers 8 --games 32 --T 96 --lr 1e-5 --kl 0.02 --temp 0.7 --dense 0.2 --death 0.05 --escape 0.2 --tape-frac 0.25 --critic-warmup 60`
  判定は 25 iter ごとの `_itK.pt` を `play3.py --games 32` と上の物差しで。前 3 走は it39〜50 で崩れた。
- **Kaggle 実行 (2026-09-16 夜、`rl/kaggle13/run_sp4.py`、kernel `mmn0222/kaggriculture-sp4-policy3-ppo`)**: 初期値 bc13_ep3、GPU、4 worker × 48 局、学習 150 分 (`--max-minutes`) → 全 `_itK.pt` を対 v41 32 戦 → `eval.txt`。
  続きは同カーネルを `kernel_sources` に足して再投入 (`sp4k.pt.state` を見つけて `--resume`)。結果取得 `kaggle kernels output mmn0222/kaggriculture-sp4-policy3-ppo -p tmp/kaggle_out_sp4 --force`。
  この PC での本走 (it2 まで) は暖機中で |r−1|=0、渇死 477→2347/iter (d0→d12 の成長分)。

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
