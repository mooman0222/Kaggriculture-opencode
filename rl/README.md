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

## raw 表現 (2026-09-24〜、現行) — 手順

engine の行動空間そのもの (`rl/raw.py`)。知見と到達点は `.opencode/knowledge/tracks/transformer.md` 0 節。

- 表現の検査: `.venv/bin/python rl/raw.py --check 'tmp/rootcause/v41json/episode-*.json' --team v41a` (記録を完全再現すること)
- データ生成 (自分のデータ): `rl/gen_selfplay.py --raw --agent <教師> --opp <相手...> --seed0 N --games M --out DIR` (`--dry LO,HI` で 1 日だけ水やりを抜く、`--noise` は DART)
- 回復データ: `rl/gen_recover.py --student <ckpt> --agent <教師> ...` (複製の誤りを実行し影の教師が 24 手回復)
- 学習: `rl/train_raw.py --data 'A/*.npz' 'B/*.npz' --out X.pt --epochs 8 --bs 512 --lr 7e-4 [--init ckpt]`
  (`--patience N --min-delta D`: val loss が D 以上改善しない epoch が N 回続いたら停止し `_best.pt` を残す。既定 0 = 固定 epoch。決まるのは最適化の収束だけで選択は閉ループゲートのまま)
- 評価: `rl/raw.py X.pt --games 32 --vs <相手>`、分岐の手番: `rl/diag/diverge_raw.py X.pt <教師> <相手> 12` (`HANDOFF=1` で分岐後を教師に任せる)
- Kaggle: `rl/kaggle27` (生成+学習)、`rl/kaggle28/30` (回復データ)、`rl/kaggle29` (時刻)、`rl/kaggle31` (雑草)、`rl/kaggle32` (標的 recover 150 局 + 最初から学習 + 新規 band seed 5200 で判定。`rl/` コードは同梱の target dataset 内 `code_rl/` を優先し panel-agents の再 upload なしに更新する)。コードは dataset `mmn0222/kaggriculture-rl-panel-agents` の `code_rl/` (更新は `cp rl/*.py tmp/kaggle_panel/code_rl/ && kaggle datasets version -p tmp/kaggle_panel -r zip`)
- 生成は Kaggle でも 1 局 2〜5 秒 (E081 は重い)。生成済みの shard は前のカーネルを `kernel_sources` に入れれば再利用できる (中身 `uop` で選ぶ。v41self には同名の旧形式 shard がある)

## Policy3: 一段 option 方策 (旧本線、行動表現の天井で打ち止め) — 現在地 2026-09-18

- `model3.py`: 各ユニットの `(目的タイル, 到着後の作業)` を 100 × 44 の joint option として直接スコアリング (op 別 rank-16 bilinear)。Policy2 の teacher-forcing と `prev` 入力を廃止。
  `bc_loss3` の `PASS_IDLE_WEIGHT` (学習器 `--pass-idle-weight`、既定 1.0) は未給水/未給餌が残る時の PASS ラベルの重み (bc11 で 0.1 を試し効果なし)。
- `train_bc3.py`: 既存 shard の `dest`/`dop` を joint label に使う。`--init` で互換重みを引き継ぐ (Policy2 の encoder・市場・数量、Policy3 checkpoint の全部)。
- `act3.py`: 現在地は実行可能手、遠隔地は地形上成立する将来作業を候補にし、選んだ option を到着まで保持。**推論に補完層・班分け・復号規則を足すと崩れる** (`tracks/transformer.md` 3 節)。
- `play3.py`: kagsim 対 v41 の閉ループ評価。`rollout3.py`: 自分の軌跡を shard 化 (DAgger 的、bc12 で不成立)。

### 知見と結果表は `.opencode/knowledge/tracks/transformer.md` にある

学習方策そのものについて分かったこと (checkpoint の系譜と成績、効いた手・打ち止めの一覧、測定の作法、
PASS ラベルのバグ、作物構成の診断、市場 pos_w、ラベルの既知の欠陥、ハイブリッド棄却の根拠) は**すべてそちら**。
このファイルは手順書 — 環境の作り方・データの取り方・学習と評価のコマンド・別 PC での再開に絞る。

**着手前に読むべき 3 点だけ再掲**:
- **判定は margin で行う。own では順位がつかない** (対 v41 64 戦で own の SE ±2,600〜2,900、margin ±1,300)。val 精度は判断材料にならない。
- **64 戦で margin 差 4k 未満は区別できない。** 同レシピの再現でも 3.3k ずれる。32 戦・16 戦の古い数字と直接比べない。
- **epoch は 4 が上限**、**教師は Majkel のみ**、**初期値は混合 1,200 局 (bc9k) を使い増やさない**。

### 次にすること (2026-09-18、優先順)

学習方策で残っている手は少ない。打ち止めの一覧は `tracks/transformer.md` の 3 節。

1. **人参の cold start** (唯一の未解決、得点差に直結): 閉ループで人参を 5.8 株しか植えず (師 68.8)、d8-19 の
   1 株あたりの実り 0.58 vs 0.82 の差になっている。給水は正常・両ヘッドの open-loop も正常・pos_w デバイアスでも戻らない。
   残るのは「d9 に Majkel が最初の人参を植える引き金」の特定。gate は盤面 (tiles) そのもの。
   反実仮想の道具は `scratchpad/probe.py` の形 (自分の閉ループ状態を集めて特徴を 1 つずつ動かす)。
2. **データ増量**: Majkel は日 50〜140 局増える。`fetch_episodes.py` で追加して bc15 レシピ。
   200 → 713 局で +7k の実績。713 → 854 局では動かなかったので、効くとしても対数的。
3. 市場バケツの丸め修正: 再取得 2.4 時間が前提 (`tracks/transformer.md` 6 節)。
4. **やらないこと**: epoch 追加、初期値のデータ増量、PASS 削減、pos_w デバイアス、自己軌跡、推論側の細工、
   容量 d256、PPO、**ハイブリッド (テープ + Transformer 売り層)**。すべて実測で棄却済み。

**ハイブリッドは 09-18 に棄却**: 実装前に伸びしろを測ったところ、対 v46 で **+$9/局** (対 v43 なら +$1,275)。
測定法は `scratchpad/headroom.py` の形 — シャシーを走らせながら「Transformer なら何を売るか」を並べて記録するだけで、
実装ゼロで答えが出る。v43 で +1,275 / v46 で +9 という予測は、実際に載せた E065 の +1,700 / +102 と一致した。
**同じ問いが来たら、まず伸びしろを測ること。**

### データと再開 (別 PC)

**まず環境**: 冒頭の「環境の再構築」で venv + kagsim をビルド。

**Kaggle dataset (git 管理外の shard と重み、すべて修正ラベル版)**

| dataset | 中身 |
|---|---|
| `mmn0222/kaggriculture-rl-majkel0917` | `data/majkel_all/` Majkel 854 局 (157MB)、`ckpt/` bc5_ep3・bc9k_ep3・bc15_ep3 |
| `mmn0222/kaggriculture-rl-pq0917` | `<team_id>/<episode>.npz` 上位 40 チーム 2,132 局 (424MB、公開データ由来) |
| `mmn0222/kaggriculture-rl-code` | `rl/` 平置き (コード配布用) |
| `mmn0222/kaggriculture-rl-bc7` | `code/kagsim` (C++ ソース) と `code/v41_main.py` (対戦相手) |

```
kaggle datasets download mmn0222/kaggriculture-rl-majkel0917 -p /tmp/kaggle_ds --force && cd /tmp/kaggle_ds && unzip -o -q '*.zip'
mkdir -p tmp/rl && cp -r /tmp/kaggle_ds/data/majkel_all tmp/rl/ && cp /tmp/kaggle_ds/ckpt/*.pt tmp/rl/
.venv/bin/python rl/play3.py tmp/rl/bc15_ep3.pt --games 32     # 95.4k / −17.1k / 2 勝 が再現すれば環境 OK
```

**⚠ 古い dataset (majkel0916 / bc6 / bc7) は旧ラベルのまま。落としたら必ず修復する**:
`.venv/bin/python rl/fix_pass_labels.py 'tmp/rl/<dir>/*.npz'` (誤ラベル集合は `dop==PASS & op∈MOVES` で shard だけから一意に決まる。冪等)。

**データを増やす**
- Majkel (日 50〜140 局増える): 提出 id は `kagglesdk` の `list_team_public_submissions(team_id=16718819)` (`tests/fetch_top.py` 参照)。09-17 時点は 56156662 と 56216119 の 2 本。
  `.venv/bin/python rl/fetch_episodes.py --sub 56156662 --team Majkel1337 --out tmp/rl/majkel_all --n 1000` (既存 id は飛ばす)。新しい提出が出たらその id も足す。
- 公開データ (`georgymamarin/kaggriculture-episodes`、毎日更新、全体 19.4GB): 月次 parquet は凍結され新しい局は **`replays_2026-09b` / `09c` に入る**ので、新しいシャードも落とす。
  `kaggle datasets download georgymamarin/kaggriculture-episodes -f <file> -p tmp/data --force` で `teams.csv` `episodes.csv` と各 parquet を取り、
  `.venv/bin/python rl/extract_parquet.py --parquet tmp/data/<file> --top 40 --out tmp/rl/pq` (既存 episode は飛ばす、実測 0.55 秒/局)。
  **クロールは数日遅れる** — 09-17 時点で top40 の局は 09-14 が 42 件、09-16 は 4 件しかない。「昨日の分」を期待しないこと。

**学習**
- Majkel fine-tune (bc15/bc18 レシピ): `.venv/bin/python rl/train_bc3.py --data 'tmp/rl/majkel_all/*.npz' --out tmp/rl/bcN.pt --init tmp/rl/bc9k_ep3.pt --epochs 4 --bs 128 --lr 2e-4`
  (Mac MPS 15.6 分/epoch @854 局、Kaggle T4 約 7 分 @713 局)。合計 4〜5 epoch を超えない。
- 初期値 (bc9 レシピ): `--data 'tmp/rl/pq/*/*.npz' --init tmp/rl/bc5_ep3.pt --epochs 4 --bs 128 --lr 5e-4 --max-games 2000`。
  **`--max-games` は RAM 制約** — `train_bc3` は全 shard を RAM に載せる (約 7MB/局)。2,000 局 ≈ 14GB、3,506 局は 15GB マシンで OOM した実績あり。

**評価**: 対 v41 `rl/play3.py CKPT --games 32` (32 戦 91 秒、SE ±3.1k)、対提出 `--games 16 --vs agents/e065/main.py`。
物差し (PASS / WATER / HARVEST / MOVE / d18 畑) は会話ログの scratchpad スクリプト、`play3.py` への組み込みは未着手。

**Kaggle で回す**
- コード更新: `cp rl/*.py tmp/kaggle_code/rl/; cp rl/sp/*.py tmp/kaggle_code/rl/sp/; kaggle datasets version -p tmp/kaggle_code -r zip -m msg`
- 雛形: `rl/kaggle15/run_bc19.py` (2 段学習 + 評価。入力は `$KAGGLE_ROOT/input/**/<目印ファイル>` で探す)。
  **乾式実行**: `KAGGLE_ROOT=<偽ツリー> KAGGLE_DRY=1 .venv/bin/python rl/kaggle15/run_bc19.py` (1 epoch / 4 局 / 2 戦に縮む)。偽ツリーは symlink でよい。
- 投入 `kaggle kernels push -p rl/kaggle15`、確認 `kaggle kernels status mmn0222/kaggriculture-bc19-policy3`、
  取得 `kaggle kernels output mmn0222/kaggriculture-bc19-policy3 -p tmp/kaggle_out_bc19 --force` (`eval.txt` に 32 戦の結果)。
- カーネルは**完了時にしか出力を保存しない**。長時間は塊に分ける。
- **落とし穴**: dataset をアップロードすると**トップ階層のディレクトリが剥がれて中身がルートに展開される**ことがある
  (`pq0917` は team_id ディレクトリが直下、`ckpt0917` は `ckpt/` の中身が直下)。`first()` の目印は**ディレクトリを含まないファイル名**にする。
  bc19・bc22 とも初回はこれで落ちた。偽ツリーの乾式実行では防げないので、`kaggle datasets files <ref>` で実際の展開形を必ず確認する。

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
