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

## Policy3: 一段 option 方策 (BC は有効・PPO は凍結) — 現在地 2026-09-19

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

### 次にすること (2026-09-19、PPO 凍結後、優先順)

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

## 自己対戦 PPO (`rl/sp/`) — 2026-09-18 に**探索の設計を入れ替えて再開**

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

### 2026-09-18: 目的地を貪欲に固定した PPO (`--greedy-dest`) — 結果は棄却

**なぜ**: 学習ゼロの測定 (`experiments.md` の 4d 行) で、得点が行動ノイズに対し **`own ≈ 63,072 × (1−ε)^10`** で落ちると分かった。
内訳は **目的地ノイズが損失の 85%** (単独で `(1−ε)^7.6`、ε を倍にすると損も倍) に対し、**作業ノイズは飽和** (ε 0.05 → 0.10 で −15.2k → −16.6k)。
さらに**温度掃引は平ら** (temp 0.05〜0.7 で own 45〜56k、0.05 が 0.2 より悪く非単調 = 差はノイズ) なので、
**収集温度を下げても貪欲比 20% は戻らない**。per-step で目的地をサンプリングする探索は、このゲームでは代償が大きすぎる。

**何を変えたか** (既定オフ。付けなければ従来と完全に同じ):

| 場所 | 変更 |
|---|---|
| `policy_batch.py` | `greedy_dest=True` で目的地を argmax に固定し、**log-prob を 0** にする。PPO 側は `live = (old != 0)` で自動的に除外 (committed と同じ扱い) |
| `train.py` | `--greedy-dest`。あわせて**エントロピー賞与を op ヘッドへ移す** (`ENT_HEAD`)。目的地を固定したまま dest のエントロピーを上げても鎖を壊す方向に押すだけ |
| `train.py` | `--max-minutes` (Kaggle の時間箱)、`--shuffle-adv` (advantage を無作為に入れ替える対照。critic・KL・比率の分布は無傷) |
| `act2.py` / `play2.py` | `--eps` (目的地) / `--eps-op` (到着時の作業) — 4d の測定ノブ。学習には使わない |

**検証の作法**: `SP_DEBUG_LP=1` を付けると初回更新の `|r−1|` が出る。**0 でなければ log-prob の帳尻が壊れている**。
`--greedy-dest` では `max|dlogp|` は非ゼロになる (合計 logp に貪欲化した dest の項が含まれるため) が、
**PPO が使う決定単位の `|r−1|` は 0 でなければならない**。実測 0.0000 を確認済み。

**ローカル実行** (Mac MPS で 143 s/iter、更新が 114s を占める):

```
SP_DEBUG_LP=1 .venv/bin/python rl/sp/train.py --init tmp/rl/bc5_ep3.pt --out tmp/rl/gd_smoke.pt \
  --tapes tmp/rl/tapes_top.pkl --tape-frac 0.25 --workers 2 --games 4 --T 24 --iters 1 --greedy-dest   # 煙試験
caffeinate -i .venv/bin/python rl/sp/train.py --init tmp/rl/bc5_ep3.pt --out tmp/rl/gd.pt \
  --tapes tmp/rl/tapes_top.pkl --tape-frac 0.25 --workers 10 --games 48 --T 96 --iters 60 --greedy-dest
```

**Kaggle 実行** (`rl/kaggle18/run_gd.py`、kernel `mmn0222/kaggriculture-ppo-greedy-dest`):
base (SP3 と同じレシピ) と gd (`--greedy-dest`) の 2 本を順に 60 iter / 各 150 分上限 → 全 `_itK.pt` を対 v41 32 戦 → `eval.txt`。

```
.venv/bin/kaggle kernels push -p rl/kaggle18
.venv/bin/kaggle kernels status mmn0222/kaggriculture-ppo-greedy-dest
.venv/bin/kaggle kernels output mmn0222/kaggriculture-ppo-greedy-dest -p tmp/kaggle_out_gd --force
```

**再走する場合**: 出力を dataset にして `kernel-metadata.json` の `dataset_sources` に足すと、
`ppo_base.pt.state` / `ppo_gd.pt.state` を見つけて `--resume` する (run_sp4 と同じ形)。

**結果 (09-18 完走)**: base / gd とも teacher 昇格後に崩壊し、**棄却** (gd の最終 own は base より低い)。
数値と解釈は `experiments.md` と `tracks/transformer.md` 4 節・9 節 4。advantage シャッフル対照も
完走し、**shuf の方が劣化が遅い = 信号に従うことが崩壊を悪化させる**と判明 (`rl/kaggle19/run_shuf.py`、
kernel `mmn0222/kaggriculture-ppo-adv-shuf`。ログは `tmp/kaggle_out_ctl/` と `tmp/kaggle_out_shuf/`)。
4b/4c (`rl/kaggle20/run_4b4c.py`、kernel `mmn0222/kaggriculture-ppo-nopromo-ent0`) も完走:
**4b 部分的 (final own 24.3k)、4c 無効 (17.3k vs base 17.4k)**。ログは `tmp/kaggle_out_4b4c/`。
残るのは pg 本体 (advantage / 信用割当) — 09-19 の診断で **critic が定数**と判明 (下記)。

### 2026-09-19: advantage/critic 診断 (`rl/diag_adv.py`) と PPO 凍結

学習なしで「PPO が使う信号」の情報量を測る。8 窓 (T=96) 回すと 8 窓目に終局が入る (720 手周期)。
局所実行は venv に torch が無いのでシステム torch と venv の kagsim を混ぜる:

```
PYTHONPATH=.venv/lib/python3.14/site-packages /usr/bin/python3 rl/diag_adv.py \
  --ckpt tmp/rl/bc5_ep3.pt --out tmp/rl/diag_x.json --workers 4 --games 120 --dev cuda
```

結果: 崩壊後 ckpt は V が定数 (R²≈0、最終手の相関 0.13)、同じ特徴の焼き直しは R²_val 0.44〜0.58。
解釈は `tracks/transformer.md` 5.5、数値は `experiments.md`。

**処方 1 — value head 専用 lr** (`rl/kaggle21/run_vlr.py`、kernel `mmn0222/kaggriculture-ppo-vlr`):
`--vlr 1e-3 --vwarmup 20` で critic の相関は 0.13 → **0.55** に改善したが、スケールが 10 分の 1 のまま
(R²_mc 0.05) で崩壊は残った (final own 33.2k、基準 63.1k、プロモーション 0 回)。ログは `tmp/kaggle_out_vlr/`。

**処方 2 — value head 事前学習** (`rl/vpretrain.py`、`rl/kaggle22/run_vpt.py`、
kernel `mmn0222/kaggriculture-ppo-vpretrain`): 2000 step 事前学習 (R²_val 0.289、終局行 0.202) してから
SP3 + `--vlr 1e-3` を回しても崩壊は同形 (final own 33,551、it25 46,564、昇格 0 回)。ログは `tmp/kaggle_out_vpt/`。

**結論: critic の質を 3 水準 (定数 → corr 0.55 → R²0.29) 変えても崩壊は変わらないため PPO は凍結。**
以降の学習系は BC (データ増量) のみ。コード dataset は vpretrain.py 入りで更新済み (md5 照合済み)。

**bc5_ep3 (Policy2) を使うのは崩れた 3 走と揃えるため。`train3.py` / `policy_batch3.py` には未移植で、
現行最良の bc18_ep3 (94k 帯) より 30k 下の古い方策である点に注意** — 理由・限界・移植の可否は
`tracks/transformer.md` 9 節 4。

**別 PC で再開するとき** (この節だけで足りる):

```
git pull
# 冒頭「環境の再構築」で venv + kagsim をビルド
kaggle datasets download mmn0222/kaggriculture-rl-majkel0917 -p /tmp/ds --force   # ckpt/bc5_ep3.pt
kaggle datasets download mmn0222/kaggriculture-rl-majkel0916 -f tapes/tapes_top.pkl -p tmp/rl --force
mkdir -p tmp/rl && cp /tmp/ds/ckpt/bc5_ep3.pt tmp/rl/
.venv/bin/python rl/play2.py tmp/rl/bc5_ep3.pt --games 32     # own 63.1k / margin −71.5k が出れば環境 OK
```

**コードを Kaggle 側へ反映**するのを忘れないこと (カーネルは dataset のコードを使う):

```
rm -rf tmp/kaggle_code && mkdir -p tmp/kaggle_code/sp
cp rl/*.py tmp/kaggle_code/ && cp rl/sp/*.py tmp/kaggle_code/sp/
echo '{"title":"kaggriculture-rl-code","id":"mmn0222/kaggriculture-rl-code","licenses":[{"name":"CC0-1.0"}]}' > tmp/kaggle_code/dataset-metadata.json
.venv/bin/kaggle datasets version -p tmp/kaggle_code -r zip -m "msg"
```

**罠**: `bc7` dataset にも古い `code/rl/sp/train.py` が同居している。`run_gd.py` は
「`greedy_dest` を含む `sp/train.py`」を探して正しい方を選ぶので、**新機能を足したら目印の文字列も更新すること**。

**次に効く手が尽きたら**: 探索を重み空間へ (1 局につき重みを 1 回だけ揺らす) か、階層化 (実行層を決定的に固定し
マクロ決定だけ学ぶ)。`tracks/transformer.md` 9 節 4。

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

## GCP T4 実行記録 (2026-09-20〜、ブランチ `rl/greedy-dest-ppo`)

VM: n1-standard-4 + T4スポット。DLVM (Ubuntu 24.04, CUDA 12.9) + `python3 -m venv --system-site-packages .venv`。
ユーザー操作: GCP設定・VM・停止/削除。コード・コマンドはエージェント側で用意する。

### 済み (VM 上のパスは永続ディスク。VM削除で消える)
| # | 内容 | 結果 |
|---|---|---|
| T1 | bc15 再現 (`train_bc3 --data /tmp/kaggle_ds/data/majkel_all/*.npz --init ckpt/bc9k_ep3 --epochs 4`) | margin **−15,608** (64戦、基準 −15,382 と区別不能) = 環境正常 |
| T2 | fresh 再学習 (`/tmp/rl/majkel0920` 636局=旧400+新236、バケツ修正つき) → bcFresh_ep3 | margin **−55,350・0勝 = 崩壊** (64戦) |
| T3 | 旧のみ再学習 (`/tmp/rl/maj_old` 400局、新コード) → bcOld_ep3 → 32戦 | margin **−51,040・0/32 = 崩壊継続**。新提出データ毒説は棄却 (旧提出のみでも崩壊) |
| T4 | **バケツ修正 revert 検証**: `rl/features.py` を 982e387 の前に戻し、同一 400局 (398局一致) を旧ラベルで再取得 (`/tmp/rl/maj_revert`) → bcRevert_ep3 → 32戦 | margin **−16,118・0/32 = 完全復帰** (T1 −15,608 と区別不能、差 510)。**982e387 が毒と確定** |
| T5 | スケール: 旧ラベル全結合 (`/tmp/rl/maj_combo` = majkel_all 854 + maj_revert 400、重複除き 1,118局) → bcCombo_ep3 → 64戦 | margin **−21,877・4/64**。400局 (−16.1k) より **−5.8k 悪化**。古い時代 (56156662 含む) を混ぜると逆効果 |
| T6 | 純度: 同時代の混合 (`/tmp/rl/maj_recent` = maj_revert 400 + maj_new 247、新提出は旧コードで取得) → bcRecent_ep3 → 32戦 | margin **−19,968・3/32**。単独 −16.1k との差 3.9k は有意でない (t≈0.9)。混合は無益・無害の境界 |
| T7 | 新教師単独 (`/tmp/rl/maj_new` 247局) → bcNewPure_ep3 → 32戦 | margin **−18,664・3/32**。247局でも coherent、主提出と同水準。教師v2 として成立 |

### T8 人参cold-startの機序特定とOP_BOOST棄却 (2026-09-20、T4本線)
- 「d9の引き金」は存在しない: 教師の初回人参植えは全期間に分散 (d9:18/d11:48/d16:41/d25:37…、392/400局)。
  初回時の文脈 = 資金43k + 小麦26株立 + 苺23株立 + 人参種6個 = **小麦エンジンが回っている豊かな状態**。
- 方策 (bcRevert) の乖離: 人参種購入が **d20以降のみ** (d8-19ゼロ。教師はd8から継続購入)。
  d8-12 の資金 **2k vs 43k**、小麦立 **9 vs 26**。小麦不足→資金枯渇→種が買えない→人参ゼロの自己強化。
- 根の機序 (open-loop、教師dest付与): PLANT_WHEAT 想起率 **0 (全日)**。教師の小麦植え場面120の top-1 は
  **同タイルWATER 83 (69%)**、(dest,WHEAT) の joint 順位は中央値5位・1位1/120。
  空タイルで WATER (41.7%) が WHEAT (3.3%、12.5倍差) に勝ち続ける。
- 処方 OP_BOOST (`train_bc3.py` に環境変数で追加、既定オフ): PLANT_WHEAT/CARROT の損失重み×3 → bcWheat_ep3。
  結果: 想起は **4/120 で不動**、閉ループ **−24,922・0/32** (基準−16,118 から−8.8k悪化、t≈2)。**棄却**。
  損失重みでは学習済み順位が動かない + 水カバレッジ側の破損とみられる。
- 結論: 人参問題は「小麦を植えない→金がない→種が買えない」の連鎖で、joint-option の argmax 病。
  学習側の安い手は出尽くし。残るは容量 (d256+2,000局)・階層化・クラスタPPO = **T4以上が必要**。

### T4 の詳細 (2026-09-20、原因特定まで)
- ラベル差分 (同一局の新旧比較 140局): mkt 0.32% / dqty 0.009% が flip。量は微少だがキリ番注文 (~48注文/局) が systematic に +1 バケツ。
- 行動指紋 (8戦プローブ、対v41同一シード): 毒 (bcOld/bcFresh) は人参 77株/局・種 +60/局の overbuy → 畜産 (MILK 155 vs 263、EGG 0 vs 46) 崩壊 → own −35k。
  revert (bcRevert) は人参 41・MILK 263・STRAWBERRY 234 と bc15 (36/233/225) に復帰。
- val 精度は盲目: bcOld 0.898 / bcRevert 0.890 (毒の方が高い)。閉ループのみが検出する。
- 結論: **982e387 を revert したままにする** (作業ツリーは revert 済み・未コミット)。将来の再取得は旧コードで行い、
  `/tmp/rl/POISON_maj_old_newbucket` `/tmp/rl/POISON_majkel0920_newbucket` は新ラベル毒のため学習に使わない。
  監査の売り過小 (55.8個/局) の修正は、締切後の課題 (生数量保存つき再取得 + 初期値からの作り直しが前提)。
- 新提出 56332038 の扱いは T6/T7 で確定: 同時代の混合は無益・無害の境界 (T6 −19.9k、t≈0.9)、
  新単独は coherent で教師v2 として成立 (T7 −18.7k)。混ぜても伸びないので、増量は**同一提出の trickle**に限る。

### 次 (別 VM で再開するとき)
1. 本コミットで revert 確定 (982e387 を打消し)。将来の再取得はこのコードで行うこと。
   新ラベル shard (`/tmp/rl/POISON_*`) は学習に使わない。
2. **T4 以降の到達点 (09-20): BC の費用対効果ある改善は打ち止め。** T5/T6/T7 の行列が示すのは「400 局の最新データが天井で、足しても混ぜても伸びない」。
   T4 スポットで回す価値のある学習実験は残っていない。次に GPU を使う条件は (a) 日 50〜140 局の trickle が数日溜まって recent-800 が組める、
   (b) d256 + 2,000 局級 (要 VRAM/時間の見積もり直し)、(c) PPO のクラスタ規模化 — いずれも「より良い GPU が必要」の側。
   それまでは **VM を停止** (永続ディスクは残る、削除しないこと) して待つのが正解。
3. Majkel提出一覧 (09-20 取得): team 16718819 の公開提出は 2 本のみ — 56216119 (3277.6) と 56332038 (3093.8)。56156662 は一覧から外れた (古いため)。
   description は API で取れず (None)。スコアは変動するので再取得すること。
4. 判定基準 (tracks/transformer.md §3): margin で判定 (own不可)。64戦で差4k未満は区別不能。val精度は判断材料にしない (T4 で再確認)。
   T5 (−21.9k 64戦) vs T4 (−16.1k 32戦) の差 5.8k は有意水準の境界 — 古い時代の混合が有害という方向の主張に留める。
5. データ配置の現況: 良品 = `/tmp/rl/maj_revert` (主400)・`/tmp/rl/maj_new` (新247)・`/tmp/kaggle_ds/data/majkel_all` (旧854)。
   毒 = `/tmp/rl/POISON_*` (新ラベル)。再開時は `maj_revert` + その後の trickle を旧コードで足す。

### 注意
- 教師は Majkel のみ。混合は初期値 (bc9k) に限定
- epoch 上限 4 (初期値の有無によらず)。val と閉ループ逆相関に注意
- 学習・評価コマンドの定型は tracks/transformer.md §2 を見ること

## 新VMでの1からの再現手順 (2026-09-20、GPU調達待ち用)

前提: GPUつきUbuntu (DLVM推奨)。ブランチ `rl/greedy-dest-ppo` の本コミット以降
(revert = `rl/features.py` の bucket が旧 `argmin` 版であること。確認:
`grep -c argmin rl/features.py` → 1)。**同点→大のバケツ変更 (982e387) を再導入しないこと** (T4で毒確定)。

1. リポジトリ: `git clone <repo> && git checkout rl/greedy-dest-ppo`
2. 環境:
   ```
   python3 -m venv --system-site-packages .venv
   .venv/bin/pip install -r requirements.txt
   uv pip install --python .venv/bin/python -e third_party/kaggriculture-cppsim   # 約1分
   .venv/bin/python -c "import kagsim; print(kagsim.Game(1).encode(0)['tiles'].shape)"  # (2,10,10,18)ならOK
   ```
3. Kaggle認証: `~/.kaggle/kaggle.json` を置く (600)。確認は手順5の提出一覧クエリ
   (team_id 16718819 → 提出が2本出ればOK)。
4. 既存データ (dataset `mmn0222/kaggriculture-rl-majkel0917`):
   ```
   kaggle datasets download mmn0222/kaggriculture-rl-majkel0917 -p /tmp/kaggle_ds --force && cd /tmp/kaggle_ds && unzip -o -q '*.zip'
   mkdir -p /tmp/rl && cp -r /tmp/kaggle_ds/data/majkel_all /tmp/rl/ && cp /tmp/kaggle_ds/ckpt/*.pt /tmp/rl/
   # → /tmp/rl/majkel_all/ 854局 + /tmp/rl/*.pt (bc9k_ep3.pt = 初期値、bc5_ep3.pt、bc15_ep3.pt)
   ```
5. 新規取得 (**旧コードで**。新規約13局/分、中断しても既存idは飛ばすので再実行で継続):
   ```
   .venv/bin/python rl/fetch_episodes.py --sub 56216119 --team Majkel1337 --out /tmp/rl/maj_revert --n 400
   .venv/bin/python rl/fetch_episodes.py --sub 56332038 --team Majkel1337 --out /tmp/rl/maj_new --n 400
   # 期待: maj_revert 400局 (listing変動で±数局)、maj_new 約250局 (歩留まり約6割)
   ```
   提出IDが変わっていたら取り直す: `kagglesdk` で `list_team_public_submissions(team_id=16718819)`。
6. 健全性チェック (必須。新規shardが壊れていないことの確認):
   `.venv/bin/python rl/fix_pass_labels.py` は**不要** (現行 `labels.py` は修正済みを直接出す)。
   目安: PASSラベル約50/局、人参 PLANT 約67/局 (主)・約40/局 (新)。
7. 学習レシピ (すべて `--init <ckpt>/bc9k_ep3.pt --epochs 4 --bs 128 --lr 2e-4`。所要はT4実績):
   ```
   .venv/bin/python rl/train_bc3.py --data '/tmp/rl/maj_revert/*.npz' --out /tmp/rl/bcX.pt --init /tmp/kaggle_ds/ckpt/bc9k_ep3.pt --epochs 4 --bs 128 --lr 2e-4
   # 400局≈15分、647局≈23分、1,118局≈40分、247局≈10分
   ```
   結果の目安 (対v41、`--sell-rule-c`): 主400局 → margin −16k (32戦)。854局再現 → −15.6k (64戦)。
8. 評価: `.venv/bin/python rl/play3.py /tmp/rl/bcX_ep3.pt --games 32 --sell-rule-c` (約5分。64戦は約10分)。
   **判定は margin** (32戦 SE ±3.1k、64戦で差4k未満は区別不能)。**val精度は合否に使わない** (毒でも0.898が出る)。
9. 長時間実行の罠: ターミナル断・ツールタイムアウトでプロセス群が殺される。
   必ず切り離して起動し、ログで監視すること:
   ```
   setsid nohup .venv/bin/python rl/train_bc3.py ... > /tmp/train_x.log 2>&1 < /dev/null &
   grep -E "^epoch" /tmp/train_x.log   # 監視
   ```
   学習は `<out>.state` + `--resume` で再開可。fetch/train の並列実行可 (fetchはCPU/API律速)。
10. 禁止事項: epoch追加 (4上限)、初期値のデータ増量、他チーム混合、推論側の細工、PPO再開 —
    いずれも `tracks/transformer.md` 4節で実測棄却済み。OP_WEIGHT 実験 (`OP_BOOST` 環境変数) も T8 で棄却。
    `tracks/transformer.md` 9節の「生きている問い」が現在の有効手のすべて。
