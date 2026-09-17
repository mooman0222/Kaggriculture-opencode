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

## Policy3: 一段 option 方策 (現行本線) — 現在地 2026-09-17

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
| bc14_ep3 | bc13_ep3 から継続 4 epoch lr 1e-4 | 32 戦 87.7k / −37.3k | 200 局では飽和 |
| **bc15_ep3** | **bc9k_ep3 から Majkel 2 提出の全量 713 局で 4 epoch lr 2e-4 (Mac 54 分)** | **32 戦 95.4k / −17.1k / 2 勝** | 現在の最良。dataset `mmn0222/kaggriculture-rl-majkel0916` v3 (data/majkel_all, ckpt/bc15_ep3.pt) |
| **bc18_ep3** | **修正ラベル + Majkel 854 局、bc9k_ep3 から 4 epoch lr 2e-4 (Mac 62 分)** | **32 戦 92.8k / −16.9k / 3 勝** (ep2 95.9k/−20.3k) | PASS 486→179・MOVE 2982→3373 と行動は 1 位に接近、得点は bc15 と同格 |
| bc9k19_ep3 | 公開データ pq 2,000 局で bc5_ep3 から 4 epoch lr 5e-4 (Kaggle `rl/kaggle15`、73 分) | 32 戦 **49.6k / −79.6k** | **val option 92.0% = 全実行で最高**なのに閉ループ最低。open-loop と閉ループの無相関の最も鮮明な例 |
| bc19_ep3 | 上を初期値に Majkel 854 局で 4 epoch lr 2e-4 (31 分) | 32 戦 **88.0k / −22.9k / 3 勝** (ep2 90.1k/−27.0k) | Majkel fine-tune 系で**最低**。初期値の局数 634→2,000 は逆効果 |
| **bc20_ep3** | **修正ラベル + Majkel 854 局のみ、初期値なし 4 epoch lr 5e-4 (Mac 62 分)** | **32 戦 92.1k / −21.4k / 2 勝** | ep0 0.7k → ep2 62.3k → ep3 92.1k と**未収束**。混合初期値の寄与は測定限界以下。PASS 156・畑 57 |
| bc21 | bc20_ep3 から Majkel 854 局で 2 epoch 継続 lr 2e-4 | 32 戦 ep0 89.4k/−31.7k、**ep1 87.8k/−22.8k/1 勝** | **伸びず**。val は 0.749→0.704 と改善するのに閉ループは 92.1k→87.8k = bc16 と同じ崩れ方。**epoch 上限はスクラッチにも当てはまる** |
| bc16k_ep0〜3 | bc15_ep3 から同データで 4 epoch 継続 lr 1e-4 (Kaggle `rl/kaggle14`) | ep0 94.4k / −14.3k / 6 勝 → ep3 64.6k / −84k | 5 epoch 目までは同格、以降は val が上がりながら閉ループ崩壊。epoch は打ち止め |

### 診断で分かったこと (scratchpad の診断スクリプトは会話ログ、結論は experiments.md 09-16 行)

- 得点差の正体は d9 以降の収入差で、畑は同数、**家畜と給水**が半分。open-loop 精度 (option 92%) は閉ループ得点と無相関。
- 1 位 Majkel1337 の実戦 16 局 (`tmp/top1_0916/`): PASS 54/局 (demo 529)、WATER 1387、渇死 17、2 日放置 1.9%、班分けなし。demo の平均像 (PASS 529) を写すことが BC の天井だった。
- 推論側の補完 (欠品なら倉庫へ・待機時に強制) と班分けマスクはいずれも悪化〜崩壊。方策は塞がれた option の代わりに PASS を選ぶ。
- 物差し: 得点のほかに **PASS ≦ 100 / WATER ≧ 1,300 / 2 日放置 ≦ 2% / 家畜逃走 ≈ 0** (1 位の値)。

### 方針 (2026-09-16 夜、ユーザー合意) と現在地

1. **本命 = 1 位 Majkel1337 の実戦だけを教師にした BC のスケール**。200 局 → 713 局で 88.5k → 95.4k (bc13 → bc15)。伸ばすのはデータ量と epoch。
   判定は `play3.py --games 32` (対 v41) と 1 位の物差し (PASS 54 / WATER 1387 / 2 日放置 1.9% / 渇死 17)。
2. **提出エージェントとの距離**: bc15_ep3 は E065 / E062 (現行提出、v41 系テープ + 反応層) に 16 戦で **−18.4k / −18.1k** (1 勝)。提出は v41 に +2.7k、bc15 は −17k で差は約 20k。置き換え候補になるにはこの差を埋める。
3. **PPO は再凍結** (sp4: it52 で停止基準到達、渇死 −22% だが逃走 +60%、対テープ margin −11k → −39k、前 3 走と同形)。再挑戦は規模 (10^6 局/日級) が前提。
4. 推論側の細工 (補完層・班分け・復号規則・待機ラベル・自己軌跡) は全部効かなかったので再挑戦しない。
5. 配備は対応済み (09-17、「次にすること」4. を参照)。提出実績 ref 56283017 (trial)。

**PASS ラベルのバグ修正 (2026-09-17)**: `labels.py` は「その日の残り時間に作業が無い」ユニットを一律 (現在タイル, PASS) にしていたため、
夕方に目的地へ歩いている途中のユニットまで PASS 教師になっていた。Majkel の実測 PASS 62/局 に対しラベルは **461/局** (うち 404 は生の行動が MOVE)。
h=23 では 64%、h=22 で 37% が PASS ラベル。bc15_ep3 の open-loop 予測 PASS は 448/局 で、ラベルの時刻分布と一致 — 方策は壊れたラベルを忠実に写していた
(物差しの「PASS ≦ 100」が届かなかった真因)。修正: 移動中で当日中に作業が無い場合は `dest=-1` (bc_loss3 の `present` から除外)、実際に待機していたときだけ PASS。
ラベルは 461 → 57/局、学習対象は 5.6% 減。**既存 shard は `.venv/bin/python rl/fix_pass_labels.py 'tmp/rl/majkel_all/*.npz'` で修復**
(Kaggle dataset `majkel0916` v3 のデータも旧ラベルなので、落としたら必ず実行する)。再学習は bc15 レシピで。

**bc18 (完了、ラベル修正の効果測定)**: 修正ラベル + Majkel 854 局 (713 +141、09-17 取得) を bc15 と同一レシピで学習。
対 v41 32 戦 ep0 76.5k/−45.1k → ep1 85.0k/−26.7k → ep2 **95.9k**/−20.3k → ep3 92.8k/**−16.9k**/3 勝 (bc15_ep3 95.4k/−17.1k/2 勝)。
**得点は横ばい (SE ±3.1k)**。一方 4 戦の物差しは PASS 486→**179**、WATER 1161→1259、HARVEST 388→411、MOVE 2982→**3373** (Majkel 3400)、d18 畑 54→56 と全項目で 1 位に接近。
→ **PASS は得点の律速ではなかった** (bc11 の「待機は症状」を、原因を除去した形で再確認)。残る 18k は収穫 (411 vs 497) と給水 (1259 vs 1387) のスループット。
残る PASS 179 の大半は h22〜23 = 修正で教師を外した領域で、初期値 bc9k_ep3 (open-loop PASS 436) の prior が埋めている。
**ただしこれはラベルのバグではない**: 混合データを修復しても bcw で masked は 41/局だけ (Majkel は 401/局)、修復後も bcw の生 PASS 532・ラベル 389、pq は 547/499。
上位陣は本当に 530 回待機している。初期値をラベル修復で焼き直しても PASS は削れない (実測済み)。
ラベル修正の価値は得点ではなく「教師の写しとして正しくなったこと」= データ増量・PPO 初期値の素性と、物差し PASS が初めて方策の実力を測れるようになったこと。

**bc16 (完了)**: bc15_ep3 の継続は ep0 が同格 (94.4k / −14.3k)、以降は崩壊 (ep3 64.6k)。epoch は合計 4〜5 が上限で、**残る本命の手はデータ量 (Majkel の日次エピソード追加)**。次に効く見込みの手: 市場層の売り時刻 (夕方一括、規則 C で +3〜5k、experiments.md「bc15 vs E065 の負け方」)、d9〜15 のメロン収穫の遅れ (残り 13k) の調査。

### 収穫・給水スループットの診断 (2026-09-17、bc18_ep3 対 v41 4 戦 vs Majkel 8 局)

指標はすべて `features.encode` の tiles/units から取り、教師 (shard) と自分 (閉ループ) で定義をそろえた (`scratchpad/throughput.py` `yield_diag.py` `seed_diag.py` `plantday.py` `probe.py`、会話ログ)。

**給水は問題ない — 物差しの読み違いだった**。日末に給水済みの株の割合は d8-19 で **自 0.77 / 師 0.75**、2 日放置は両者 0.00。
「WATER 1259 vs 1387」は総手数の差であって、株あたりのカバレッジでは差がない。**この項目は追わない**。

**収穫の差の正体は「1 株あたりの実り」**: d8-19 で畑株数は同じ (53.6 / 52.4) なのに yield_units は **0.58 / 0.82**、実っている株の割合は **0.30 / 0.49**。

**原因は作物構成**。PLANT 要求/局: WHEAT 179.0/152.6、**CARROT 5.8/68.8**、TOMATO 3.5/5.8、**STRAWBERRY 52.0/35.2**、MELON 12.0/12.5。
初収穫までが 2 日の人参を捨て、10 日かかる苺を積んでいる。種切れではない (**空打ち 0 件**)、`decode_action` の 10 注文上限も無関係 (超過 0.3%、捨てられるのは HIRE のみ)。

**両ヘッドとも open-loop では正しい**: 農場ヘッドの PLANT_CARROT 一致率 **91.1%**、市場ヘッドの人参種の予測は **93.8 個/局 (ラベル 64.7)** とむしろ多い。
閉ループでは人参の種在庫が**全期間 0**。純粋な state distribution の乖離。

**分岐点は d6-8**: 苺を 19.5 株 (師 11.1) 植えて畑を固める。以後、自分の畑に人参 0 株 (師 6 株) の盤面しか現れず人参を買う文脈が来ない自己強化ループ。
種在庫・資金・株数を個別に動かす反実仮想では人参買いは復活せず、gate は盤面 (tiles) そのもの。

**`pos_w` デバイアスの A/B (棄却)**: `model._mkt_ce` は市場ラベルの非ゼロを `pos_w` 倍に重み付けする (sell/buyp/seed 6・anim 20・hire 8・land 30)。
副作用で買いを過大予測しており (open-loop の買い step: WHEAT 199/154、STRAWBERRY 47.3/29.2、TOMATO 19.7/10.0)、推論で非ゼロロジットから log(pos_w) を引いて戻す
`--mkt-debias` を `act_common.mkt_argmax` として実装し 32 戦で A/B した。

| mkt-debias | own | margin | 勝 | 苺 種購入 | 人参 種購入 |
|---|---|---|---|---|---|
| なし | **92.8k** | **−16.9k** | 3 | 71.2 | 6.0 |
| seed | 89.4k | −21.3k | 2 | **54.0** (師 41.5) | 10.5 |
| seed,buyp | 94.0k | −21.3k | 1 | 52.5 | 6.0 |
| all | 59.9k | −73.8k | 0 | — | — |

**機序は当たったが得点は動かない**: `pos_w` が苺の過大購入の原因であることは確認できた (71.2 → 54.0、師 41.5 に接近、植えも 52.0 → 40.8 で師 35.2 に接近)。
しかし**人参は 6.0 → 10.5 にしか戻らず** (師 67.1)、得点も改善しない。人参の欠落には別の原因がある。
`all` の崩壊 (59.9k) は hire/land/anim の pos_w が**意図どおり働いている**ことの裏返しで、pos_w 自体はバグではない。
`--mkt-debias` は既定オフのまま実験用フラグとして残す (`--cow-cap` 等と同じ扱い)。**np_act3.py には未移植**。

**次にやるなら**: 人参の cold start。Majkel の初人参は d9。その時点の盤面で何が引き金になっているか (店の解禁・価格・寿命切れタイル) を反実仮想で潰す。

### ラベル監査 (2026-09-17、Majkel 8 局 = 5,752 step / 57,527 unit-step) — 未修正の欠陥

PASS バグの発覚を受けて全ラベルを生リプレイと突き合わせた (`scratchpad/audit_labels.py`、会話ログ)。

**❌ 市場の数量ラベルが系統的に少なめ (要修正)**
`features.bucket()` は `argmin` で最寄りバケツを選ぶが **同点は必ず小さい方**に落ちる。`MKT_BUCKETS = [0,1,2,3,4,6,8,12,16,24,32,48,64,96]` では
`5→4 7→6 9→8 10→8 13→12 14→12 17→16 18→16 19→16` と 1〜20 のうち 11 通りが下振れ、上振れは 11→12 と 15→16 の 2 通りだけ。片側に偏っている。
Majkel 実測で **売り 1,477 個/局 のうち 55.8 個 (3.8%、名目 $4,840) 、買い 494 個/局 のうち 14.5 個 (小麦 11.4)** を取りこぼす。
実害は名目より小さい: 売り規則 C (h≥20 で shed 全投売り) を入れると小麦以外は同日夕方に回収される。残るのは (a) 規則 C の除外品である**小麦**、(b) 売却が遅れる価格差、
(c) **買いの取りこぼし** — 小麦 11.4 個/局は FEED の原料なので家畜チェーンに効く。なお `play3.py` は既定で規則 C を切っているので、表の 32 戦の数値は回収なしの条件。
**修正方針**: `bucket()` の同点を大きい方に倒す (1 行、`5→6 7→8 10→12` が直り取りこぼしの約半分 279/581 件が消える、market ヘッドの次元は変わらないので checkpoint 互換)。
バケツに 5,7,9,10 を足せば完全に直るが `N_MB` が変わり market ヘッドの重みを引き継げない。
**⚠ コードは 1 行だが shard は作り直しになる**: shard が持つ `mkt` はバケツ**添字**で、元の数量は復元できない (idx 4 ← {4,5}、idx 6 ← {8,9,10}、idx 8 ← {15〜20})。
`fix_pass_labels.py` のような後から直す道はなく、**Majkel 854 局のリプレイを全部落とし直す必要がある** (JSON は `fetch_episodes.py` が捨てている。実測 ~10 秒/局 = 約 2.4 時間)。
**再取得するなら同時に `extract.py` へ生の市場数量を足しておく** (`mktq` 等)。そうすればバケツ定義を変えるたびに再ダウンロードせずに済む。

**△ `PLACE_COW` が 19 件 (57,527 中) 自分の mask で非合法** — `legal_ops` の牧場条件がエンジンより厳しい箇所がある。実害は小さいが `legal_ops` の該当分岐を要確認。

**△ 目的地ラベルの 239/53,772 (0.44%) が `legal_ops_at` で非合法** — 学習は `bc_loss3` の `target_mask` で強制的に通すが、推論では `legal_option_mask` に塞がれる。
「教えても出せない option」を 0.44% 混ぜている状態。`legal_ops_at` と実際の到着時合法性のズレを潰すか、ラベル側で落とすか。

**✅ 問題なし**: `unit_op` の取りこぼし 0 件 (エンジンの op 語彙 `kagsim.cpp` の `OPI` と `features.OPS` は完全一致)、市場注文の種類の取りこぼし 0 件、
上限クリップ 0 件、ユニットの量バケツ (PICKUP/PLACE) の丸め誤差 6 件/57,527 (無視可)。

### 次にすること (2026-09-17 夜、優先順)

0. **夜間 Kaggle bc22 の結果**: `kaggle kernels output mmn0222/kaggriculture-bc22-policy3 -p tmp/kaggle_out_bc22 --force`。
   Majkel 854 局・初期値なし 4 epoch lr 5e-4 (bc20 レシピ) の各 epoch + bc15/bc18/bc20_ep3 を**対 v41 64 戦**で評価 (`rl/kaggle16/`)。
   32 戦 (SE ±3.1k) では bc15 95.4k / bc18 92.8k / bc20 92.1k の順位がつかないため、64 戦で決着させる。
0b. **ハイブリッド (テープ + Transformer 売り層)**: 土台は `agents/e066/` (v46)。**追加のみ**の売り層として `main.py` の `_sell` の兄弟に置く。
   実装前に `.opencode/knowledge/refs/chassis_market_anatomy.md` を読むこと (全面差し替えは在庫の二重売り、スロット 0 は契約、終局 712-717 は 9 本 SELL 固定、10 本上限)。
   推論は numpy (`export_agent.py` → `np_policy3.py` + `np_act3.py`)。market ヘッドだけ使えばよい。


主軸は **Majkel との差 (18k) を埋める**こと。PASS は解決済みで律速ではないと判明したので、残る差の実体を追う。

1. **市場バケツの同点丸めを直す** (上の監査): `bucket()` の同点を大きい方へ。売り +28 個/局・買い小麦 +6 個/局 相当。
   コードは 1 行だが **shard の作り直し = Majkel 854 局の再ダウンロード 約 2.4 時間**が要る (バケツ添字から元の数量は復元不能)。着手時は `extract.py` に生数量も保存すること。
2. **人参の cold start** (上の診断の続き、本命): 閉ループで人参を 5.8 株しか植えず (師 68.8)、d8-19 の実り 0.58 vs 0.82 の差になっている。
   給水は正常・両ヘッドの open-loop も正常・`pos_w` デバイアスでも戻らない。残るのは「d9 に Majkel が最初の人参を植える引き金」の特定。
3. ~~bc19 (初期値のデータ増量)~~ **済・棄却**: 初期値を 634→2,000 局にすると bc19_ep3 88.0k で bc18 (92.8k)・bc20 (92.1k) を下回る。
   bc20 (初期値なし 92.1k) と併せて、**混合データの事前学習は無価値どころか増やすと有害**。混合データが教える待機癖 (生 PASS 547/局) を fine-tune が剥がしきれないためと推定。教師は 1 位のみで確定。
4. ~~Majkel 単独データのみ (初期値なし) の学習~~ **済 (bc20)**: 32 戦 92.1k/−21.4k で bc18 (92.8k) と SE 内 → **混合初期値 bc9k の寄与は測定限界以下**。
   **継続 (bc21) は不発**: ep2→ep3 の +30k は OneCycle の annealing が終わっただけで未収束ではなかった。2 epoch 足すと val は改善・閉ループは 92.1k→87.8k と悪化 (bc16 と同型)。
5. データ増量の継続: Majkel は 09-17 に +141 局 (854)。`fetch_episodes.py` で日次追加。
6. やらないこと: epoch 追加 (**初期値の有無によらず 4 epoch が上限**。bc16 = fine-tune 継続で崩壊、bc21 = スクラッチ継続でも同型。いずれも val が改善しながら閉ループが落ちる)、PPO (再凍結、規模が前提)、農場側の推論強制・班分け・待機ラベル・自己軌跡・温度サンプリング・容量 d256、
   **PASS を減らす目的の細工** (bc18 で「PASS 486→179 でも得点不変」と実証済み)。

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
