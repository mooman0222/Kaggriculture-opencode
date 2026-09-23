# Kaggriculture Transformer軸 セッション引き継ぎ (2026-09-22 15:40 UTC)

## 再開時の最初の3コマンド

```bash
# 1) 現状のスコア確認 (エグゼキュータ+スクリプト市場、~3分)
cd /tmp/t10 && OMP_NUM_THREADS=1 python3 eval_exec.py /tmp/t10/t2.npz
# 2) GPU空き確認 (kd2 は kill 済みのはず)
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
# 3) コードの安定コピー (tmp が消えていたら復旧)
ls /home/mooman_che/Kaggriculture-opencode/tmp/t10/
```

作業の続きは「5. 次の作業」の 1 → 2 → 3 の順で。

## 0. 目的と現在地

- 目的: **Transformer軸で上位10位** (締切 9/30、評価は締切時の最新2提出)。データセット以外はリセット済み (`git checkout main` 済み、`zero/` は `/tmp/backup_20260921/` に退避)。
- 使用データ: `/tmp/work/diverse/*.npz` (7,884件、上位23提出の全手データ) と `/tmp/work/val` (2,000件)。
- **現在のベスト: 群管理エグゼキュータ + スクリプト市場 ≈ 15.4k / 15.9k money** (kagsim seed 500000/500001、sold 206-207、revenue ~35.5k)。
  - 公開教師 v46 (同一 seed) は 130〜190k。学習方策単体 (model モード) は 1〜4k。
  - LB上位は rating 3148、公開シャシーは 2700 前後と推定。

## 1. 最重要の一次知見 (エンジン規則) — 再発見に時間を使わないこと

1. **Atomic PLANT (最大の落とし穴)**: 1ターン内で同一作物の PLANT 要求数が所持種数を超えると、**その作物の PLANT が全て無効化**される (`third_party/kaggriculture-cppsim/sim/sim.hpp` の `apply_unit_actions`、実Python env も同じ)。
   - 群で並列に植えると全滅する。→ 1ターン1作物1ユニットに制限すること (`agent_np.py` の `self._planted`)。
2. **HARVEST は成熟が必須**: `yield_units > 0` だけでは不可。`day - planted_day >= first_yield_day` が必要。未成熟HARVESTは no-op。
3. **植えた当日に水やりしないと翌日 weed 化** (`consecutive_unwatered >= 2`)。ongoing (TOMATO/STRAWBERRY) も同様。
4. **市場**: 1ターンの有効注文は10 (kagsimは16)。HIRE/BUY_LAND は `[["HIRE"]]` の list 形式必須 (文字列 `"HIRE"` は無視される)。
5. **NPZ特徴量の完全復元**: `encode.py` に実装済み・全チャネル検証済み (tiles/units/items/glob)。items col4 と tile c15 は未解読で 0 埋め (学習側でも 0 にする)。glob は g33=ego hires_today/12, g34=opp を追加した版が現行。
6. ラベル対応 (NPZ): `fop` = 生op+1 (0=inactive), `farg` = item+1, `dop` = タスクop+1 (9=WATER/移動先で行うop), `dest` = 目標タイル y*10+x+1, `darg` = item (0..12, オフセット無し)。**darg だけ +1 しない**のが要注意。
7. NPZ step t の特徴量 = リプレイ step t の observation、ラベル = step t+1 の action。

## 2. 何が失敗したか (再実験の無駄を避けるため)

| 試行 | 結果 |
|---|---|
| 上位23提出の混合BC (m1/s2, d=256〜512) | fop 33%, dop 63%, dest 45% → 閉ループ ~1k money |
| style条件付き + タスク重み + 市場クラス重み (s2) | dop 63%, 市場の過剰買い/無買いが両極端 |
| 公開教師 v46 の蒸留 (t1/t2, ktask.py, 200〜350ゲーム) | dop 67%, dest 57% だが **単一ゲームに120epoch過学習しても fop 62%, dop 69% が上限** → 教師は観測・行動履歴から復元不能な内部状態を持つ (決定論的・乱数非依存を確認) |
| 行動履歴入力 (act_enc, t4) | 精度改善なし (上限は同じ) |
| PPO (kick.py) | logprob 再計算が文脈長境界で一致しない問題が未解決 → pg 爆発。wpg=0 の純BC/DAgger でも money 0〜5 で改善せず |
| DAgger (kd2, 42iter) | 失敗。現在 PID 79228 で稼働中だが**効果なし、killしてよい** |
| 群管理エグゼキュータ (claim + Atomic PLANT 対応) | **15〜21k まで回復。現行の主力** |

## 3. 成果物の場所

- **コード**: `/tmp/t10/`
  - `encode.py` … 観測→NPZ特徴量 (検証済み、**提出物にインライン化される**)
  - `policy_np.py` … numpy推論 (torch不要、提出物にインライン化)
  - `agent_np.py` … エージェント本体 (タスク永続化 + 群エグゼキュータ `fallback_task` + 市場デコード)
  - `model3.py` … BCV3 (style embedding + act_enc + value head)
  - `ktask.py` … 公開教師のゲーム生成 → タスクラベル抽出 → 蒸留学習
  - `kick.py` … 群エグゼキュータ入りPPO/DAgger (未完成・不安定)
  - `eval_exec.py` … エグゼキュータ+スクリプト市場の評価 (`market()` が市場ロジック)
  - `build_sub.py` … 提出物生成 (main.py + model.npz fp16)
  - `export_np.py` … .pt → .npz 変換 (verify付き。**入力スケール tiles/16, units/12 を合わせること**)
  - `rl_train.py`, `dagger.py`, `distill_bc.py`, `dagger_bc.py` … 各試行
- **チェックポイント**: `/tmp/t10/s2.pt` (上位データBC), `/tmp/t10/t2.pt` (教師蒸留タスク方策・現行), `kd2.pt` (DAgger), `t4.pt` (行動履歴版)
- **リプレイ**: `/tmp/replay_test/episode-110360748-replay.json` ほか (ラベル対応表の検証に使用)
- **提出物ビルド**: `python3 build_sub.py /tmp/t10/sub /tmp/t10/t2.npz 32` → `sub/main.py` + `sub/model.npz` (60MB)
  - kaggle-environments 実環境で DONE を確認済み (1ゲーム ~298s)。tar にして `kaggle competitions submit` (毎回ユーザー確認)。
- **安定コピー (推奨)**: 上記コード一式 + `t2.pt` + `t2.npz` + `mkt_weights.npy` + ビルド済み `sub/` を
  `Kaggriculture-opencode/tmp/t10/` にコピー済み (302MB)。/tmp が消えてもここから復旧可能。

## 4. 評価コマンド

```bash
# エグゼキュータ+スクリプト市場 (現行主力)
cd /tmp/t10 && OMP_NUM_THREADS=1 python3 eval_exec.py /tmp/t10/t2.npz
# 個別 (executor/model モード × seed)
cd /tmp/t10 && OMP_NUM_THREADS=1 python3 -c "...(eval_exec.run を参照)"
# 提出物の実環境確認
cd /tmp/t10 && OMP_NUM_THREADS=1 python3 -c "import importlib.util; spec=...; sub=...; from kaggle_environments import make; env=make('kaggriculture'); env.run([sub.agent, sub.agent])"
# 教師のベースライン
cd /home/mooman_che/Kaggriculture-opencode/third_party/kaggriculture-cppsim && OMP_NUM_THREADS=1 python3 -c "
import sys; sys.path.insert(0,'.'); sys.path.insert(0,'/tmp/t10')
import kagsim; from dagger import load_agent
tf,_=load_agent('/home/mooman_che/Kaggriculture-opencode/third_party/public_agents/v46/main.py')
g=kagsim.Game(500000)
for t in range(719): g.step(tf(g.observe(0)), tf(g.observe(1)))
print(g.reward(0))"
```

## 5. 次の作業 (優先順)

1. **エグゼキュータ強化** (最も費用対効果が高い):
   - 土地購入 (4象限化) は `eval_exec.py` の市場に追加済み → 効果測定中 (15.4k/20.7k/15.9k と不安定)。
   - 家畜ループ (pasture建設→PICKUP→PLACE→FEED→COLLECT) は未完成。MILK/WOOL/EGGは高価。
   - 作物ミックス: MELON優先が最良 (20.9k)、WHEAT優先は16.2k。STRAWBERRY (ongoing) の活用未検証。
   - 収穫物の売りタイミング (`eval_exec.market` の sell 条件: 価格≥0.95×base or 在庫≥12)。値崩れ防止に分割売りを検討。
2. **Transformer市場ヘッドの統合**: エグゼキュータの状態で教師の市場行動をBC (warm-start DAgger)。`kick.py` の wpg=0 + 群エグゼキュータ + 市場ヘッドのみ学習、が有望。現状 `model` モード (モデル市場) は 0〜5k と悪い。
3. **提出**: 締切前に最低1回、動くものを提出 (要ユーザー確認)。現時点では「エグゼキュータ+スクリプト市場」を main.py 化するのが安全。
4. 余裕があれば: PPOのlogprob不一致修正 (文脈長境界)、act_enc の寄与再評価。

## 6. 落とし穴メモ (このセッションで何度も踏んだ)

- **パッチが silent に失敗して `Agent.__call__` の except が空行動を返し「何もしない」症状**を繰り返した。
  → `agent_np.py` を触ったら必ず 1 ゲーム回し、非PASS行動数と money を確認。`BC_DEBUG=1` で traceback 表示。
- `_remember` / `hist_acts` / `ITEM_BY_IDX_INV` / `SHED_ADJ_TILES` / `CROP_FIRST_YIELD` など、リファクタで消えやすい補助定義がある。ロード確認だけでなく**実ゲームでの動作確認**を必ず行う。
- 群エグゼキュータは「同じターゲットに全員が吸われる」問題がある → `self._claimed` でターン内claim必須。
- 評価は seed 固定 (500000/500001) で比較。kagsim と実env は規則一致 (Atomic PLANT 含む)。
- 学習系の検証は必ず **held-out ゲーム**で。単一ゲーム過学習の精度を汎化性能と誤認しない (このセッションで一度誤認した)。
