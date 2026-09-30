# tests/ — 現役ツール (用途別)

**評価 (kagsim、速い)**
- `chassis_probe.py --specs SPEC.json --replays GLOB --out RESULT.json` — 候補の paired 席差し替え。**CPU 負荷対策: 1 ワーカー・試合ごと 2 秒待機** (`--jobs 1 --pause-per-game 2`)、他の評価と同時起動しない。`--resume` で保存済み試合を再利用、`--baseline-cache RESULT.json` で変更していない対照の再計算を省略。
- `goose_headroom.py --n 8 --out RESULT.json [--goose-only]` — 行動を変えず、ガチョウ上の PASS で補える未給餌・未 CARE を日末と照合する。
- `short_hold_headroom.py --n 8 --out RESULT.json` — 需要更新をまたぐ 1 手の売却延期の診断。相手注文は後知恵なので、結果は採用根拠ではなく追加検証の判断用。
- `paced_duel.py --candidate PATH --opponent PATH --n 4 --out RESULT.json` — ショップ列固定・両席の直接対戦。1 プロセス、試合間 2 秒待機、計算中も CPU 時間に応じて休止する。
- `test_short_hold.py` — 売却延期試作の注文維持・翌手放出・類似農場除外の単体テスト (対戦計算なし)。
- `kag_eval.py` — 直接対決 (`--vs`, 両席) / 実戦席差し替え (`--replays --team --base`) / 固定ショップ (`--shops`)
- `live_eval.sh AGENT` — 定型 3 点: 対 v41 16 戦、対 E058 16 戦、苦手世界 8 × 両席 (own bank)
- `live_debug.py AGENT --seed N` — 1 試合の日別サマリ (現金・雇用・畑・棚・相手・注文拒否)。`LE_DEBUG=1` で例外を表面化、`LE_ACTS=1` で行動内訳
- `prod_compare.py A B --seeds ... --from D` — 同シードで 2 エージェントの日別収穫ユニット (品目別) と労働内訳を比較
- `h2h.py A B` — 実エンジン (kaggle_environments) での直接対決 (遅い。提出前確認向け)
- `tape_eval.py`, `eval_replays.py` — 記録行動の再生評価 (旧世代の席差し替え)

**実戦・相手の分析**
- `fetch_battles.py --dir DIR --team MMN0222 --lb LB.csv` — 24 手署名・家畜構成別の勝敗 (battles.json を出力)
- `lineage_census.py --dir DIR` — 開幕 72 手ハッシュ別の勝敗
- `match_versions.py --replays GLOB --me agents/e060` — 相手の版を全手再現で特定 (候補 = third_party/public_agents/*)
- `identify_seats.py GLOB OUT.json CAND...` — 任意の試合の各席を候補が何手目まで再現できるか (我々の試合でなくてよい。719 = 同一)
- `panel_eval.py CANDS OPPS SEED0 N OUT.json` — 候補を反応する相手と同じ seed・席で並べ、先頭候補とのペア差と勝数
- `rating_traj.py <submission id>` — 非公式 API でレーティング推移・相手帯別成績
- `fetch_top.py --lb LB.csv --top N --per M --out DIR` — LB 上位のベスト提出のリプレイ取得
- `team_profile.py DIR TEAM [--brief]` — チームの各試合 (署名・購入・品目別売上) と試合間の農場行動一致 (固定テープか適応型か)
- `team_blueprint.py DIR TEAM` — 日別平均スケジュール (雇用・植栽・収穫・水・土地・現金) とユニット行動の内訳

**テープ表の拡張 (E060 手順)**
- `route_choice.py --agent agents/e060/main.py --tapes .opencode/data/<tapes>.json --offset 13` — 64 ペア × seeds で既定プラン vs 新家系プランを v41 相手に比較し、ペア別の採用表を出す

**GA (`ga/`)** — kagsim でテープを進化させる基盤 (pool/fitness/evolve/wrapped は 0909 13 本前提、fit58/evolve58 は E058 の層付き評価)。層の上では伸びず現在は未使用

**09-30 の複製突き・上位分析 (`tests/tr/`、リポジトリ直下から実行)**
- `plan_ident.py TEAM GLOB CANDS...` / `plan_ident_opp.py TEAMS GLOB` — 記録席 (またはその相手) の農場行動が候補と何手目まで一致するか (シャシー系か自作系かの判定)
- `diff_cats.py TEAM GLOB` / `mkt_cmp.py` / `mkt_diff.py` — 同じ状態で E090 が出す注文との差を種類別に数える
- `ablate.py TEAM GLOB` — 記録から肥料・小麦の往復だけを抜いて再生し、その価値を測る (農場は記録どおり)
- `vs_team_tape.py TEAM GLOB CANDS...` — 指定チームの記録テープを相手に、候補を対戦相手の席へ置く (上の帯の代理)
- `pair_panel.py CANDS OPPS PAIRS SEED0 N OUT` — 先頭 2 店を固定した世界で候補を比べる (`E090PATCH` でパッチ 12 ペア)
- `precheck.py AGENT SEED` / `realduel.py A B SEED` — 実エンジンの自己対戦・直接対戦 (状態・報酬・最大手番時間)
- `tape_probe*.py`・`build_lib.py`・`loo*.py`・`fixed_tape.py` — 上位テープ流用の検証 (棄却済み、experiments.md 09-30)
