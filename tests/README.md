# tests/ — 現役ツール (用途別)

**評価 (kagsim、速い)**
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
- `rating_traj.py <submission id>` — 非公式 API でレーティング推移・相手帯別成績
- `fetch_top.py --lb LB.csv --top N --per M --out DIR` — LB 上位のベスト提出のリプレイ取得
- `team_profile.py DIR TEAM [--brief]` — チームの各試合 (署名・購入・品目別売上) と試合間の農場行動一致 (固定テープか適応型か)
- `team_blueprint.py DIR TEAM` — 日別平均スケジュール (雇用・植栽・収穫・水・土地・現金) とユニット行動の内訳

**テープ表の拡張 (E060 手順)**
- `route_choice.py --agent agents/e060/main.py --tapes .opencode/data/<tapes>.json --offset 13` — 64 ペア × seeds で既定プラン vs 新家系プランを v41 相手に比較し、ペア別の採用表を出す

**GA (`ga/`)** — kagsim でテープを進化させる基盤 (pool/fitness/evolve/wrapped は 0909 13 本前提、fit58/evolve58 は E058 の層付き評価)。層の上では伸びず現在は未使用
