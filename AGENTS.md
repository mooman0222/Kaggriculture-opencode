# Kaggriculture (Kaggle コンペ) — エージェント用ガイド

2人対戦の農業シミュレーション (720 ターン = 30 日)。終了時の所持金が多い方が勝ち。締切 9/30、最終評価は締切後 ~2 週間の
Bradley-Terry。**締切時点で「エラーの出ない最強 2 提出」がスロットにあることが全て** (有効なのは最新 2 提出、選べない)。
時系列の経緯は `.opencode/knowledge/history.md`、実験の一次記録は `.opencode/knowledge/experiments.md`。

## 方針と現在地 (2026-09-23)

**知見はここに書かない。方針別の `tracks/` にある。** 索引は `.opencode/knowledge/README.md`。

| 方針 | 状態 | 現在地 | 資料 |
|---|---|---|---|
| **公開シャシー + 自前の市場層** | **本線** | 提出中 **E074** (09-23, herd2700 + YARN ルート修正) / E073。実測: e072 2562.3 / e073 2545.8 (team rank 453)、e074 評価中。**E074 は本路線初の held-out で生き残った改善** (7 ペア、+0.5k〜+4.1k/該当ペア) | `tracks/chassis.md` |
| 学習方策 (Transformer/Policy3) | 停滞 | 対 v41 64 戦 margin −15.4k。**提出系との差 20k、埋める道が立っていない**。市場層 BC も閉ループで崩壊 (E074 期) | `tracks/transformer.md` + `rl/README.md` |
| GA によるテープ進化 | 棄却 | 素のテープに 245 点負けた (E055 2698.8 vs E057 2453.9) | `tracks/ga.md` |
| 自作プランナー / ルート外科 | 棄却 | 固定ルートは農場側の介入を一切受け付けない | `tracks/planner.md` |

**最重要の方法論 (09-23 発見)**: ショップ列 = f(seed, day, **両者の空きタイル累計**)。1 手の農場行動変更で世界が丸ごと入れ替わる
(±30-50k)。**農場行動を触る実験は必ず pinned shops (`kagsim.Game(seed,720,[8 shops])`) で行う**。詳細 `refs/world_draw.md`。

## 次にやること (2026-09-23)

1. **E074 の実戦結果を見る** (09-23 提出、1 日後)。ルート修正は農場・市場を触らないため転移する可能性が高い。
2. **非 YARN 49 ペアも同じ全 40 ルート掃引** — YARN では 7/15 ペアに実改善があった (V92 上書きの副作用)。
   非 YARN は未掃引。手順: 1 世界で全ルートをスクリーニング → 上位 2 を held-out 5 世界 × 4 相手 × 両席で確認。
3. **世界の引き直しを利用した検証の徹底** — 相手が変わると世界も変わる。pinned + 複数相手 + 両席の 3 点セット。
4. **帯 (2700-2800) の変種の出所を探す** — 8 チームが同一ユニット行動 (1.00 一致)。公開 NB 約 30 種と不一致。新着 NB を毎日監視。
5. **エンジン規則の残りの穴を探す** — `refs/engine_facts.md` × `sim/sim.hpp`。

## リポジトリ構成

```
agents/e072, e073, e074       現行提出 (e074 = e072 + YARN ルート表パッチ)。main.py は単体完結 (base.py なし)
agents/e069                  夜明けガードをシャシー非依存にした版 (旧 v46 系。新土台では参考)
agents/e074.tar.gz           提出物 (tar czf ... main.py LICENSE.txt NOTICE.txt)
agents/sr0909_base, sr0909_live  素の 0909 と E055 (GA ツールの参照・対戦相手)
agents/legacy/live_e         保留中の毎手プランナー。他の退役物は git 履歴 (agents/README.md 参照)
third_party/public_agents/   公開 NB の実体 (v39〜v48、v57)。herd2700 系の土台は v39/v50 系
third_party/kaggriculture-cppsim  kagsim (bit-exact C++ エンジン、1 試合 0.1〜1.5 s)
tests/                       現役ツール (一覧と用途は tests/README.md)
.opencode/knowledge/         tracks (方針別) / refs (路線横断の参照) / experiments・history (追記専用ログ)。索引は README.md
.opencode/data/              0913 テープ (shop_router_0913_tapes.json)、E060 の世界別選択結果 (e060_route_choice.json)
tmp/                         git 管理外の作業領域 (リプレイ、プール、公開 NB の生データ)。消えても再取得できる
```

## 環境とコマンド

- venv: `.venv/bin/python` (kaggle-environments + kagsim)。Kaggle CLI: `.venv/bin/kaggle` (トークン `~/.kaggle/access_token`)。
- **inline の環境変数を複数渡すときは bash を使う** (`bash -c "A=1 B=2 .venv/bin/python ..."`)。zsh は `$cfg` を単語分割しない。
- 直接対決 (両席、kagsim): `.venv/bin/python tests/kag_eval.py agents/X/main.py --vs third_party/public_agents/v41/main.py --games 48 --seed0 5000`
- 実戦席差し替え: `.venv/bin/python tests/kag_eval.py agents/X/main.py --replays 'tmp/<dir>/episode-*.json' --team MMN0222 --base agents/e068/main.py`
- 定型 3 点 (対 v41 / 対現行 / 固定ワールド): `bash tests/live_eval.sh agents/X/main.py`
- 1 試合の日別デバッグ: `LE_DEBUG=1 .venv/bin/python tests/live_debug.py agents/X/main.py --seed N [--shops A,B,...]`
- 実戦リプレイ取得: `kaggle competitions episodes <ref>` → `kaggle competitions replay <id> -p tmp/<dir>`
- 提出: `.venv/bin/kaggle competitions submit kaggriculture -f agents/X.tar.gz -m "..."` (1 日 5 回。**毎回ユーザー確認**。メッセージに `$` を使わない)

## 定型手順

1. **実戦チェック** (提出後 1 日): リプレイ 80 戦を取得 → `tests/fetch_battles.py --dir DIR --team MMN0222 --lb LB.csv` (署名別勝敗)
   → `tests/lineage_census.py` (開幕ハッシュ別) → `tests/match_versions.py --replays 'DIR/episode-*.json' --me agents/e068` (相手の版を全手再現で特定。
   同系は開幕ハッシュでは区別できない) → 大負け相手は `tests/live_debug.py`/`tests/prod_compare.py` で分解。
   レーティング推移: `tests/rating_traj.py <ref>`。上位のリプレイ: `tests/fetch_top.py --lb LB.csv --top 4 --per 10 --out DIR` → `tests/team_profile.py DIR TEAM` / `tests/team_blueprint.py DIR TEAM`。
2. **新 NB / 新テープの取り込み**: `kaggle kernels pull <ref> -p tmp/nb/<ref> -m` → ipynb のコードセルから main.py (writefile か base64/base85 blob) を展開
   → `third_party/public_agents/<name>/main.py` に置く → `kag_eval --vs` で現提出と比較。**手順の全体は `refs/chassis_upgrade_playbook.md`** (sha256 照合を含む。人気ノートが既存版の再掲のことがある)。
3. **提出前チェック (必須)**: (a) 最後の callable が `agent` (`.venv/bin/python -c "src=open('agents/X/main.py').read(); env={}; exec(compile(src,'x','exec'),env); print([k for k,v in env.items() if callable(v)][-1])"`)
   (b) 実エンジン self-match で両者 DONE・対称 (`tests/live_eval.sh` 末尾の要領、または kaggle_environments `make().run([agent, agent])`)
   (c) tar を展開して同じ確認 (d) 最大手番 < 1 s。
4. **評価の定石**: 相手は決定的 (v41 / E058 / 記録テープ)。クローン戦は両席ペア margin の正シード率で見る。実戦席差し替えは相手がテープなので
   「相手の適応」が抜ける (適応型には楽観)。注文ベースの売上推定は不執行スパムで過大 → kagsim telemetry の `sell_revenue`/`sold_units` を使う。

## ナレッジの扱い

- 知見は `tracks/` (方針別) か `refs/` (路線横断)、経緯は `experiments.md`。**同じことを二箇所に書かない**。
  09-16〜17 に AGENTS.md と rl/README.md へ二重に書き、後日の訂正が片方に反映されず矛盾した。
- 知見は「その時点の証拠」。メタは 2〜3 日で変わる。条件が変わったら再実験してよく、矛盾したら実験を優先して**古い記述を消す**。
- 新しい仮説は安いうちに試す (kag_eval 16 戦 = 10 秒)。禁止系の知見も再評価の出発点として扱う。
- **数字を見出しにする前に SE と t を出す。** 平均値だけを見た結論は何度も覆っている。
- **既存系に層を足す話は、実装の前に伸びしろを測る** (`tracks/transformer.md` 6 節の形。1 時間で丸一日を節約した)。
