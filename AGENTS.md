# Kaggriculture (Kaggle コンペ) — エージェント用ガイド

2人対戦の農業シミュレーション (720 ターン = 30 日)。終了時の所持金が多い方が勝ち。締切 9/30、最終評価は締切後 ~2 週間の
Bradley-Terry。**締切時点で「エラーの出ない最強 2 提出」がスロットにあることが全て** (有効なのは最新 2 提出、選べない)。
時系列の経緯は `.opencode/knowledge/history.md`、実験の一次記録は `.opencode/knowledge/experiments.md`。

## 方針と現在地 (2026-09-24)

**知見はここに書かない。方針別の `tracks/` にある。** 索引は `.opencode/knowledge/README.md`。

| 方針 | 状態 | 現在地 | 資料 |
|---|---|---|---|
| **公開シャシー + 自前の市場層** | **本線** | 提出中 **E079** (09-24 05:50 UTC、ref 56513094) と **E078** (04:22 UTC、ref 56511519)。E076 (03:35 UTC) は押し出し済み。09-24 時点 519 位 (2489.1)、50 位 = 2787.5 | `tracks/chassis.md` |
| 学習方策 (Transformer/Policy3) | 停滞 | 別セッション (HANDOFF.md) | `tracks/transformer.md` + `rl/README.md` |
| GA によるテープ進化 | 棄却 | 素のテープに 245 点負けた | `tracks/ga.md` |
| 自作プランナー / ルート外科 | 棄却 | 固定ルートは農場側の介入を受け付けない | `tracks/planner.md` |

**09-24 の核心**: 実戦の相手の **67% は農場行動が我々と完全一致する鏡像** (同じ herd 系・同じルート表)。
牛乳・羊毛は両者の供給で飽和しており増産は損。鏡像には (1) **需要窓先回し** (E076) と (2) **鏡像と違うルート** (E078/E079、
ガチョウの多い r112/r120/r110/r103) で勝つ。実戦リプレイ 458 局の席差し替え (`tests/swap_eval.py`) で E074 → E079 は勝ち 279 → 334。
公開 herd2700 本体に E079 は 29W3L、more-wheat に 27W5L。

## 次にやること (2026-09-24)

1. **E078 / E079 の実戦結果を見る** (`tests/rating_traj.py 56511519 56513094`、同じ時間帯どうしは `tests/fit_strength.py --since`)。
2. **ルート掃引の続き** — 2 回目の掃引 (E078 を自分側) は 12,000/26,880 局で中断 (`tmp/sweep2_tune.json`)。残りを回して候補を検証する。
   **CPU は `nice -n 19`・4 並列まで** (11 並列でユーザーの Mac が操作不能になった)。
3. **公開エージェント総当たりの残り** (`agents/pub_*` = 公開 NB の手元コピー、git 管理外。`tmp/panel_e079.sh`)。負ける相手がいればその層を読む。
4. **帯 (30〜220 位) のリプレイ 466 局** (`tmp/band0924`、未 slim) で、帯チームの席に E079 を差し替えて彼らとの差を測る。
5. 先回し勢 (鏡像の 25%、我々の売りの直前に売る相手) への適応は、反応する先回し相手のモデルがないと測れない (x078 の AFR は不採用)。

## リポジトリ構成

```
agents/e079, e078, e076       現行 (e076 = e074 + 市場先回し、e078 = + 対鏡像ルート 21 ペア、e079 = + 4 ペア + トマト閾値 7,500)
agents/x076, x078             開発用 (環境変数で変種を切替)。agents/pub_* は公開 NB の手元コピー (git 管理外)
agents/e072, e073, e074       前世代 (e074 = e072 + YARN ルート表パッチ)。main.py は単体完結 (base.py なし)。e072 は鏡像の代理
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
