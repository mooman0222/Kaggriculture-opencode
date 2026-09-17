# Kaggriculture (Kaggle コンペ) — エージェント用ガイド

2人対戦の農業シミュレーション (720 ターン = 30 日)。終了時の所持金が多い方が勝ち。締切 9/30、最終評価は締切後 ~2 週間の
Bradley-Terry。**締切時点で「エラーの出ない最強 2 提出」がスロットにあることが全て** (有効なのは最新 2 提出、選べない)。
時系列の経緯は `.opencode/knowledge/history.md`、実験の一次記録は `.opencode/knowledge/experiments.md`。

## 明日やること (2026-09-17 夜に中断)

1. **夜間 Kaggle の結果を取る**: `kaggle kernels output mmn0222/kaggriculture-bc22-policy3 -p tmp/kaggle_out_bc22 --force` →
   `eval.txt` に bc22 (Majkel 854 局・初期値なし 4 epoch) の各 epoch と bc15/bc18/bc20_ep3 の**対 v41 64 戦**。
   ねらいは (a) 09-17 の結論がすべて 32 戦 SE ±3.1k に埋もれている順位を 64 戦で確定させる、(b) 別ハードで bc20 (92.1k) が再現するか。
2. **ハイブリッド実装**: v46 シャシー (`agents/e066/`) の上に Transformer の売りヘッドを**追加のみ**で載せる。
   制約は `.opencode/knowledge/refs/chassis_market_anatomy.md` を**必ず先に読む** (全面差し替えは在庫の二重売りを起こす、スロット 0 は契約、終局は 9 本 SELL 固定)。
   形は `agents/e066/main.py` の `_sell` と同じ兄弟層。重みは numpy 版 (`rl/export_agent.py` → `np_policy3.py`) で載せる。torch は使えない。
3. **E066 の実戦結果を見る** (09-17 提出、ref は `kaggle competitions submissions kaggriculture`)。
4. 保留: 市場バケツの丸め修正 (`features.bucket()` の同点が下振れ。shard 作り直し = Majkel 854 局の再取得 2.4 時間が前提、`rl/README.md`「ラベル監査」節)。

## 現在地 (2026-09-17)

- 提出中: **E066** (`agents/e066/`, 09-17 提出、**v46 土台** + E065 のライブ層) と E065 (`agents/e065/`, ref 56243664, v43 土台)。
  **v43 は 3 版古く、E065 は素の v45 に 1W15L・素の v46 に 5W11L で負けていた**。E066 は E065 に 14W2L +1,386。
  ただし **E066 vs 素の v46 は 19W13L +102** — ライブ層の価値は v46 が同じ着想 (same-turn sale race / sale timing) を取り込んで消えた。
  **今後の差別化は Transformer 売り層のみ**。公開シャシーの新版確認は `kaggle kernels list --competition kaggriculture --sort-by dateRun --page-size 40`、
  取り出し手順は `third_party/public_agents/NOTICE.md` 末尾。
- 学習側の現在地と打ち止めになった手は `rl/README.md`「Policy3」節。最良は bc15_ep3 95.4k / bc18_ep3 −16.9k (対 v41 32 戦、すべて SE 内)。

## 旧・現在地 (2026-09-15)

- 提出中: **E065** (`agents/e065/`, ref 56243664、09-15 提出、v43 土台 + SELL 先頭化 + 適応リード) と **E062** (`agents/e062/`, ref 56243229、E061 + SELL 先頭化)。E061 (2814) は押し出し。
- 構成: 公開 NB「ahmedberatozer v42」の反応層 (base.py、そのまま) + 自前ライブ層 (main.py、E058 と同一)。
  v42 は step144 に世界別生産ライブラリを選択する (YARN 世界は旧版維持)。E060 の 0913 表は未移植。
- 実戦診断 (09-15、E058 122戦・E060 116戦): E060 が勝率 70% で E058 (66%) を上回る。敗戦の最大塊は v41 同系ミラー。
  新市場変種 7291e57ab5 (農場一致・売り層違い) が出現中。市場層への介入は全て中立〜大敗で打ち止め (experiments.md の RL-MKT/E061前史)。
- 1 位 Majkel1337 (3233)、10 位 Catalyst (2966)。上位は全て適応型。
- 1 位 Majkel1337 (3233) と、それを 23勝2敗で破った私設 M&M&P&Q は毎手適応型。記録行動の再生は崩壊し (−93k)、模倣不能。
  ローカルに適応型の代理はない → 上位帯の勝率は実戦データでしか測れない。
- 構成メモ (旧): E058/E060 = v41 反応層 + テープ差替 (actions.json) + 自前ライブ層。E060 は 0909 13 本 + 0913 29 本の 42 本表。
  E061 には未移植 (v42 の生産ライブラリと競合するため。seed5010 では E060 表が v42 に勝つ世界あり → 将来の統合余地)。
- 相手分布: 実戦の 75% は ahmedberatozer 系 (0909 と同一テープ + 反応層、step0 小麦往復開幕)。E061 は v42 素に 18W6L +1.2k。
- 09-15 敗因分析: E061 は対 2800+ 帯 7W17L で 2810 に均衡。同農場 34 戦の市場層入替 (kagsim 完全再現) で勝敗が反転 → 差は売り層。
  **E062** (`agents/e062/`, E061 + SELL 先頭スロット化) = ミラー +771・対 v42 +1.7k・席差し替え悪化なし。**提出済み ref 56243229**。位相保留は棄却。
  **E063** = E062 + v43 土台 (倉庫溢れ回収、対 E062 +238)。**E065** = E063 + 適応リード (相手の先行を検知したら売り先行 6→12 手): 対固定リード −1,410 → −14、席差し替え +299/+1,802 で現状最良。**提出済み ref 56243664**。
  公開 NB 調査 (experiments.md「公開 NB 調査 09-15」): 店の抽選は seed 依存で使えない、往復売買は純損ゼロ、勝敗差は sell のみ。PPO3 は it 15 で発散し打ち切り (RL 路線は凍結)。
- 保留: 毎手プランナー `agents/legacy/live_e` (対 v41 −25k、壁は経路効率)。テープ上の局所介入 (live_f、git 履歴) は中立〜悪化。

## リポジトリ構成

```
agents/e060, agents/e058     現行提出 (main.py = ライブ層, base.py = v41 本文+注入, actions.json = テープ, NOTICE.txt)
agents/e060.tar.gz           提出物 (tar -czf ... main.py base.py actions.json LICENSE.txt NOTICE.txt)
agents/sr0909_base, sr0909_live  素の 0909 と E055 (GA ツールの参照・対戦相手)
agents/legacy/live_e         保留中の毎手プランナー。他の退役物は git 履歴 (agents/README.md 参照)
third_party/public_agents/   ローカル対戦相手にする公開 NB の実体 (v41 / v39 / v38 / more_yield / aurax7_v4 / guru_v3 / shop_router_0913)
third_party/kaggriculture-cppsim  kagsim (bit-exact C++ エンジン、1 試合 0.1〜1.5 s)
tests/                       現役ツール (一覧と用途は tests/README.md)
.opencode/knowledge/         findings (検証済み知見) / experiments (実験ログ) / meta / refs / history (時系列)
.opencode/data/              0913 テープ (shop_router_0913_tapes.json)、E060 の世界別選択結果 (e060_route_choice.json)
tmp/                         git 管理外の作業領域 (リプレイ、プール、公開 NB の生データ)。消えても再取得できる
```

## 環境とコマンド

- venv: `.venv/bin/python` (kaggle-environments + kagsim)。Kaggle CLI: `.venv/bin/kaggle` (トークン `~/.kaggle/access_token`)。
- **inline の環境変数を複数渡すときは bash を使う** (`bash -c "A=1 B=2 .venv/bin/python ..."`)。zsh は `$cfg` を単語分割しない。
- 直接対決 (両席、kagsim): `.venv/bin/python tests/kag_eval.py agents/X/main.py --vs third_party/public_agents/v41/main.py --games 48 --seed0 5000`
- 実戦席差し替え: `.venv/bin/python tests/kag_eval.py agents/X/main.py --replays 'tmp/<dir>/episode-*.json' --team MMN0222 --base agents/e060/main.py`
- 定型 3 点 (対 v41 / 対現行 / 固定ワールド): `bash tests/live_eval.sh agents/X/main.py`
- 1 試合の日別デバッグ: `LE_DEBUG=1 .venv/bin/python tests/live_debug.py agents/X/main.py --seed N [--shops A,B,...]`
- 実戦リプレイ取得: `kaggle competitions episodes <ref>` → `kaggle competitions replay <id> -p tmp/<dir>`
- 提出: `.venv/bin/kaggle competitions submit kaggriculture -f agents/X.tar.gz -m "..."` (1 日 5 回。**毎回ユーザー確認**。メッセージに `$` を使わない)

## 定型手順

1. **実戦チェック** (提出後 1 日): リプレイ 80 戦を取得 → `tests/fetch_battles.py --dir DIR --team MMN0222 --lb LB.csv` (署名別勝敗)
   → `tests/lineage_census.py` (開幕ハッシュ別) → `tests/match_versions.py --replays 'DIR/episode-*.json' --me agents/e060` (相手の版を全手再現で特定。
   同系は開幕ハッシュでは区別できない) → 大負け相手は `tests/live_debug.py`/`tests/prod_compare.py` で分解。
   レーティング推移: `tests/rating_traj.py <ref>`。上位のリプレイ: `tests/fetch_top.py --lb LB.csv --top 4 --per 10 --out DIR` → `tests/team_profile.py DIR TEAM` / `tests/team_blueprint.py DIR TEAM`。
2. **新 NB / 新テープの取り込み**: `kaggle kernels pull <ref> -p tmp/nb/<ref> -m` → ipynb のコードセルから main.py (writefile か base64/base85 blob) を展開
   → `third_party/public_agents/<name>/main.py` に置く → `kag_eval --vs` で E060 と比較。
   テープなら `.opencode/knowledge/refs/live-policy-ga.md` 末尾の手順 (actions.json 追記 → base.py `_FIN` 登録 → `tests/route_choice.py` で 64 ペア比較 → 勝つペアだけ表へ)。
3. **提出前チェック (必須)**: (a) 最後の callable が `agent` (`.venv/bin/python -c "src=open('agents/X/main.py').read(); env={}; exec(compile(src,'x','exec'),env); print([k for k,v in env.items() if callable(v)][-1])"`)
   (b) 実エンジン self-match で両者 DONE・対称 (`tests/live_eval.sh` 末尾の要領、または kaggle_environments `make().run([agent, agent])`)
   (c) tar を展開して同じ確認 (d) 最大手番 < 1 s。
4. **評価の定石**: 相手は決定的 (v41 / E058 / 記録テープ)。クローン戦は両席ペア margin の正シード率で見る。実戦席差し替えは相手がテープなので
   「相手の適応」が抜ける (適応型には楽観)。注文ベースの売上推定は不執行スパムで過大 → kagsim telemetry の `sell_revenue`/`sold_units` を使う。

## 現行アーキテクチャの要点 (E058/E060)

- base.py = v41 本文に 1 箇所注入 (`del _PAYLOAD` 直後: `_ROUTES` を同ディレクトリの actions.json から読む)。E060 はさらに `_router` の辞書を世界別表に差し替え、
  層内の「最終プラン = route 2 @ step 648」参照 10 箇所を `_FIN(route)` (0909 家系→2、0913 家系→14) に置換。
- 層は `_IMPL.chassis.routes` 経由でテープを読むため 719 手 × 任意本数で差替可。ただし `_R86_FEED_CACHE` 等は route キーの永久キャッシュ →
  同一プロセスでテープを替えるなら module を作り直す。層は 0909 の農場配置に合わせて調整済み (0913 全面差替は YARN 世界で −7〜−21k)。
- main.py (ライブ層): 資金ガード (テープの今後 8 手の購入費)、クローン門 (72 手の位置一致) 付き 6 手先回し売り、shed 圧力逃がし、暇な手の WATER/CARE/FERTILIZE。
  OHFR (相手の収穫待ち在庫を見て先売り) は v41 系に逆効果で OFF。step ≤ 1 と 718 は素通し (v41 の小麦ダンプ開幕と終局プランナーを尊重)。
- GA (`tests/ga/fit58.py`, `evolve58.py`) は層の上では 3 世代で頭打ち (層が売却・給餌・購入を書き換えるため)。13 本前提で 42 本表には未対応。

## 分かっていること (詳細は findings.md)

- 価格弾性: イチゴ +31 個で半値・+62 で底値、牛乳 +38/+76、羊毛 +42/+59。小麦・卵は何百個でも半値にならない。メロンは店需要なし (供給のみ)。
- 2 人市場では「相手が独占する高値品目を潰す」ことが勝敗を決める (供給が無いと相手が 130〜180k 稼ぐ)。テープ同士のミラーが均衡するのはこのため。
- 上位の適応型は世界を問わず ~100k を安定して稼ぐ (我々のテープは世界・シードで 50〜160k)。同シードのペア総額で ±100k 揺れる。
- 開幕: 小麦往復を持つ相手には step0 の小麦ダンプ (v41) が +30k、往復を持たない我々にはほぼ無害。
- 系統の版特定は「同シード・同席・自分は実提出物」で全 719 手再現 (`tests/match_versions.py`)。

## 打ち止め (再試行しない)

- 固定ルート農場への局所介入 (E018-M7、live_f): 給水日・収穫日の書き換え、作物置換、暇な手の施肥 → 全て中立〜悪化。
- テープの作物置換 (GA crop swap、メロン→イチゴ、小麦→ニンジン): −20〜−38k。小麦は給餌原料、収穫リズムは作物固有。
- 高値待ちの保持 (E023、live_e): 相手が売り続ける限り価格は戻らず棚が溢れる。
- 適応型の記録行動をテープ再生 (Majkel / MMPQ): 購買タイミングが崩れて崩壊。
- 上位ルートの丸ごと再生 (E019 期) と部分移植 (E026/E045)。

## RL 路線 (Transformer 方策、ユーザー方針: 諦めない) — 現在地 2026-09-16 夜

- 目的: 1 位 (Majkel1337 3187.7、学習方策) に勝つ学習エージェント。手順・結果表・再開方法は `rl/README.md`「Policy3」節、経緯は experiments.md の 09-16 行。
- 現行本線 = **Policy3** (`model3.py`/`train_bc3.py`/`act3.py`/`play3.py`: joint (タイル, 作業) option、prev 廃止、option を到着まで保持)。
- 到達点: **bc15_ep3 = bc9k_ep3 (bc7 混合 1,200 局) を 1 位 Majkel1337 の実戦 713 局 (2 提出の全量) で 4 epoch 追加学習 → 対 v41 32 戦 own 95.4k / margin −17.1k / 2 勝**
  (bc13 200 局 88.5k/−27.8k、旧基準 bc5_ep3 59.6k/−69k)。**現行提出 E065/E062 には 16 戦 −18k (1 勝)**: 提出は v41 に +2.7k なので差は約 20k。重み・shard は dataset `mmn0222/kaggriculture-rl-majkel0916` v3。
- 今日の結論: (1) BC の天井は demo の平均像 (PASS 529/局) を写すこと。教師を 1 位に絞ると給水 1203・畑 57 株に伸びる。
  (2) 推論側の補完層・班分け・復号規則、待機ラベルの重み、自己軌跡の混合はすべて効かないか悪化 (方策は塞がれた option の代わりに PASS を選ぶ)。
  (3) 1 位の物差し: PASS 54 / WATER 1387 / 2 日放置 1.9% / 渇死 17。bc13 は 510 / 1203 / 4.0% / 62 DIG (生きた作物 24)。
- bc14 (bc13 継続 4 epoch、200 局) は飽和、bc15 (713 局) で +7k → データ量で伸びる。bc16 (bc15 継続 4 epoch) は ep0 同格・ep3 崩壊で epoch 打ち止め。
- **09-17 の訂正 — PASS はラベルのバグだった**: `labels.py` が「当日中に作業が無い」ユニットを一律 (現在タイル, PASS) にしていたため、夕方に歩いているユニットが全部 PASS 教師に
  (Majkel 生 PASS 62/局 に対しラベル 461/局、うち 404 は MOVE)。bc15_ep3 の予測 PASS 448 は**ラベルの時刻分布と一致**していた。修正 (`rl/fix_pass_labels.py` で既存 shard も修復) 後の
  **bc18 = 32 戦 92.8k/−16.9k/3 勝、PASS 486→179・MOVE 2982→3373 と行動は 1 位に接近するが得点は bc15 と同格**。→ **PASS は得点の律速ではない**。上の (2)(3) の「待機」まわりの解釈はこれで置き換わる。
  混合データ (bcw/pq) は修復しても生 PASS 530 前後で、上位陣は本当に待機している。**PASS を減らす目的の施策は打ち止め**。残る 18k は収穫 411 vs 497 / 給水 1259 vs 1387。
- **bc19 (初期値を公開データ 634→2,000 局で焼き直し) は棄却**: bc19_ep3 88.0k で bc18 92.8k・bc20 92.1k を下回る。bc20 (初期値なし) と併せ **混合データの事前学習は無価値〜有害**。教師は 1 位のみで確定。
  1 段目 bc9k19_ep3 は **val option 92.0% (最高) で閉ループ 49.6k (最低)** — val 精度を指標にしてはいけないことの最も鮮明な例。
- **bc20 = Majkel 854 局のみ・初期値なし 4 epoch → 32 戦 92.1k (bc18 92.8k と SE 内)**。混合初期値 bc9k の寄与は測定限界以下。bc20 の継続 (bc21、2 epoch) は 87.8k で不発 — val は改善するのに閉ループが落ちる bc16 と同型で、**epoch 上限 4 は初期値の有無によらない**。
- **09-17 スループット診断**: 給水は問題なし (日末給水済みの株の割合 0.77 vs 師 0.75)。収穫の差は 1 株あたりの実り (yield_units 0.58 vs 0.82) で、原因は**作物構成** — 人参 PLANT 5.8 vs 68.8、苺 52.0 vs 35.2 (人参は初収穫 2 日、苺は 10 日)。
  両ヘッドの open-loop は正常 (農場 91.1%、市場は人参種を多めに予測) で純粋な閉ループ乖離。分岐点は d6-8 の苺の積みすぎ。市場 pos_w デバイアス () は苺を師の水準まで戻すが人参は戻らず得点も動かない → 棄却。**次は人参の cold start**。
- **未修正の欠陥 (09-17 のラベル監査、`rl/README.md`「ラベル監査」節)**: `features.bucket()` の同点が必ず小さい方に落ちるため**市場の数量ラベルが系統的に少なめ**
  (売り 55.8 個/局 = 3.8%・名目 $4,840、買い小麦 11.4 個/局 を取りこぼす)。修正は `bucket()` の同点を大きい方へ倒す 1 行で、bc21 と同じレシピで A/B する。
  ほかに PLACE_COW の mask 違反 19/57,527、目的地ラベルの 0.44% が `legal_ops_at` で非合法。
- **方針 (ユーザー合意)**: 本命 = Majkel 実戦の BC スケール (別提出のエピソードで 400〜600 局 → bc15)。PPO は検証扱い (見込み 3〜4 割): Policy3 用 `rl/sp/train3.py` を Kaggle
  `mmn0222/kaggriculture-sp4-policy3-ppo` (`rl/kaggle13/`) で 2.5 時間の塊、停止基準 = it25〜50 で渇死・逃走が減らない、または 32 戦が bc13_ep3 未満。推論側の細工は再挑戦しない。
- 制約: Mac は BC と PPO を同時に走らせない (MPS メモリ)。Kaggle kernel は `rl/kaggleN/` (投入前に偽 `/kaggle/input` で乾式実行)。
- コミュニティ (09-16 調査): 学習系で動いている公開物は Moriuchi 氏の純粋 RL (BC 4,600 局 + PPO 自己対戦、no-idle/sticky 規則) のみで、kagsim 対 v41 では 70.5k と bc9 と同帯。
  zhincez 氏の失敗録: PPO 10 走ゼロ勝 (4 つの罠)、農場と市場の半分ずつを別方策から取ると 0 コイン (per-turn 模倣の一致率は得点と無相関)。

## 次のステップ

1. E058/E060 の収束確認 (1 日後): 定型手順 1。特に v41 開幕 (sig24 f4c178e5c1) の比率と E060 の 0913 世界の実戦効果。
2. ahmedberatozer / yhay81 の新版監視: `kaggle kernels list -s kaggriculture --sort-by dateRun`。新テープは手順 2 で表に足す (1 版 1 時間)。
3. RL 路線 (09-16 夜: bc15+規則C 97.2k/−14.1k、対 E065 −13.1k。詳細は rl/README.md「次にすること」節と experiments.md 09-16 夜行):
   済: (1) 売り規則 C + 肥料投売りを市場層へ (act_common.py、+3k)、(2) メロン遅れは植え staggering + 売り時と特定 (バグなし)、(4) export Policy3 対応と配備検証 (coin-exact)。
   打ち止め: 日中投売り・牛→羊・メロン保持・温度・容量 d256・bc17 (+19 局)。残る本命は Majkel の日次 drip (900〜1000 局で bc19)。PPO 再凍結。
   核心病理: idle 500 (teacher 54)・無給水 25%・頭数あたり収集半分 = idle 税 15k 級。データ量では直らず (200→713 で不変)。推論・農場側の細工は全滅済みで再挑戦しない。
4. 上位帯との差はプランナー路線でしか埋まらないが壁は経路効率。再開するなら「テープの巡回路を抽出して route にする」から (`agents/legacy/live_e/planner.py`)。

## ナレッジの扱い

`.opencode/knowledge/` の知見は「その時点の証拠」。メタは 2〜3 日で変わる。条件が変わったら再実験してよく、矛盾したら実験を優先して上書きする。
新しい仮説は安いうちに試す (kag_eval 16 戦 = 10 秒)。禁止系の知見も再評価の出発点として扱う。
