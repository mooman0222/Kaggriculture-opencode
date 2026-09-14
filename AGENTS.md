# Kaggriculture プロジェクト (Kaggle コンペ)

2人対戦の農業シミュレーション。シーズン終了時 (720ターン) の所持金が多い方が勝ち。
現行提出は `agents/e058_base.tar.gz` (E058、2026-09-14、ref 56220023) と `agents/live_d_e057.tar.gz` (E057、ref 56166331)。`agents/e060.tar.gz` (E060、ref 56221811、E058 + 世界別に 0913 テープを混ぜた 42 本表) も提出済。E035a 時点の主力は `agents/kaito_v56_orak16.py` (kaggle が呼ぶのは最後の callable `agent_entry`。
`main.py` は E019c 時点の化石だったため 2026-09-04 に削除済み、E019b が必要なら git 履歴から復元)。

## 環境とコマンド

- venv: `.venv/bin/python` (kaggle-environments + pygame-ce)。
  ※ .venv 内の python が古い絶対パスを指して動かない場合は
  `PYTHONPATH=.venv/lib/python3.14/site-packages /usr/bin/python3` で代用する
- 対戦履歴の置き場: プロジェクト内の `tmp/` (git 管理外。`.gitignore` 済み)。
  `/tmp` は容量 7.8GB のためリプレイ全量 (1戦 ~30MB) ですぐ枯渇する
- Kaggle CLI: `.venv/bin/kaggle` (トークンは `~/.kaggle/access_token`)
- 1試合: `.venv/bin/python tests/run_match.py --seed N --opponent {random,starter,base}`
- 統計評価: `.venv/bin/python tests/evaluate.py --games N --opponent {random,starter,base}`
- 強敵評価: `.venv/bin/python tests/strong_eval.py --agent agents/adaptive_route.py --replays tmp/<battles dir>`
- LB 実戦リプレイ取得: `kaggle competitions episodes <ref>` → `kaggle competitions replay <episodeId> -p tmp/<dir>`
- **提出前チェック (必須)**: kaggle_environments はファイル内で最後に定義された callable を呼ぶ。
  `.venv/bin/python -c "src=open('agents/X.py').read(); env={}; exec(compile(src,'x','exec'),env); print([k for k,v in env.items() if callable(v)][-1])"`
  が `agent_entry` であること (E031b の事故: ヘルパーが最後になり ERROR。E022c〜E030 はオラクル抜きの層が動いていた)
- 提出: `.venv/bin/kaggle competitions submit kaggriculture -f agents/kaito_v56_orak16.py -m "..."` (1日5回まで、提出前にユーザーに確認。
  **メッセージに $ を使うと bash に食われるので注意**)

## 現在の最優先プロジェクト (2026-08-29〜)

**自前ルート生成 (E018)** — 上位 (レーティング~3000) に太刀打ちする唯一の道と判断。
設計書: `.opencode/knowledge/refs/route-generation-project.md`
ツール: `tests/route_gen.py`、`agents/adaptive_route.py` (ルート農場+全品目市場+ギャップフィラー)

### 進行状況

- **M1〜M3d**: ルート農場+リアクティブ市場+ギャップフィラーで avg $61.8k (LB 583.3)
- **E018-M5b (2026-08-29) — E018 最大のブレイクスルー**: ルート再生の1ステップ遅れが
  滞留動物・配置失敗の真因と判明。`obs[k]` では `route[k+1]` を提出 (M5_OFFSET) +
  ルートの HIRE を orig と同時刻に再現 (M5_ROUTE_HIRES、ギャップフィラーはルート雇用
  完了後に M5_GAP_HIRE_LATE) + 種の JIT 化 (12ステップ窓、M5_JIT_SEEDS) +
  フィード資金フロア (M5_FEED_FLOOR)。
  **avg $86.4k vs base (30試合/30勝、旧 $61.8k)、動物配置 15/16、ファーマー位置 orig と 0/240 完全一致**
- **提出済み: ref 55861066 (M5b 反映版、PENDING 2026-08-29 06:18)**
- **E018-M5d (2026-08-29) — ベストルート交代**: 強敵プール構築中に (a) seed_first_day=None の
  TypeError クラッシュ (相手ルートが $0 崩壊した真因) を修正、(b) 修正版で全16ルートを再評価したら
  E018-M4a のランキングが覆り **Subramanya 101692531 p0 が $97.9k で1位** (ИТМОНИ $87.8k)。
  main.py の埋め込みルートを Subramanya (GOOSE8+SHEEP7+COW3・全4作物・土地2区画) に差し替え。
  **vs base $95.1k / 強敵プール勝率 25%→86% / LB 実戦プール 100% / mirror $58-69k**
- **提出済み: ref 55861715 (M5d Subramanya 版、LB 750.7)** — 新規6戦 4勝2敗。
  敗戦2試合の分析 (E018-M5g): (1) 終盤 d25-29 の生産枯渇 (ルート自体の設計、orig も同様)
  (2) 動物配置 15/18 (GOOSE 2頭脱走 = 小麦不足)
- **E018-M5e/M5f (2026-08-29) — 棄却**: 動物・小麦購入のルート同期は 30試合でそれぞれ
  -$2.2k / -$19.2k (資金難シードでキャッシュフロー悪化)。FEED_CASH_FLOOR 100・
  SEED_LOOKAHEAD_JIT 24/36 も悪化。**JIT 12 + フロア 250 + ペース購入が最適のまま**
- **E018-M5h (2026-08-29) — 採用**: 最終日 (d29) の小麦リザーブ解放 (+$941/30試合)。
  d29 は無給餌でも脱走せず基本生産も入るため、当日給餌分だけ残して売却
- **E018-M5i (2026-08-29) — 棄却**: 価格暴落時の保有戦略。mirror で 5戦全敗 —
  相手が売り続ける限り価格は回復しないため早期売却が正しい (安値でも)
- **E018-M5k (2026-08-29) — 採用**: 肥料購入の停止 (+$274、オーダー1枠の節約)
- **E018-M6d (2026-08-29) — 採用**: ギャップフィラーの雇用窓 d3-22 (+$4.2k)。
  終盤 (d23-29) は作物が枯渇するのにフィラー2人を $610/日雇い続けていた。
  **現行: M5h+M5k+M6d で vs base $103.6k (30試合/30勝)**
- **E018-M6f (2026-08-30) — LB ボトルネック調査 (提出 55870361 = 753.8、34試合16勝17敗)**:
  (1) 我々の売上構成はゲーム間でほぼ同一の固定レシピ → 天井 $85-105k = プール中央
  (強い相手 $120-135k) (2) **最大要因: 終盤イチゴ給水崩壊** — d15 33株 → d29 3株。
  植物は2日連続無給水で雑草化。ルートのハンドは d20+ で小麦植え・肥料集め優先で
  給水 ~40回/日が 33株+小麦27株 に不足。勝者 BJFC は34株を d29 まで維持。
  終盤6日収入 我々 $33.7k vs 相手 $64.5k (3) 動物: 牛1頭 $4.3-7.2k ≫ 羊 $2.8k ≫
  ガチョウ $1.8k (BJFC 牛15頭でミルク $64k vs 我々 $21.7k) (4) 人件費 $16.5k vs
  $5.1k (ギャップフィラー2人 = d10-22 で $8k超) (5) レーティング数学: 1時間で
  1000 超えには初期勝率 ~85% 必要 — **確信のある強化版まで再提出を止める**。
  詳細は experiments.md E018-M6f / findings.md 2026-08-30 群

### E019 (2026-08-31): HarvestForge-X の採用 — main.py 全面交代

- **LB 上位 (~2800-3000) の正体はクローン群**: 上位8チームのリプレイ解析 (scratchpad/topdata、
  各チームのベスト提出から5エピソード) で、牛8-9+羊4-5・イチゴ種44・小麦種187・メロン12・
  雇用290人日・ガチョウ0 の同一シグネチャを確認。出どころは公開ノートブック
  **salemali7/kaggriculture-2900 (HarvestForge-X)** — 固定720手 (_ACTIONS) +
  雑草修復 (意図復元8ステップ) + プレミアム4品目 (MELON/MILK/STRAWBERRY/WOOL) の
  前倒し売り (front-run)
- **上位ルートのリプレイ再生は全滅**: 16候補全てが旧 main.py に H2H 負け (0-1/3)。
  上位はリアクティブ補正つきのため素の再生では崩壊する — コード本体の採用が正解
- **検証**: vs base $153.6k (旧 $103.4k) / vs 旧 main.py **H2H 8勝0敗** ($109.4k vs $63.6k) /
  self-match 両者 DONE 同点 $75,490 (検証エピソード OK) / mirror margin ±$1k 以内 or 同点
- 旧 E018 実装は agents/e018_route.py に退避 (A/B 相手用)

### E019b (2026-08-31): クローン戦エッジ — front-run 先読み4ターン化 (採用)

HF mirror は margin ≤$1k の僅差勝負。相手の _ACTIONS が既知なので、front-run の
先読みを 1→4 ターンに拡張 (品目別デット方式)。**素の HF に 19勝1敗・ペア margin
+$5.1k (10/10 シード正)**、vs base/e018 劣化なし、K4 同士も自壊なし。
K スイープ: K=2 +$4.6k / **K=4 +$5.1k (採用)** / K=8 +$3.9k。

### E020 (2026-08-31): 1500 停滞の実戦調査とポスト HF メタ

実戦46戦の解析で判明 (詳細 experiments.md E020): E019b はクローン級に +$1.4-3.5k で
勝つが (FR4 実戦でも機能)、負けは全て新世代 — **C6S9 = prvsiyan/kaggriculture-frontier-
the-soil-remembers-rain (ショップ抽選適応型アンサンブル、day3 の最初のショップで
農場計画を分岐)** と C10S4 変種。frontier は vs base $169-171k・対 K4 ペア +$10.6k。
売りタイミング軸は K=4 が最適と確定 (S∞/WOOLSMART/小麦アービトラージは全て棄却)。
**提出: 55907164 (frontier 素のまま、メタヘッジ)** + 55905717 (E019b K4) の2枚体制。
frontier 原本 = agents/frontier_prvsiyan.py (公開 NB のペイロードを展開、無改変)。

### E020b/c (2026-08-31): frontier への K4 移植と kaito 分岐の調査

- **E020b (採用・提出 55907271)**: frontier 内蔵 V174 (HF 同系統・due_step 方式) に
  main.py と同一の K4 デット式パッチを適用しペイロード再パック (scratchpad の
  make_frontier_k4.py 参照、成果物 = agents/frontier_k4.py)。素 frontier に
  **21勝11分0敗 (+$1.8k/ペア、負けシードゼロ)**。+$0 は kaito 分岐の試合
- **E020c (棄却)**: kaito v48 内蔵 clone_preempt_horizon 2→4 はペア -$3.0k で事故。
  **kaito 分岐は素のまま維持**。副産物: C6S9 = kaito v48 と同定 (構成はシード適応)
- 現行スロット: frontier_K4 (55907271) + frontier 素 (55907164)。E019b は 1432.6 で退役

### E021〜E029 (2026-09-01〜02): v56 世代への移行とエッジの完成 (詳細は experiments.md)

- **E021**: frontier_K4 の 2000 停滞を実戦85戦で分解 → 敗因は kaito v56 系。
  **v56 (kaitofukami/137-161-replay-9-12-final-v56-shop-hybrid) をコア採用**
- **E022a/b/c**: v56 に3層のエッジを構築 — 終局スイープ (d29h18+、素v56 に20勝0敗) +
  セルフオラクル front-run (ファントム v56 を +1..+K ステップ並走させ売却予定を先回り。
  リアクティブ型はデット簿記不要) + K スイープで **K=16 が最適** (K12 に直接 20勝0敗 +$722)
- **E023〜E027 (全棄却の負の資産)**: 暴落ホールド (-$28〜32k、保持は経済の生命線を切る) /
  d27-28 農場介入 (-$13.6k、安全は d29 終盤のみ) / 第2階オラクル (素系勝率を侵食、
  双子対称性でゲート原理不能) / ルート外科 (イチゴ前倒し 1日 -$11k・3日 -$72k、
  v51 エンジンの修復は移植を救わない) / クラス別ルーター (orak16 が全8クラス優位で不要)
- **現行最強 = agents/kaito_v56_orak16.py** (v56 + K16 マルチホライズン + sweep)。
  変種生成器は scratchpad 消失時も experiments.md の記述から再構成可能
  (make_oracle_k.py = _FR 系の K 窓化+デット、make_sweep.py = d29h18 スイープ)
- **E028 (2026-09-02)**: 終局スイープの shed 投影 (同ステップ DROP 分も売る。+$3〜18、無害)。
  公開 NB 由来の即効ネタはこれで枯渇 (v58 は動的優位なし、salemali 2900+ は退役済み素材)
- **E029 (2026-09-02) — 実戦151戦の敗因分析から生まれた新エッジ**: 相手が別ルートなら
  73/104 勝 +$6.8k、同ルートなら 30/47 +$121 のコイントス。我々を破る系統 (eb0a05e1e1) は
  「2軒目が YARN_STORE なら step144 でヤーンルートへ切替」する v56 変種だった。
  **同じ遅延切替 (2-3軒目 YARN_STORE) を実装 → 発火19シード全て正 (+$3〜27k)、非発火は同一**。
  4軒目 (step288) は -$13k で遅すぎ。v51 エンジンはルート丸ごとの交代には耐える。
  ツール: tests/fetch_battles.py (24手署名+動物構成で相手分類)、tests/h2h.py (同シード両席の直接対決)
- スロット (2026-09-02 03:26): **E029 (55952937, PENDING) + orak16 (55934144, 2249)**

### E030〜E036 (2026-09-03〜04): 開幕戦争と Router 対策 (詳細は experiments.md)

- **E030**: 上位15解析 → 開幕小麦先買い N=30 を採用 (素v56 に +137k)。**提出 55973961 (2689.9)**
- **E031**: yhay81 旧 Router テープへのテープオラクル K4 (Router に 8勝4敗→10勝2敗)。
  **E031b** で agent_entry 修正 (kaggle は最後の callable を呼ぶ。E022c〜E030 はオラクル抜き層が本番稼働だった)。
  **提出 55978117**
- **E031d**: E031b (2599) < E030 (2712) は収束段階の差。テープオラクルは実戦 0/47 発火
  (同 Islet 開幕で先回り在庫なし) も無害。天井の正体は**更新版 Router (a887、新主流)**
- **E032-E034**: a887 の step216 以降ルート移植 + 第2テープオラクル。新 Router 実体に 12勝0敗 +7.0k。
  **提出 56003003 (E034)**
- **E035a**: プレミアム即売り市場層 (E034 ミラー 12勝0敗 +446、新 Router・素v56・c43 に劣化なし)。
  全品目版は棄却。**提出 56013057 (2026-09-04)**
- **E036 (棄却)**: 分割開幕 (Jesse 型) は v51 が雇用遅延に耐えず -64k。Knight 代理は無効
- **実戦検証 P1 (2026-09-04)**: E031b 全233戦 (166W67L 71%) vs E034 132戦 (106W26L 80%)。
  a887 は 41W14L +2.3k → **48W6L +4.8k** (移植が実戦で機能)、ce2e は 33W14L → 10W0L、
  Ahmed C13 (-27k) にリベンジ +8.4k。残存: 09969bca1d 崩壊シグネチャ再発 (自C5S6、
  小麦売り越し→給餌切れ連鎖の疑い、E034 でも未修正)、bf5f 0W1L (n=1、-1.1k)、
  15da159ff0 (0W2L -4.2k)、C6S11 (1W3L)。データ: `tmp/e031b_battles/`、`tmp/e034_battles/`
- スロット (2026-09-04 13:08 UTC): **E035a (56013057, PENDING) + E034 (56003003, 2636.2)**。
  E031b (55978117, 2650.0) は押し出し

### E053 (2026-09-09): 上位系統調査とコピー提出 (詳細は refs/top-survey-2026-09-09.md)

- 上位40チームのリプレイと公開ノートの同一シード実走で系統を確定: 13〜40位の約6割が yhay81「Shop Router 0908」
  (09-08 公開) の完全コピー、#3/#5/#8/#12 は Thomas Tschinkel 開幕の派生、#1 SpaTaro・#2 Otter Vibe はライブ方策
- 我々 (2086) は 2000-2399 帯に勝率 47%/32% で実力値。後半テープが家畜を脱走させる欠陥あり (8戦中3戦)
- **提出済み: ref 56111494 = Shop Router 0908 無改変 (`agents/shop_router_0908.tar.gz`)**。完全コピーの実測は 2838
- **E054 提出済み: ref 56112322 = `agents/sr0908_live_k6.tar.gz`** (0908 + クローンゲート付き K6 売り先回し + 暇な手 WATER/CARE)。
  ミラー 24勝0敗 +3.9k、実戦席差し替えで 0908 コピー 16/17・変種 8/9、非クローン相手は素の 0908 と同一。
  評価ツール: `tests/h2h.py A B --games N` (同シード両席)、`tests/eval_replays.py X --base agents/sr0908_base/main.py --glob ...`

### E055 (2026-09-11): 0909 土台への移行 + 開幕ガード (詳細は refs/top-survey-2026-09-09.md)

- E054 は 2740→2590。相手の 8割が Shop Router 0909 系 (09-09 公開) に置き換わり 0908 土台が五分に。開幕荒らし系 (a17674d6) に 0勝5敗 −73k (t0 小麦往復を突かれ t1 の羊購入が失敗)
- **提出済み: ref 56156691 = `agents/sr0909_live_k6.tar.gz`** (0909 + クローンゲート K6 先回し + 暇な手 + 開幕ガード)。0909 ミラー 24勝0敗、実戦60席差し替えで 2勝→53勝
- 現行スロット: 56156691 (E055) と 56112322 (E054 2590)

### E057-live / E058 (2026-09-14): ahmedberatozer 系への乗り換え (詳細は refs/live-policy-ga.md 末尾、experiments.md E057-live〜E058)

- E057 (GA 0909 テープ + 売り層) は 09-13〜14 の実戦 80 戦で 25W55L、LB 2464 (724 位)。相手 75% は **ahmedberatozer 系** (More Yield / v38 / v39 と再配布 aurax7 v4・guru v3)。
  `tests/match_versions.py` (同シード・同席で全 719 手を再現照合) で版まで特定できる — 開幕ハッシュや sig24 では同系が全部同じに見える
- **この系の 13 本テープは 0909 と同一**。強さは反応層 (d12 羊6頭 / d18 トマト投資 / 肥料雇用 / 経済給餌 / 小麦供給 / 売却先回し) と step0 開幕。
  最新 v41 (09-13 22:21) は step0 に小麦ダンプ (BUY5+BUY10+SELL60) を持ち、旧開幕 (R42) の同系に +30k/戦、素 0909 には −50k
- **E058 = `agents/e058/`**: base.py (v41 本文 + テープを actions.json から読む注入) + main.py (E057 のライブ層を chassis 内部に接続、OHFR は逆効果で OFF)。
  対 v41 48W0L +2.3k / 対 v39・More Yield +30k / 対 E057 +11.7k / 実戦 80 席 25→75 勝 (out-of-sample 80 席 23→76)。**提出 ref 56220023**
- GA は `tests/ga/fit58.py` (実戦 80 席 + v41/More Yield 反応クローン、L1) と `tests/ga/evolve58.py` で 0909 テープから再走中 (`tmp/e058/ga/`)。
  E056 の GA テープは旧ライブ層に過学習 (v39 層で 0W24L) — **土台を替えたらテープは素から**
- 評価の定石: `tests/kag_eval.py X --vs Y --games 48` (両席)、`--replays 'tmp/e058/e057_battles/episode-*.json' --team MMN0222 --base Z` (実戦席差し替え)。
  実戦データ: `tmp/e058/e057_battles/` (直近 80)、`tmp/e058/e057_battles2/` (その前 80)、公開 NB 実体: `tmp/e058/agents/<ref>/main.py`

### C 路線 (毎手プランナー) と E059/E060 (2026-09-14 午後、詳細は experiments.md live_e / live_f / route-table / E059 / E060)

- `agents/live_e` (需要適応マクロ + ゾーン巡回ミクロ) は d18 引き継ぎで対 v41 −25k、d6 で −50k。原因は経路効率 (収穫 651 vs 1,006 単位)。
  `agents/live_f` (テープの移動を骨格にタイル判断だけライブ化) は追加判断がすべて中立〜悪化。**固定ルート上の局所介入は打ち止め** (E018-M7 と同じ結論)
- **E060**: 0913 (yhay81 09-13、29 本・64 世界表) を v41 層に載せると YARN 世界で崩れるが BAK/BRU/PET 世界で勝つ → ペア別に良い方を選ぶ 42 本表。
  対 v41 48W0L +2.9k (E058 +2.3k)、対 E058 負けなし +0.8k。評価ツール: `tests/live_eval.sh`, `tests/live_debug.py`, `tests/live_sweep.py`, `tmp/e058/route_choice.py`

### 次のステップ

1. **E058 (56220023) の収束確認**: 1 日後に `kaggle competitions episodes 56220023` → `tests/fetch_battles.py` / `tests/lineage_census.py` で系統別勝率。
   特に v41 系 (sig24 f4c178e5c1) の比率と対戦成績 — 相手が v41 開幕を採用すると +30k の開幕差は消え、層の差 (+2k) だけになる
2. **GA run2 の完走** (`tmp/e058/ga/run2.log`、best = `tmp/e058/ga/best_run2.json`) → `agents/e058/actions.json` に載せて
   `kag_eval --vs v41 --games 48` と実戦 80 席 (base=E058) で確認、正なら第 2 スロット (E057 と入替)
3. **メタ監視**: ahmedberatozer は 1 日 1〜2 版を公開 (`kaggle kernels list -s kaggriculture --sort-by dateRun`)。新版が出たら
   `tmp/e058/agents/` に展開 → `kag_eval --vs` で E058 と比較 → 強ければ base.py を差し替え (注入は `del _PAYLOAD` 直後の 1 箇所)
4. **提出運用**: 締切 9/30、最終評価は締切後 ~2 週間の Bradley-Terry。締切時点で「エラーの出ない最強 2 つ」をスロットに置くことが全て。提出は毎回ユーザー確認

**検証済みの失敗 (再試行しない)**: フルリアクティブ / ファーマー置換 / PASS 介入
(E018-M4b)。ルート再生のステップずれ対策は M5_OFFSET (route[step+1]) が正解で、
農場だけ/市場だけをずらすハイブリッドは崩壊する。M4c (ルート書き換え) は M5b で不要に。
**E018-M7 (2026-08-31): 農場側の追加介入は「移動なしの PASS→WATER/CARE 置換」ですら
-$643、イチゴ植え直し (11,15) は -$8.6k。給水ガード案 (+$15-20k 見込み) は根拠喪失**

### 実装上の注意 (E019 系)

- **農場側 (固定720手) への介入は禁止水域** (E018-M7 / M4b / M6 / M6b が全て悪化)。
  安全な改善は市場側オーバーレイのみ。A/B は変種ファイルを生成して ab.py で比較する
- 評価は決定的相手 (base) 必須。組み込み random は無シードで評価が汚染される (E018-M3)。
  **クローン戦の判定は両席ペア margin の正シード率** (片席だけでは席バイアスが乗る)
- 検証3点セット: (1) vs base (劣化확認) (2) vs 素HF/e018 の H2H (3) 自分同士ミラー (自壊確認)。
  提出前に self-match (debug=True) で両者 DONE を確認する (検証エピソード対策)
- 資産: 旧 E018 実装 = `agents/e018_route.py` (A/B 相手用)。素の HF との比較は
  git 履歴 (E019 コミット時点の main.py。現行ツリーからは 2026-09-04 に削除) か変種生成スクリプトから復元。
  旧ルート JSON (.opencode/data/) は E018 系ツール (eval_routes/strong_eval) 専用の遺産
- 旧 E018 の注意書き (plan 焼き込み・None 罠・adaptive_route 同期) は
  agents/e018_route.py と adaptive_route.py を触るときのみ有効 — git 履歴の
  本セクション旧版を参照

## 自己改善ループ

毎回の改善作業は次のサイクルで回す:

1. **仮説**: `.opencode/knowledge/hypotheses.md` から未検証の仮説を選ぶ (または新規追加)
2. **実装**: `agents/kaito_v56_orak16.py` に実装。既存挙動と A/B できるようフラグ/分岐で保持
3. **検証**: `evaluate.py` で 10〜20試合 + mirror match (自分同士) を確認
4. **記録**: `.opencode/knowledge/experiments.md` に結果を追記
5. **反映**: 有意な改善なら `findings.md` を更新し `agents/kaito_v56_orak16.py` のデフォルトに反映
6. **提出**: 価値があればユーザーに確認して提出、レーティング推移を確認

## ナレッジベースの扱い (最重要)

`.opencode/knowledge/` に検証済み知見・仮説・実験ログが蓄積されている。
ただし**この情報にとらわれないこと**:

- 知見は「その時点の証拠」であって「正しさ」ではない。すべて日付・条件・検証方法付きで記録されている
- メタは変わる: 対戦相手層・評価設定・コミュニティの戦略は日々変化する。条件が変わったら再実験してよい (K16 フル層のローカル実験は1試合約26秒と重い。E018 時代の 1.4秒ではない)
- 実験結果が既存知見と矛盾したら**実験を優先**し、知見を上書きする
- 「やってはいけない」系の知見 (例: 土地購入はメロン専業で逆効果) も、戦略が変われば再評価の価値がある。知見は行動の禁止ではなく再評価の出発点
- 新しい仮説はコストが安いうちにどんどん試す。既存知見との整合性チェックは実験の前ではなく後でよい

## ゲームメカニクスの要点 (詳細は kaggle-environments パッケージ内 README)

- 作物: WHEAT(10/2d) CARROT(20/2d) TOMATO(50/8d,継続) STRAWBERRY(100/10d,継続) MELON(80/10d)
- 動物: GOOSE(300,卵) COW(400,ミルク) SHEEP(500,羊毛) — 毎日小麦フィード必須
- 価格: 売ると下がり続ける (永続)。店の需要で上がる。メロンは供給に極端に敏感
- 納屋キャップ100、ハンドは1日限り (コスト fib)、土地 NE/SW/SE = $1k/2k/4k
- 市場オーダーは1ターン10件まで (売却は最優先で組み立てること)
