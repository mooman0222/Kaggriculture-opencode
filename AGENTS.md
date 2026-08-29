# Kaggriculture プロジェクト (Kaggle コンペ)

2人対戦の農業シミュレーション。シーズン終了時 (720ターン) の所持金が多い方が勝ち。
`main.py` の `agent(obs)` をそのまま提出する。

## 環境とコマンド

- venv: `.venv/bin/python` (kaggle-environments + pygame-ce)
- Kaggle CLI: `.venv/bin/kaggle` (トークンは `~/.kaggle/access_token`)
- 1試合: `.venv/bin/python tests/run_match.py --seed N --opponent {random,starter,base}`
- 統計評価: `.venv/bin/python tests/evaluate.py --games N --opponent {random,starter,base}`
- 強敵評価: `.venv/bin/python tests/strong_eval.py --agent agents/adaptive_route.py --replays /tmp/opencode/lbdata`
- LB 実戦リプレイ取得: `kaggle competitions episodes <ref>` → `kaggle competitions replay <episodeId> -p /tmp/opencode/lbdata`
- 提出: `.venv/bin/kaggle competitions submit kaggriculture -f main.py -m "..."` (1日5回まで、提出前にユーザーに確認。
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

### 次のステップ

1. **LB 結果確認 (最優先)**: `kaggle competitions submissions kaggriculture` で
   ref 55861066 の publicScore を確認。判定は **meta.md の基準** (60試合=~5時間待つ、
   50pt 未満の差はノイズ、周辺レーティング相手の勝率で比較)。
   スコア確定後は `kaggle competitions episodes 55861066` で実戦相手を確認
2. **強敵ローカル評価の整備 (高優先)**: LB 実戦リプレイ (/tmp/opencode/lbdata、9本) の
   相手を tests/strong_eval.py で再生するが、現状は相手側が崩壊しがち ($0)。
   相手を「topdata ルート+適応市場」で強くする改良が必要。
   代替: 旧リアクティブ main.py ($41k) や Kaito v27 埋め込みを相手プールに
3. **残課題 (改善余地)**: ハンド位置が orig と ~6% ずれる (資金不足時の HIRE 遅延) →
   低シード (min $37.6k) の改善。E018-M5b 後に未着手

**検証済みの失敗 (再試行しない)**: フルリアクティブ / ファーマー置換 / PASS 介入
(E018-M4b)。ルート再生のステップずれ対策は M5_OFFSET (route[step+1]) が正解で、
農場だけ/市場だけをずらすハイブリッドは崩壊する。M4c (ルート書き換え) は M5b で不要に

### 実装上の注意 (E018 系)

- **A/B テストはフラグを plan に焼き込むこと**: `adaptive_route.py` のモジュール定数
  (GAP_HIRE_MAX, M5_*) は `build_plan` 実行時に plan へコピーされ、以降は plan を参照。
  実行中にモジュール定数を書き換えると既存エージェントの挙動まで変わり A/B が汚染される
  (E018-M5b で実際に踏んだ罠)
- 評価は決定的相手 (base/starter) 必須。組み込み random は無シードで実行ごとに相手が変わり
  評価が汚染される (E018-M3)。**seeds 0-14 で評価** (seeds 100-104 は不利なシード帯、E018-M4a)
- 資産: ベストルート = `.opencode/data/route_itmoni_101730370_p0.json` (永続化済み、
  main.py の `_ROUTE_B85` からも復元可)。LB 実戦リプレイ = /tmp/opencode/lbdata (9本)
- main.py と agents/adaptive_route.py は同じロジックのコピー。**修正は両方に反映すること**

## 自己改善ループ

毎回の改善作業は次のサイクルで回す:

1. **仮説**: `.opencode/knowledge/hypotheses.md` から未検証の仮説を選ぶ (または新規追加)
2. **実装**: `main.py` / `agents/adaptive_route.py` に実装。既存挙動と A/B できるようフラグ/分岐で保持
3. **検証**: `evaluate.py` で 10〜20試合 + mirror match (自分同士) を確認
4. **記録**: `.opencode/knowledge/experiments.md` に結果を追記
5. **反映**: 有意な改善なら `findings.md` を更新し `main.py` のデフォルトに反映
6. **提出**: 価値があればユーザーに確認して提出、レーティング推移を確認

## ナレッジベースの扱い (最重要)

`.opencode/knowledge/` に検証済み知見・仮説・実験ログが蓄積されている。
ただし**この情報にとらわれないこと**:

- 知見は「その時点の証拠」であって「正しさ」ではない。すべて日付・条件・検証方法付きで記録されている
- メタは変わる: 対戦相手層・評価設定・コミュニティの戦略は日々変化する。条件が変わったら再実験してよい (ローカル実験は1試合約1.4秒と安価)
- 実験結果が既存知見と矛盾したら**実験を優先**し、知見を上書きする
- 「やってはいけない」系の知見 (例: 土地購入はメロン専業で逆効果) も、戦略が変われば再評価の価値がある。知見は行動の禁止ではなく再評価の出発点
- 新しい仮説はコストが安いうちにどんどん試す。既存知見との整合性チェックは実験の前ではなく後でよい

## ゲームメカニクスの要点 (詳細は kaggle-environments パッケージ内 README)

- 作物: WHEAT(10/2d) CARROT(20/2d) TOMATO(50/8d,継続) STRAWBERRY(100/10d,継続) MELON(80/10d)
- 動物: GOOSE(300,卵) COW(400,ミルク) SHEEP(500,羊毛) — 毎日小麦フィード必須
- 価格: 売ると下がり続ける (永続)。店の需要で上がる。メロンは供給に極端に敏感
- 納屋キャップ100、ハンドは1日限り (コスト fib)、土地 NE/SW/SE = $1k/2k/4k
- 市場オーダーは1ターン10件まで (売却は最優先で組み立てること)
