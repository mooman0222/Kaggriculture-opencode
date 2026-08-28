# Kaggriculture プロジェクト (Kaggle コンペ)

2人対戦の農業シミュレーション。シーズン終了時 (720ターン) の所持金が多い方が勝ち。
`main.py` の `agent(obs)` をそのまま提出する。

## 環境とコマンド

- venv: `.venv/bin/python` (kaggle-environments + pygame-ce)
- Kaggle CLI: `.venv/bin/kaggle` (トークンは `~/.kaggle/access_token`)
- 1試合: `.venv/bin/python tests/run_match.py --seed N --opponent {random,starter,base}`
- 統計評価: `.venv/bin/python tests/evaluate.py --games N --opponent {random,starter,base}`
- 提出: `.venv/bin/kaggle competitions submit kaggriculture -f main.py -m "..."` (1日5回まで、提出前にユーザーに確認)

## 自己改善ループ

毎回の改善作業は次のサイクルで回す:

1. **仮説**: `.opencode/knowledge/hypotheses.md` から未検証の仮説を選ぶ (または新規追加)
2. **実装**: `main.py` に実装。既存挙動と A/B できるようフラグ/分岐で保持
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

## ゲームメカニクスの要点 (詳細は .venv 内 README)

- 作物: WHEAT(10/2d) CARROT(20/2d) TOMATO(50/8d,継続) STRAWBERRY(100/10d,継続) MELON(80/10d)
- 動物: GOOSE(300,卵) COW(400,ミルク) SHEEP(500,羊毛) — 毎日小麦フィード必須
- 価格: 売ると下がり続ける (永続)。店の需要で上がる。メロンは供給に極端に敏感
- 納屋キャップ100、ハンドは1日限り (コスト fib)、土地 NE/SW/SE = $1k/2k/4k