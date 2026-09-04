---
description: 改善ループを1サイクル回す (仮説選定→実装→検証→記録→提出判断)
agent: build
---

Kaggriculture エージェントの改善ループを1サイクル回してください。

1. **仮説選定**: `.opencode/knowledge/hypotheses.md` から優先度が最も高い未検証仮説を1つ選ぶ。全部検証済みなら新規仮説を追加
2. **実装**: `agents/kaito_v56_orak16.py` に仮説を実装する (市場側オーバーレイのみ。農場側=固定720手への介入は禁止水域)。
   A/B できるよう変種ファイルを生成し、既存挙動はフラグ/分岐で保持する
3. **検証**:
   - `.venv/bin/python tests/evaluate.py --games 10 --opponent base` (random は無シードで汚染されるため使わない)
   - 変種同士は `.venv/bin/python tests/h2h.py A.py B.py --games 6` で同シード両席対決
   - self-match (自分同士、debug=True) で両者 DONE を確認
   - 平均所持金・勝率・min/max を記録
4. **記録**: `.opencode/knowledge/experiments.md` に追記 (ID 採番、日付、仮説、セットアップ、結果、判断)
5. **反映**: 有意な改善なら `.opencode/knowledge/findings.md` を更新 (日付・条件付き) し、`agents/kaito_v56_orak16.py` のデフォルトに反映。既存知見と矛盾したら実験結果を優先して上書きする
6. **提出判断**: 価値があればユーザーに確認して `kaggle competitions submit kaggriculture -f agents/kaito_v56_orak16.py -m "..."` で提出 (1日5回制限)。
   提出前に `exec` でファイル内最後の callable が `agent_entry` であることを確認する (kaggle は最後の callable を呼ぶ)

注意: 既存知見との整合性は実験の前に気にしすぎないこと。実験コストは K16 フル層で1試合約26秒 ( screening はシード少なめで)。