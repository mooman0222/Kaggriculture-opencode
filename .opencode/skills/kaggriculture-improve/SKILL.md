---
name: kaggriculture-improve
description: Kaggriculture (Kaggle の2人対戦農業シミュレーション) エージェントを改善するときに使う。自己改善ループ (仮説→実験→検証→記録→提出) の手順と、ナレッジベース (.opencode/knowledge/) の読み書き規則を提供する。Use when improving the kaggriculture agent, running A/B experiments on main.py, or updating the knowledge base.
---

# Kaggriculture 改善ループ

このスキルは Kaggriculture エージェントの継続的改善のための方法論を提供する。
**セッションをまたいでも改善が続けられる**ことが目的。

## 1. ナレッジベースの場所と役割

`.opencode/knowledge/` 配下:

- `README.md` — ナレッジの扱い方 (最重要規則)
- `findings.md` — 検証済み知見 (日付・条件・検証方法・根拠つき)
- `hypotheses.md` — 未検証の改善仮説 (優先度つき)
- `experiments.md` — 実験ログ (ID 採番制)
- `meta.md` — コンペ状況・スコアリング・本番環境設定

## 2. 改善サイクル

1. **仮説**: hypotheses.md から未検証のものを選ぶ (または新規追加)
2. **実装**: main.py に実装。A/B できるようモジュール定数フラグで保持
3. **検証**: ローカル評価 (下記) で 10〜20試合 + mirror match
4. **記録**: experiments.md に追記
5. **反映**: 有意なら findings.md 更新 + main.py デフォルト反映
6. **提出**: ユーザー確認後に提出、レーティング推移を確認

## 3. ローカル検証の方法

- 勝率・平均所持金: `.venv/bin/python tests/evaluate.py --games 10 --opponent {random,starter,base}`
- 1試合の動作確認: `.venv/bin/python tests/run_match.py --seed N --opponent base`
- mirror match: 自分同士で挙動が安定しているか (メロン戦略は互角に頭打ちする点に注意)
- 評価は 5試合で傾向把握 → 有意差があれば 20試合で確定、の2段構えが効率的
- 本番評価設定はデフォルト構成 (meta.md 参照)。ローカルもデフォルト設定でテストする

## 4. とらわれないための規則 (最重要)

1. 知見は「証拠のある仮説」。**必ず日付・条件・検証方法とセット**で読み書きする
2. 条件 (評価設定・相手層・メタ) が変わったら再検証してよい。ローカル実験は1試合約1.4秒と安価
3. 実験結果が既存知見と矛盾したら**実験を優先**し、findings を上書きする
4. 「やってはいけない」系の知見 (例: 土地購入はメロン専業で逆効果) も状況が変われば再評価の価値がある
5. 仮説はコストが安いうちにどんどん試す。整合性チェックは実験の後でよい

## 5. ゲームの重要パラメータ (詳細は .venv 内 README)

- 作物: WHEAT(10/2d) CARROT(20/2d) TOMATO(50/8d,継続) STRAWBERRY(100/10d,継続) MELON(80/10d)
- 動物: GOOSE(300,卵) COW(400,ミルク) SHEEP(500,羊毛) — 毎日小麦フィード必須
- メロン市場: 供給 +k で価格 = 250-0.01k²。~70個で $200 割れ、~158個で $1 床
- 納屋キャップ100、ハンドは1日限り (コスト fib 1,1,2,3...)、土地 NE/SW/SE = $1k/2k/4k
- 売却は価格が永続的に下がる。店 (unlocked_shops) の消費で需要が増える