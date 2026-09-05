---
name: fetch-battles
description: Use when downloading Kaggriculture battle replays from Kaggle and analyzing defeat causes by opponent lineage. Covers episodes listing, parallel replay download, leaderboard fetch, and tests/fetch_battles.py aggregation.
---

# Fetch Battles (実戦履歴の取得と敗因分析)

提出スロットの対戦履歴を全取得し、相手系統 (24手署名・動物構成) 別に集計する手順。
対戦履歴は 1戦 ~30MB のため、プロジェクト内の `tmp/` (git 管理外) に置く。
`/tmp` (容量 7.8GB) は使わない。

## 前提

- venv の python が古い絶対パスで動かない場合、全コマンドに
  `PYTHONPATH=.venv/lib/python3.14/site-packages` を付けて `/usr/bin/python3` で実行する
  (以降の例では `PY=` と略記する。例: `PY="PYTHONPATH=... /usr/bin/python3"`)
- チーム名: `MMN0222`

## 手順

1. **スロット特定**: `kaggle competitions submissions kaggriculture` で最新の2件
   (有効スロット) を確認し、対象の ref を選ぶ
2. **エピソード ID 一覧**: `kaggle competitions episodes <ref>` の1列目
   (ヘッダ2行を除く、数値のみ) を `tmp/<name>/ids.txt` に保存する
3. **リプレイ取得** (並列):
   `cat tmp/<name>/ids.txt | xargs -P 6 -I{} env PYTHONPATH=... /usr/bin/python3 .venv/bin/kaggle competitions replay {} -p tmp/<name>`
   - `kaggle competitions episodes` の一覧は 20 分前後遅延する。直後の試合は含まれない
4. **LB 取得**: `kaggle competitions leaderboard -d -p tmp/<name>` を展開し、
   `kaggriculture-publicleaderboard-*.csv` を得る (相手レーティング用)
5. **集計**: `tests/fetch_battles.py --dir tmp/<name> --team MMN0222 --lb <lb.csv>`
   - 出力: 全体勝率、敗戦一覧、相手署名別・構成別の成績。`battles.json` に保存される
   - margin==0 (引き分け・検証エピソードの self mirror) は敗北扱いで数えられる点に注意

## 分析の注意 (E031b/E034/E035a 実戦の教訓)

- **提出注文の金額評価は不執行スパムで過大になる** (Knight 型の 999 個注文、
  Harrison 戦の 100万単位など)。品目分解は注文数量ベースか農場スナップショット
  (d9/d29 の作物・動物数) で行い、金額は最終所持金 (報酬残差) で確定させる
- 開幕分析は step1-2 の `BUY_PRODUCT WHEAT` 数量 (N) を見る
- 崩壊試合は自農場の動物数推移 (d9→d29) と `consecutive_unfed`・shed 小麦在庫で追跡する
- 席バイアスは `seat` 別勝率で確認する (通常なし)
- 150戦以上溜まったら hypotheses.md 最優先 (収束確認) の材料になる

## ディレクトリ規約

- `tmp/eNNNN_battles/` (例: `tmp/e034_battles/`)。分析スクリプト等の副産物も同梱してよい
- `.gitignore` 済みのためコミットされない
