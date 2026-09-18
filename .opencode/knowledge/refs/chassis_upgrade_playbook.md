# 公開シャシー更新時の手順書 — 2026-09-18

ahmedberatozer 氏は **1 日 1〜2 版**を公開する。09-15〜17 の 3 日で v44 → v48。
**放置すると素の公開エージェントに負ける** (E065 は自分の土台 v43 には 16W0L で勝ちながら、素の v45 に 1W15L だった)。

## 1. 新版の検出と取り込み (30 分)

```
kaggle kernels list --competition kaggriculture --sort-by dateRun --page-size 40
kaggle kernels pull ahmedberatozer/<ref> -p tmp/nb/<ref> -m
```
ipynb の cell 4 が `SOURCE_BYTES = b''.join((...))` + `EXPECTED_MAIN_SHA256`。
`MAIN = WORKDIR` の手前までを `exec` して取り出し、**sha256 を照合してから**書き出す (第三者コードを丸ごと実行しない)。
**罠**: 人気ノートが既存版の再掲のことがある (flexonafft「Multi-Route Farming Agent」93 票は v45 とバイト一致)。
取り込む前に既存の `third_party/public_agents/*/main.py` と sha256 を突き合わせる。

## 2. 載せ替え (30 分)

`agents/eNNN/` を作り、`base.py` = 新版の main.py、`main.py` = **`agents/e069/main.py` をコピー**。
e069 のライブ層は夜明けガードが**シャシー非依存**に書かれている (生の observation だけから棚を射影する)。
`_load` の第 2 引数 (モジュール名) だけ変える。

## 3. 測る (30 分)

```
tests/kag_eval.py agents/eNNN/main.py --vs third_party/public_agents/<新版>/main.py --games 32   # 自分の土台に対する上乗せ
tests/kag_eval.py agents/eNNN/main.py --vs agents/<現提出>/main.py --games 32                     # 現提出との比較
tests/kag_eval.py agents/eNNN/main.py --vs agents/eNNN/main.py --games 8                          # 自己対戦は avg +0 (対称) が必須
tests/kag_eval.py agents/eNNN/main.py --replays 'tmp/<battles>/episode-*.json' --team MMN0222 --base agents/<現提出>/main.py
```
最後の実戦席差し替えが最も当てになる (記録された実際の相手が使える)。

**ライブ層が土台に追い越されていないかを必ず見る**: 「自分の土台に対する上乗せ」が誤差なら、その層はもう価値がない。
E065 のライブ層は v43 で +1,700 だったが v46 では **+102** に消えた (本家が same-turn sale race / sale timing を取り込んだため)。

## 4. 層の価値が消えていたら — 伸びしろの測り直し

**実装の前に測る。** シャシーを kagsim で走らせながら「自分の層なら何をするか」を並べて記録するだけで、実装ゼロで上限が出る
(`scratchpad/headroom.py` の形)。Transformer 売り層はこれで対 v46 +$9/局 と分かり、丸一日の実装を節約した。

エンジンの規則を突く層は**土台が変わっても効く**。現行の夜明けガードがその例で、v43 +203 / v45 +149 / v48 +335。
一方、本家と同じ着想の層 (売り時・スロット順) は取り込まれて消える。**次に探すならエンジン規則側**。

## 5. A/B の作法

- 対 素の土台で**同 seed・同席のペア差**を取る (`scratchpad/ab_dawn.py` の形)。ユニット行動を触らない層なら分散が小さく微差が測れる。
- **調整用と判定用の seed を分ける**。夜明けガードは tune 7000-7059 で +248±24、held-out 8000-8059 で +244±29 と一致したので過適合なしと判断した。
- 自己対戦が avg +0 で対称であること、max act が 50ms 未満であることを提出前に確認する。
