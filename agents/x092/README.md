# 1 手売却延期の開発用試作

**未採用・未提出。単体完結していないので、このディレクトリを tar にして提出しない。**

E090 を読み込み、需要更新直前の MILK/WOOL/STRAWBERRY の一部を次手に売る。
ユニット行動と小麦の注文は変えない。購入注文を含む手では延期しない。

検証対象の設定は `_MULT=2` と `_SKIP_SIMILAR=True`。
無条件に延期する版は E090 との直接対戦に負けるため採用不可。
結果と採否は `.opencode/knowledge/experiments.md` の 2026-09-29 の記録を参照。

再現用の評価設定:

```json
[
  {"name":"e090","path":"agents/e090/main.py"},
  {"name":"hold2_different_farm","path":"agents/x092/main.py",
   "set":{"_MULT":2,"_SKIP_SIMILAR":true}}
]
```

`tests/test_short_hold.py` は対戦計算なしの契約テスト。
ローカル対戦は他の評価と並行せず、低優先度・1ワーカー・試合間2秒以上待機で実行する。
