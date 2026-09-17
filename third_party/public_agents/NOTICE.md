# public_agents — ローカル対戦相手にする公開 Kaggle ノートブックの実体 (すべて Apache-2.0、各 main.py 冒頭に原著者の表記あり)

| dir | 出典 (Kaggle) | 公開日 | 役割 |
|---|---|---|---|
| **v46** | ahmedberatozer/kaggriculture-v46-first-turn-microstructure-and-s | 2026-09-16 | **E066 の土台 (現行)**。v43 に 16W0L +2,191 |
| v45 | ahmedberatozer/kaggriculture-v45-first-turn-wheat-round-trip | 2026-09-15 | v43 に 16W0L +2,294、v46 に 1W15L |
| v44 | ahmedberatozer/kaggriculture-v44-winning-the-same-turn-sale-race | 2026-09-15 | v43 に 15W1L +1,492 |
| v43 | (E063〜E065 の土台) | 2026-09-14 | `agents/e065/base.py` とバイト一致 |
| v41 | ahmedberatozer/kaggriculture-v41-review-candidate | 2026-09-13 | E058/E060 の土台。RL 側の基準相手 (`play3.py` の既定) |
| v39 | ahmedberatozer/kaggriculture-v39-ready-before-the-rush | 2026-09-13 | 実戦相手の主流 (tetsutani NB も同一) |
| v38 | ahmedberatozer/kaggriculture-v38-smarter-feed-stronger-margins | 2026-09-12 | 実戦相手 (reyhanksatria NB も同一) |
| more_yield | ahmedberatozer/more-yield-smarter-labor | 2026-09-12 | 実戦相手の最多版 (80 戦中 25) |
| aurax7_v4 | aurax7/kaggriculture-shop-router-reactive-v4 | 2026-09-13 | 同系派生 |
| guru_v3 | guruprasaathas111/kaggriculture-master-engine-v3 | 2026-09-13 | 同系派生 |
| shop_router_0913 | yhay81/shop-router-0913 | 2026-09-13 | 29 本テープ + 64 世界表。E060 の 0913 側の出典 (テープは .opencode/data/shop_router_0913_tapes.json) |

再取得: `kaggle kernels pull <ref> -p tmp/nb/<ref> -m` → ipynb の `%%writefile main.py` セル、または base64/base85 blob をデコード (AGENTS.md 定型手順 2)。

## v44 以降の取り出し方 (2026-09-17)

ipynb の cell 4 が `SOURCE_BYTES = b''.join((...))` + `EXPECTED_MAIN_SHA256` の形。`MAIN = WORKDIR` の手前までを
`exec` して `SOURCE_BYTES` を取り、sha256 を照合してから書き出す (第三者コードを丸ごと実行しないため)。

```python
src = "".join(json.load(open(nb))["cells"][4]["source"])
ns = {"hashlib": hashlib}; exec(compile(src.split("MAIN = WORKDIR")[0], "<blob>", "exec"), ns)
assert hashlib.sha256(ns["SOURCE_BYTES"]).hexdigest() == ns["EXPECTED_MAIN_SHA256"]
```
新版の確認: `kaggle kernels list --competition kaggriculture --sort-by dateRun --page-size 40`
