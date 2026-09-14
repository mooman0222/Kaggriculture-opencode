# public_agents — ローカル対戦相手にする公開 Kaggle ノートブックの実体 (すべて Apache-2.0、各 main.py 冒頭に原著者の表記あり)

| dir | 出典 (Kaggle) | 公開日 | 役割 |
|---|---|---|---|
| v41 | ahmedberatozer/kaggriculture-v41-review-candidate | 2026-09-13 | E058/E060 の土台。基準相手 (最新・最強の同系) |
| v39 | ahmedberatozer/kaggriculture-v39-ready-before-the-rush | 2026-09-13 | 実戦相手の主流 (tetsutani NB も同一) |
| v38 | ahmedberatozer/kaggriculture-v38-smarter-feed-stronger-margins | 2026-09-12 | 実戦相手 (reyhanksatria NB も同一) |
| more_yield | ahmedberatozer/more-yield-smarter-labor | 2026-09-12 | 実戦相手の最多版 (80 戦中 25) |
| aurax7_v4 | aurax7/kaggriculture-shop-router-reactive-v4 | 2026-09-13 | 同系派生 |
| guru_v3 | guruprasaathas111/kaggriculture-master-engine-v3 | 2026-09-13 | 同系派生 |
| shop_router_0913 | yhay81/shop-router-0913 | 2026-09-13 | 29 本テープ + 64 世界表。E060 の 0913 側の出典 (テープは .opencode/data/shop_router_0913_tapes.json) |

再取得: `kaggle kernels pull <ref> -p tmp/nb/<ref> -m` → ipynb の `%%writefile main.py` セル、または base64/base85 blob をデコード (AGENTS.md 定型手順 2)。
