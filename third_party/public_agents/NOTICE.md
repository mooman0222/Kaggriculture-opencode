# public_agents — ローカル対戦相手にする公開 Kaggle ノートブックの実体 (すべて Apache-2.0、各 main.py 冒頭に原著者の表記あり)

| dir | 出典 (Kaggle) | 公開日 | 役割 |
|---|---|---|---|
| **v50** | ahmedberatozer/kaggriculture-v50-early-yarn-commit | 2026-09-19 09:33 | **E072 の土台 (live は末尾 `_e343_agent**。`m.agent` は旧エントリなので末尾 callable で解決すること)。v49 に +353 (発火局のみ) |
| **v49** | ahmedberatozer/kaggriculture-v49-funded-sale-timing-and-worker | 2026-09-19 08:00 | **E071 の土台**。v46 に 30W2L +3,587。Thomas Tschinkel 2945 Farm 経済層の後付け |
| v48 | ahmedberatozer/kaggriculture-v48-clear-the-queue | 2026-09-17 22:09 | **クローン戦では v46 と同一挙動 (avg +0)**。新層が同士討ちで発火しない。なお v48 も末尾 `_PG_HOST` が live で `m.agent` と別物 |
| v47 | ahmedberatozer/kaggriculture-v47-reactive-market-coordination | 2026-09-17 15:44 | 同上 |
| v46 | ahmedberatozer/kaggriculture-v46-first-turn-microstructure-and-s | 2026-09-16 | E066/E068/E070 の土台 (退役)。v43 に 16W0L +2,191 |
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

## 棚卸し (2026-09-18、09-20 追記)

- 09-20 新着: haideptry「Countering the Big 3 Meta」(V39+v9 層、09-20) が E072 に +1,458/+641・席差し替え +460 で **E076 として素のまま提出**。同作者「The 2950 Peak Farm」も +1,074/+413。v9 層の中身は RACE 予約 (step192〜)・COURIER・CARROT・HERD・0911 開幕・aurax7 日末守衛・Gluzdov 終端救済。実体は `agents/e076/main.py` (verbatim) と `agents/e075/base.py` (h2950)。NB 生データは `tmp/nb/haideptry-*/` (git 管理外、再取得可)

- **yhay81 氏は 09-13 の `shop-router-0913` (4 票) を最後に Kaggriculture から離脱**した (以後は ARC Prize)。
  `shop-router-0909` は 132 票あった主流だったので、ahmedberatozer 系への移行は選択ではなく必然。
- **更新を続けているのは ahmedberatozer 氏だけ**。09-15〜17 の 3 日で v44 → v48。**1 日 1〜2 版のペースなので毎日見る**。
- 紛らわしい再掲に注意: flexonafft「Multi-Route Farming Agent」(93 票) は **v45 とバイト一致**。
  取り込む前に `sha256` を既存の `third_party/public_agents/*/main.py` と突き合わせること。
- 他候補の実力 (対 v46 16 戦): aurax7 v7 −204 (3W13L)、salemali7「2900+」−33,291 (0W16L)。**v46 を超える公開物はない**。
