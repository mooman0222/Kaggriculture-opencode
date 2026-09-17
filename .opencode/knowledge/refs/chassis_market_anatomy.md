# 公開シャシー (ahmedberatozer v4x) の市場処理の解剖 — 2026-09-17

ハイブリッド (テープ実行系 + Transformer 市場層) を設計するために `agents/e065/base.py`
(= `third_party/public_agents/v43/main.py` と**バイト一致**、3,366 行) を解析した結果。
v45/v46 も構造は同じ (行数 3,496 / 3,627、`_IMPL` `chassis` `_View` などの名前は共通)。

## テープの持ち方

- `base.py:944` `_R108_DATA = json.loads(zlib.decompress(base64.b85decode(...)))` — 1 行 94,551 字、展開後 860KB の JSON。
  `actions` = 3,982 個の行動 dict のプール、`routes` = 41 ルート (719 ステップ分のプール添字)、`shops` = 64 組の世界表。
- `base.py:945` `_ROUTES`、`:946` `_R108_SHOP_ROUTES`、`:964-967` `_R42_OPENING` が**全ルートの step 0 の市場**を上書き。
- ルート 0 は 719 中 716 ステップが非 PASS、414 ステップに市場注文、計 969 注文 (SELL 396 / HIRE 260 / BUY_SEED 189 / BUY_PRODUCT 65 / BUY_ANIMAL 12 / BUY_LAND 2 / 空スロット 45)。
- **罠**: 29,479 のテープスロットが 4,022 個の dict を共有しており `Chassis.__init__` (`:417`) は `list(tape)` の浅いコピー。
  `routes[r][t]['market']` を破壊的に触ると他ルートに漏れる。`_route_action` (`:436-440`) は読み出し時に deep-copy するので**返り値を触るのは安全**。

## ルーター

`_router(observation, step, state)` — `base.py:952-962`。判断材料は 3 つだけ。
step>=144 で `observation['town']['unlocked_shops'][:2]` の組から表引き (YARN 世界は旧 V39 表)、step>=648 でルート 2 に固定、それ以外はルート 0。

## 市場注文を触る層 (約 25)

`_route_action` → `Chassis.act` の層 (`:475-500`、末尾で 10 本に切る) → 約 25 個のラッパーエージェント → `main.py` のオーバーレイ。
`_SETTINGS` (`:948`) が `budget_guard` `room_guard` `clamp_sells` `dead_stock` `terminal_liquidation` `front_run` を**全部 False** にし、
`_sell_lead` / `_apply_suppression` は `:1657-1658` で差し替え済み。実際に効くのは `hand_align` `weed_repair` とラッパー群。

主なもの: `_v224_sales_first` (`:1478-1494`、空/数量 0 を落とし SELL を前へ。ただし同一品目の BUY は SELL の前に残す = 小麦の同ターン往復)、
`_r36_reserve`/`_r36_suppress` (`:1660-1704`/`:1648-1655`)、`_r97_supply` (`:2786-2856`)、`_r127_priority` (`:2995-3023`、スロット 0 に BUY_PRODUCT WHEAT を**前置**)、
`_r128_sale_credit` (`:3057-3065`)、終局プランナ (`:1054-1102`)。

## 売りを外部方策で差し替えるときの制約 (実装前に必読)

1. **全面差し替えは不可**。`_r36_reserve` は**将来のテープ売却量に対して負債を記帳**し `_r36_suppress` が後で差し引く。
   テープの売りが出力に届かないと負債が返済されず**在庫を二重に売る**。
2. `_r97_supply` (`:2826-2834`) と `_r95_replenish` (`:2715-2722`) はテープの `SELL WHEAT` を**わざと削って** PICKUP 用の小麦を守る。小麦の売りには触らない。
3. **スロット 0 は契約**。`_r128_sale_credit` (`:3057-3065`) は `orders[0]` だけを読み、非小麦の SELL であることを要求する。
   並べ替えると売却クレジットが 0 になり `_r128_credit_supply` が無言で降りる (AssertionError は `:3187` の裸 except に飲まれる)。
4. **10 本上限**。`cfg['max_orders']`、`:500` ほか 12 箇所で明示的に確認している。
5. **数量 0 / 空 `[]` はスロット確保の意図**がある (`:614` `:857-862` `:2721`)。ただし `_v224_sales_first` (`:1480`) が step 144 以降で落とすので、不変条件は上流のみ。
6. **終局のテープ形状**: `_shadow_terminal` (`:1044`) は step 712-717 が**きっかり 9 本の SELL** (各 PRODUCT 1 本、数量 >= 100) であることを要求する。書き換えると終局プランナが無言で降りる。
7. 全 SELL ゲート: `_r85_fertilizer` (`:2581`) と `_r44_after` (`:1854`) はその step の注文が全部 SELL のときだけ発火する。

**結論**: 差し込める場所は `make_agent(routes, router, **settings)` (`:910-938`) だけで、売り方策のコールバックは存在しない。
現実的な唯一の道は `main.py` が既にやっている**返り値の後処理で「追加のみ」** (`main.py:96-153` `_sell`)。
main.py:87 のコメントがそのまま危険を名指ししている: *"never drop the base's non-SELL orders (purchases feed the tape's cash chain)"*。

## main.py が base.py に依存している名前

`b.agent` (`base.py:3366` で最後に再束縛)、`b._IMPL` (`:968`、以後再代入されない = 唯一の確かな保証)、
`b._IMPL.chassis.{routes, cfg, players, _projected_shed}` (`:937` で意図的に公開)、`b._View(obs, seat, cfg)` (`:378-402`)。
すべてアンダースコア私有名で `__all__` もなく、`agent` は約 25 回再束縛される。版を上げたら必ず実挙動で確認すること。
