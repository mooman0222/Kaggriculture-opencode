# agents/

| パス | 何か | 状態 |
|---|---|---|
| `e060/` | 現行提出 (ref 56221811)。v41 反応層 + 42 本テープ (0909×13 + 0913×29) の世界別表 + ライブ層 | 現役 |
| `e058/` | 現行提出 (ref 56220023)。v41 反応層 + 0909 テープ 13 本 + ライブ層 | 現役 |
| `e060.tar.gz` | E060 の提出物 | 現役 |
| `sr0909_base/` | 素の Shop Router 0909 (yhay81)。GA ツール (`tests/ga/`) の基準・対戦相手 | 参照 |
| `sr0909_live/` | E055 = 0909 + クローン門付き K6 先回し + 開幕ガード。`tests/ga/wrapped.py` が使う | 参照 |
| `legacy/live_e/` | 毎手プランナー (需要適応マクロ + ゾーン巡回)。対 v41 −25k で保留。`planner.py` が本体 | 保留 |
| `legacy/live_f/` | テープの移動を骨格にタイル判断だけライブ化 (`tilelive.py`)。追加判断は全て中立〜悪化 | 却下 |
| `legacy/e059/` | v41 層 + 0913 全面差替 (YARN 世界で崩れる) → E060 に置換 | 退役 |
| `legacy/live_d/`, `live_d_e057.tar.gz` | E057 (GA 0909 テープ + 相手適応売り OHFR)。2464 | 退役 |
| `legacy/ga_live/`, `ga_live_run4.tar.gz` | E056 (kagsim GA テープ + E055 層) | 退役 |
| `legacy/live_a〜c/` | ライブ方策の初期実験 (価格保持売り / day12 プランナー / 第4区画ガーデン) | 退役 |
| `legacy/sr0908_*`, `shop_router_0908.tar.gz`, `sr0908_live_k6.tar.gz`, `sr0909_live_k6.tar.gz` | 0908/0909 期の提出 (E053〜E055) | 退役 |
| `legacy/kaito_v56_*.py` | kaito v56 系 (E021〜E052a、K16 オラクル・スイープ・ヤーン切替など) | 退役 |
| `legacy/frontier_*.py`, `e018_route.py`, `adaptive_route.py`, `route_agent.py`, `planner.py`, `base.py` | E018〜E020 期 (自前ルート生成、frontier) | 退役 |

作り方 (E058/E060 型): base.py = 公開 NB 本文に「`_ROUTES` を actions.json から読む」注入 (`del _PAYLOAD` 直後) [+ E060 は `_router` 表と `_FIN`]、
main.py = ライブ層 (最後の callable が `agent`)。提出物は `tar -czf X.tar.gz -C agents/X main.py base.py actions.json LICENSE.txt NOTICE.txt`。
