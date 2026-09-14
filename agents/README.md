# agents/

| パス | 何か | 状態 |
|---|---|---|
| `e060/` | 現行提出 (ref 56221811)。v41 反応層 + 42 本テープ (0909×13 + 0913×29) の世界別表 + ライブ層 | 現役 |
| `e058/` | 現行提出 (ref 56220023)。v41 反応層 + 0909 テープ 13 本 + ライブ層 | 現役 |
| `e060.tar.gz` | E060 の提出物 | 現役 |
| `sr0909_base/` | 素の Shop Router 0909 (yhay81)。GA ツール (`tests/ga/`) の基準・対戦相手 | 参照 |
| `sr0909_live/` | E055 = 0909 + クローン門付き K6 先回し + 開幕ガード。`tests/ga/wrapped.py` が使う | 参照 |
| `legacy/live_e/` | 毎手プランナー (需要適応マクロ + ゾーン巡回)。対 v41 −25k で保留。`planner.py` が本体 | 保留 |

退役した提出 (kaito v56 系 E021〜E052a、E018 ルート、live_a〜d、live_f、e059、0908/0909 期の tar) は git 履歴にあります
(`git log --all -- agents/legacy/<name>`、整理コミット 471c6d6 の直前 bb665d7 で取り出せる)。経緯は `.opencode/knowledge/history.md`。

作り方 (E058/E060 型): base.py = 公開 NB 本文に「`_ROUTES` を actions.json から読む」注入 (`del _PAYLOAD` 直後) [+ E060 は `_router` 表と `_FIN`]、
main.py = ライブ層 (最後の callable が `agent`)。提出物は `tar -czf X.tar.gz -C agents/X main.py base.py actions.json LICENSE.txt NOTICE.txt`。
