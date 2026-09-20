# agents/

| パス | 何か | 状態 |
|---|---|---|
| `e076/` (+`e076.tar.gz`) | **現行提出 (ref 56377779)**。haideptry「Countering the Big 3 Meta」素のまま (V39+v9 層)。vs E072 +1,458/+641、実戦席差し替え +460・勝数+1 | 現役 |
| `e072/` (+`e072.tar.gz`) | **現行提出 (ref 56375841)**。v50-live 土台 + ライブ層 (夜明けガード等、末尾 callable 解決)。vs E071 無敗 +354 | 現役 |
| `e071/` (+`e071.tar.gz`) | v49 土台 + ライブ層 (ref 56375789)。vs E068 32W0L +3,336、席差し替え W34→65 +2,901 | 枠落ち (E076 に押出) |
| `e075/` | h2950 + 自前層の実験。素 h2950 に +347 だが実戦席で勝数 66→63 の反転3のため**見送り** | 実験残骸 |
| `e074/` | E072 + MILK/WOOL 温存層の実験。vs E072 +14・席差し替え −17 で**棄却** | 実験残骸 |
| `e073/` | E071 + 709f 開幕移植の実験。0W32L −7/局で**棄却** | 実験残骸 |
| `e068/` (+`e068.tar.gz`) | v46 + 夜明けガード (ref 56319317、LB 2268.8)。敗因は 709f 系 1W27L とミラー負け越し | 退役 |
| `e060/` | v41 反応層 + 42 本テープ世界別表 + ライブ層 (ref 56221811) | 退役 |
| `e058/` | v41 反応層 + 0909 テープ + ライブ層 (ref 56220023) | 退役 |
| `sr0909_base/` | 素の Shop Router 0909 (yhay81)。GA ツール (`tests/ga/`) の基準・対戦相手 | 参照 |
| `sr0909_live/` | E055 = 0909 + クローン門付き K6 先回し + 開幕ガード。`tests/ga/wrapped.py` が使う | 参照 |
| `legacy/live_e/` | 毎手プランナー (需要適応マクロ + ゾーン巡回)。対 v41 −25k で保留。`planner.py` が本体 | 保留 |

退役した提出 (kaito v56 系 E021〜E052a、E018 ルート、live_a〜d、live_f、e059、0908/0909 期の tar) は git 履歴にあります
(`git log --all -- agents/legacy/<name>`、整理コミット 471c6d6 の直前 bb665d7 で取り出せる)。経緯は `.opencode/knowledge/history.md`。

作り方 (E071/E072 型): base.py = 公開 NB 本文、`main.py` = ライブ層 (最後の callable が `agent`)。
v50 以降は base の live が末尾定義 (`_e343_agent`) で `b.agent` と別物のことがあるため、main.py 側で末尾 callable を解決する (E072 の形)。
提出物は `tar -czf X.tar.gz -C agents/X main.py base.py LICENSE.txt NOTICE.txt` (E076 型の素提出は base なしで `main.py LICENSE.txt NOTICE.txt`)。
