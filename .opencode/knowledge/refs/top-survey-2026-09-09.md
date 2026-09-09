# 上位系統調査 (2026-09-09 02:10 UTC) — 一次データのみ、過去知見は不参照

## 方法
- LB CSV (`tmp/top0909/kaggriculture-publicleaderboard-*.csv`、8,262 チーム、2700+ が 76、2500+ が 371)
- 上位12チーム×8戦 + 13〜40位×2戦のリプレイ (`tmp/top0909/replays*/`)。`scratchpad/xray.py` で
  行動列 (farmer+hands) の 24/72 手ハッシュ、試合間プラン一致率 (日別窓)、農場構成・売上を集計 → `xray.txt` / `xray_13_40.txt`
- 公開ノートの main.py を同一シードでミラー実走し、リプレイと行動列を突き合わせて系統確定 (推測ではなく再現一致)

## 系統別の内訳
| 系統 | チーム | 一致 |
|---|---|---|
| yhay81 **Shop Router 0908** (公開 09-08 12:25 UTC) 完全コピー | Jun_value #11 (plan 1.00)、DECEM、mikelou1、Sebastian Mateus、heinado、t3l3k3n3sis、ultimatum game、unreal ほか **13〜40位の約6割** | t608 まで完全同一 |
| 同ノートの近縁 (t70 で分岐) | **mtmr_s1 #4**、Sonu Yadav #10、Ant | plan 0.87〜0.92 |
| Thomas Tschinkel 公開ルーターの開幕 (〜t72) + 独自後半 | **Matthew Huang #3**、Mengfei Li #5、自己找差距 #8、carbonapi #12、Agent 0、kaggricodex | t73〜152 で分岐 |
| kaito v56 開幕 (我々と同一 h24=49813e4a) + 独自後半 | Tarang222 #9 (t112 で分岐、d12 以降ライブ)、Atakan Aldemir | |
| **ライブ方策** (試合ごとにプランが異なる) | **SpaTaro #1 (2960)、Otter Vibe #2 (2947)**、binghua #6、Ad Space Available #7、keiz、Suliman Tadros、M&M&P&Q | 公開物と一致なし |
| 不明 (h24=7bf84ada) | bharat、let cats farm | |

## 頂点2チームの挙動
- **SpaTaro**: 8戦すべて開幕24手から異なる (自己プラン一致 d0-3 で 0.10)。土地を d0〜8 に 3〜7回購入、毎ターン多品目の BUY_PRODUCT を少額発注 (資金不足で大半は未執行)。テープでもルーターでもなく、状態から毎ターン生成する方策。
- **Otter Vibe**: 開幕 d0-3 のみ固定、以降プラン一致 0.05、市場行動は一貫 (0.77)。
- 上位共通の農場: 土地 d6・d11 の2回 (3区画)、牛 8前後 + 羊 3〜10 + 鵞鳥 0〜5、雇用 260〜320/季、
  売上の柱は WHEAT・FERTILIZER・MILK・STRAWBERRY。構成は皆ほぼ同じで、差は後半のプラン適応にある。

## 我々 (MMN0222、980位・2086) の状態
- E052a 直近155戦: 2000-2199 帯に 47%、2200-2399 帯に 32%。2100 は実力値 (収束後の減衰ではない)
- 09-08 以降に Shop Router 0908 の複製が数十チーム分 600 から駆け上がり、これに負けている (相手 sub 56100000+ に 4勝9敗)
- 自前8戦中3戦で d26〜28 に家畜 17→3 頭へ消滅 (小麦在庫あり)。後半テープの給餌が農場実態とずれて脱走させている疑い

## 対応
- **ref 56111494 (2026-09-09 02:26)**: yhay81 Shop Router 0908 の tar.gz を無改変で提出 (`agents/shop_router_0908.tar.gz`、sha256 cec3d031…)。
  完全コピー (Jun_value) の実測は 2838。複製同士は引き分けが増えるため、複製密度が上がるほど頭打ち
- 2900 超はライブ方策のみ。次の本命はテープ改良ではなくライブ方策の設計

## 参考にしたコミュニティ資料 (要旨)
- zhincez「Replaying someone else's tape gets you 88%」: テープ複製は持ち主スコアの ~88%、複製系の上限は金圏未満
- rogerrogerroger3r「The best player's tape is not the best tape」: 高レート者の録画ほど再生で弱い (閉ループの出力だから)
- destbreso「Mutants at the top」「Can I clone you from your DNA」: rank1 は固定プランを持たない別種、BRANCHER は複製可能
- dariushafshar「Rating decay」: 同一提出のまま放置すると 2000-2500 帯で −80/日
