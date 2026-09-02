# 上位動向調査 (2026-09-02 01:50 UTC)

## LB スナップショット
- top: tetsuya 2908.8 / Crop Dusta 2884.4 / **3정훈 2862.0 (新顔)** / QQ Farming 2861.4 / **RngRng 2855.2 (新顔)** / yukino 2825 / MtN 2819 / senkin13 2802 / Alperen Aydın 2801 / cg 2795
- 2700+ 22 チーム、2000+ 659 チーム。2500 帯から急減 (2500:59, 2600:30, 2700:13, 2800:8)
- 我々: MMN0222 **2374.4 (239位)**、orak16 ref 55934144 (09-01 06:52)。2300-2400 帯は約 100 チームの密集帯

## 新規ノートブック (08-31〜09-02)
| NB | 票 | 結論 |
|---|---|---|
| kaitofukami **v58 Minimax Closed Loop** | 38 | v56 と同一バックボーン + 10 ルート×4 チェックポイント (step72/96/144/360) の完全一致シグネチャルーター。凍結ストリーム 238/238 だが **動的 v58 vs v57 = 2勝8分2敗、本人も優位なしと明記**。我々の orak16/sweep とは直交で無効化なし。唯一の収穫: **step360 ミラー検出 (公開状態 240 ターン一致) → 市場タイミングずらし clone ルート** (E025 双子問題への v58 側の回答) |
| salemali7 "2900+" | 71 | = V16-RC5 (E019 で採用→E021 で退役済み)。LB 証拠なし、新規性なし |
| tetsutani "Shape the Shop" | 69 | yhay81 Fieldbook 派生のテープ再生機。tetsuya 本人の証拠なし。収穫: **step718 の SELL FERTILIZER を shed 容量 (100) で過大発注** (COLLECT が市場より先に解決するため同ターン採取分を拾える)。我々の _sweep は現在値しか売っていない → 1 行で試せる |
| motemen Meta Census | 0 | 24 手署名で lineage 集計。08-29 Kaito 系 58% → 08-30 38% に減少、無名 `891495828b` 16% / `e9c9700032` 7% が急伸 (3정훈/RngRng の系統は不明) |
| georgymamarin Live Meta Report | 31 | 分散はほぼ matchup 由来、within-episode margin が正しい物差し。labor (hires/peak crew) が bank と最相関 |
| destbreso Island GA / kagsim | 9-11 | 自前スケジュール探索 + bit-exact C++ シミュ (2000 ep/s、github destbreso/kaggriculture-cppsim)。緩和上界 $229k (strawberry 主導 + tomato/egg)。**強者パネル最適化は実プールで順位逆転**、fitness は Phi(margin/σ) を実プール分布で |

## 含意
1. 公開 NB 由来の即効ネタは **sweep の肥料過大発注** のみ (低リスク・要計測)
2. v58 は追わない (動的優位なし)。ミラー検出→clone 市場ルートだけ E025 の再検討材料
3. 2800+ の新顔 (3정훈/RngRng) と無名 lineage 2 本の 24 手署名を daily episodes から取り、ローカル勝率を測るのが次の観測課題
4. ラダー 2374→2700 の壁: 上位は非公開の自前スケジュール勢 (Island GA 論の通り)。公開 v56 系列の改良では 2500 帯が天井の可能性
