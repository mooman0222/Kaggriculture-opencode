# 現行メタ: HarvestForge-X クローン群の解析 (2026-08-31, E019/E019b)

分析日: 2026-08-31
データ: LB 上位8チーム (2818-3025) のベスト提出から各5エピソード、計29リプレイ
(Milan Leonard / A Poor Vul / Yusuke Hayashi / tiki tiki tiki / MtN / ringbearer /
Subramanya N / Jiro2)

## 結論

**LB 上位 ~2800-3000 は公開ノートブック `salemali7/kaggriculture-2900`
(HarvestForge-X、2026-08-29 公開・70票) のクローン群**。Subramanya N (羊15+牛2) を
除く7チームが同一シグネチャ。

## クローンのシグネチャ (最終盤面 + 売買集計)

| 要素 | 値 |
|---|---|
| 動物 | 牛8-9 + 羊4-5 (ガチョウ0、卵売却0) |
| 種 | 小麦184-195 / イチゴ43-47 (継続植え直し) / メロン12 / ニンジン6-10 |
| 雇用 | 276-290 人日 |
| 土地 | NW+NE+SW (SE 不買) |
| 終盤 | 農場ほぼ空・shed 完全清算 (売り残しゼロ) |
| 売却量 | ミルク 226-289 / イチゴ 217-311 / 小麦 345-461 / 羊毛 88-242 / メロン 72 / 肥料 322-360 |
| 出力 | vs base $153-158k。実戦 (クローン同士) は市場を食い合い $57-134k |

## HarvestForge-X の構造 (main.py に採用済み、E019)

- **固定720手 (_ACTIONS)**: 上位3トレース (92165990/92185587/92223213) の多数決で再構成。
  市場決定の一致率 99.91% と NB 本文に記載
- **雑草修復**: PLANT/BUILD_PASTURE が雑草タイルに当たったら DIG に置換し、意図した
  行動を翌ターンに復元 (最大8ステップのトレース再生でスケジュール復帰)
- **プレミアム front-run**: MELON/MILK/STRAWBERRY/WOOL について、次ターンの SELL 予定を
  「今ターンに町需要がない」場合に前倒し (在庫キュー位置の改善、価格コストゼロ)。
  boatlee V16-RC5 の Premium Market Lead が出典

## クローン戦の構造とエッジ (E019b)

- クローン同士のミラーは **margin ≤$1k か完全同点** (シード間の絶対額は $34k-124k と
  ショップ抽選で大きく振れるが、勝敗は僅差)
- **相手の全売買計画が既知** → front-run の先読みを 1→4ターンに拡張 (品目別デット方式で
  将来 SELL から前倒し分を相殺) するだけで **素 HF に 19勝1敗・ペア margin +$5.1k
  (10/10 シード正)**。K=2 +$4.6k / K=8 +$3.9k で K=4 が最適
- vs base $153.2k (劣化なし)、K4 同士のミラーも自壊なし

## 再取得の手順 (メタ再チェック用)

```python
# チームID は kaggle competitions leaderboard kaggriculture --show で取得
from kagglesdk import KaggleClient
from kagglesdk.competitions.types.competition_api_service import (
    ApiListTeamPublicSubmissionsRequest, ApiListSubmissionEpisodesRequest)
with KaggleClient() as client:
    req = ApiListTeamPublicSubmissionsRequest(); req.team_id = <teamId>
    subs = client.competitions.competition_api_client.list_team_public_submissions(req).submissions
    # best.id を ApiListSubmissionEpisodesRequest → kaggle competitions replay <epId> -p <dir>
```

リプレイ本体 (~31MB/本) はリポジトリに置かない (再取得可能)。
新規人気 NB の確認: `kaggle kernels list -s kaggriculture --sort-by {voteCount,dateRun}`
