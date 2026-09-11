# live-policy ブランチ: kagsim による自前テープ探索 (2026-09-11〜)

## 基盤
- `third_party/kaggriculture-cppsim` = destbreso 氏の bit-exact C++ エンジン (kagsim 0.4.0) + 当リポジトリのパッチ
  (空注文 `[]` がキュー枠を失うバグ修正)。実録12試合を完全再現、L1 (Python エージェント同士) は実環境と同一の勝敗差。
  導入: `uv pip install --python .venv/bin/python -e third_party/kaggriculture-cppsim`
- `tests/kag_eval.py`: 直接対決 (`--vs`) と実戦席差し替え (`--replays --team --base`)。60戦 20秒、24戦 4秒
- `tests/ga/pool.py`: リプレイから相手行動列プール (`tmp/ga/pool.json`, 134本: 0909 系 76、他 39、Thomas 11、荒らし 6)
- `tests/ga/record_live.py`: 反応するクローン (0909 base / E055) の行動列を我々のテープ相手に記録 (`tmp/ga/pool_live.json`, 24 seeds × 両席)
- `tests/ga/fitness.py`: 13本テープ → 世界別ストリーム → kagsim L0 で全プール対戦 (230戦 0.05秒)。系統重み付き平均 margin + 勝率
- `tests/ga/wrapped.py`: 候補テープに E055 のライブ層 (sr0909_live/main.py) を載せ、L1 で 0909 base / E055 相手 24 seeds×両席 (8プロセス、3秒)
- `tests/ga/evolve.py`: (1+λ) 山登り。L0 で λ 候補をふるい、上位 `--top` をライブ層付き L1 で確認して採用 (2段階)
- `tests/ga/validate.py`: 進化テープを `agents/ga_live/` に組み、対 base / 対 E055 / 実戦60戦プールで検証
- `tests/ga/diff.py`: テープ差分の分類

## 変異演算子
売却の移動・数量・分割・削除、作物入替 (PLANT + BUY_SEED)、購入の前倒し、
運搬中の肥料を作物へ FERTILIZE (L1 ミラーの単位トレースから実行可能地点を抽出)、同一世界で記録された 0909 派生の後半移植。
開幕 0..143 は全13本共通なので、開幕への変異は tape0 に適用して全本へ複写。

## 学んだこと
- 素のテープ同士 (L0) の適応度だけで探索すると、売却時期の移動で「K6 先回し層の代替」を見つけて満足し、
  ライブ層を載せた実環境相当ではほぼ効かない (run1〜3: 対 E055 21勝3敗 +1.2k、実戦プール −0.3k)。→ 採用判定はライブ層付き L1 で行う (run4〜)
- Phase A (テープの売りを全部ライブ売却に置換) は失敗: 相手テープはダンプするので保持しても価格は戻らず、shed 満杯・購入失敗で −12k。
  市場層は E055 でほぼ限界、伸びは生産計画側
- 公開の per-turn プランナー (Automaat 1.17.0、phucthaiv02、mohui666 v66) は 0909 に −8k〜−60k。頂点ライブ勢のコードは非公開

## run4 (2段階、guarded 初期化) 途中経過
- it 21: ライブ層付きで 対 0909 base +3.8k/0.98、対 E055 +2.0k/0.90 (holdout seeds: +3.2k/1.00、+1.4k/0.71)
- 実戦60戦プール (E055 対照): +0.8k、53勝→55勝
