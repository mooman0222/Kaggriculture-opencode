# Kaggriculture

[Kaggle コンペ Kaggriculture](https://www.kaggle.com/competitions/kaggriculture) 用エージェント。2 人対戦の農業シミュレーション (30 日 × 24 ターン)、終了時の所持金で勝敗。

- 現行提出: `agents/e060/` (ref 56221811) と `agents/e058/` (ref 56220023)。公開 NB「ahmedberatozer v41」の反応層にテープ差替と自前ライブ層を載せたもの。
- 現在地・手順・ツール: `AGENTS.md`、`agents/README.md`、`tests/README.md`。知見: `.opencode/knowledge/`。
- 高速評価は `third_party/kaggriculture-cppsim` (kagsim、bit-exact C++ エンジン): `uv pip install --python .venv/bin/python -e third_party/kaggriculture-cppsim`
