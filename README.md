# master_research

修士研究: **金融ニューステキストを価格由来の数値特徴量に加えると、株価の予測は改善するか**を検証する。
(学部の卒論は「ニューステキストのみ」で、セクターETFの上昇/下降を予測した。)

- 研究方針: [docs/RESEARCH_PLAN.md](docs/RESEARCH_PLAN.md)(未決事項あり)
- 運用方針: [docs/OPERATIONS.md](docs/OPERATIONS.md)
- 決定の記録: [docs/DECISIONS.md](docs/DECISIONS.md)
- 変数の運用方針(目的変数・説明変数): [docs/VARIABLES.md](docs/VARIABLES.md)
- 日報: [docs/daily/](docs/daily/)
- データセット評価: [docs/DATASET_評価_修士.md](docs/DATASET_評価_修士.md)

## ディレクトリ構成
```
修士研究/
├─ 検討したデータセット/     ← git外。検討したKaggle等のデータ原本(1.2GB)
├─ 学部時代のGDELT研究/      ← git外。学部のデータ・コード(8GB)。修士では使わない
└─ master_research/          ← このリポジトリ(GitHubへpush)
   ├─ data/
   │   ├─ raw/        採用した生データ。読み取り専用・本体はgit管理外(README, CHECKSUMS.txtのみ追跡)
   │   ├─ interim/    加工の途中成果物(git管理外)
   │   └─ processed/  モデル入力(git管理外)
   ├─ src/{data,features,models,eval}/   加工・特徴量・モデル・評価のコード
   ├─ tests/  configs/  notebooks/
   ├─ results/{runs,tables,figures,logs}/   runs=1実験1ディレクトリ
   ├─ requirements.txt  Makefile  pytest.ini
   └─ docs/        方針・決定・評価・日報(docs/daily/)。docs/legacy/ は学部データ整理の旧メモ
```

## データの流れ
`data/raw/`(不変)→ `src/data` → `data/interim/` → `src/features` → `data/processed/` → `src/models` → `results/`
`data/raw/` のファイルは書き換えない。整合性は `data/raw/CHECKSUMS.txt` で確認する。

## 環境と操作
```bash
make env      # .venv を作って依存を入れる(Python 3.12)
make verify   # data/raw の整合確認
make test     # テスト
make fetch    # yfinanceから価格を取得
make features # 数値データ(特徴量・目的変数の候補)を作る
```

## データの入手
データ本体はリポジトリに含まれない。入手元・チェックサム・既知の問題は [data/raw/README.md](data/raw/README.md) を参照。
