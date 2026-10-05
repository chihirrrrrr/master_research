# 運用方針

研究を「再現できる」「リークしない」「あとから説明できる」状態で進めるためのルール。

## 1. 基本原則
1. **生データは不変、加工は全部スクリプト。** `data/raw/` は読み取り専用。手作業でCSVを直さない。`interim/`・`processed/` は、いつでもスクリプトで作り直せる状態にする。
2. **リーク防止を最優先する。** 時刻の整合(ニュース→取引日)は1か所の関数にまとめ、`tests/` で守る。
3. **テスト期間は最後まで見ない。** 目的変数の選択・特徴量・ハイパーパラメータは、検証期間だけで決める(`configs/split.yaml`)。
4. **1実験 = 1ディレクトリ。** 設定・指標・メモを残す(`results/runs/`)。失敗した実験も残す。
5. **決定は記録する。** 方針は `docs/RESEARCH_PLAN.md`、決定とその理由は `docs/DECISIONS.md`。
6. **gitに入れるのはコードと文書だけ。** データ本体、鍵・`.env`、仮想環境、大きな成果物は入れない。pushは指示があるときだけ。

## 2. ディレクトリごとの役割
| 場所 | 役割 | git |
|---|---|---|
| `data/raw/` | 採用した生データ。README(出典・注意点)と `CHECKSUMS.txt` | READMEとCHECKSUMSのみ |
| `data/interim/` | 加工の途中成果物(寄り前ニュースの抽出、価格の整形など) | 管理外 |
| `data/processed/` | モデルの入力(特徴量行列、ラベル) | 管理外 |
| `src/data/` | 取得・整形・時刻整合のコード | ○ |
| `src/features/` | テキスト特徴量(感情・埋め込み)と価格特徴量 | ○ |
| `src/models/` | 学習・予測 | ○ |
| `src/eval/` | 分割・指標・検定・ベースライン | ○ |
| `configs/` | 分割・実験の設定(YAML) | ○ |
| `tests/` | データ契約テスト、時刻整合のテスト | ○ |
| `notebooks/` | 探索専用(確定した処理は `src/` へ移す) | ○(出力は消す) |
| `results/runs/` | 実験の記録(§4) | config・metrics・notesのみ |
| `results/{tables,figures}/` | 論文に載せる最終の表・図(スクリプトで再生成できるもの) | ○ |
| `docs/` | 方針・決定・評価・この運用方針 | ○ |

データの流れ: `data/raw/` → `src/data` → `data/interim/` → `src/features` → `data/processed/` → `src/models` → `results/runs/`

## 3. 日々の操作
```bash
make env        # 初回: .venv を作って依存を入れる
make verify     # raw が CHECKSUMS.txt と一致するか
make test       # テスト(raw の整合、データ契約)
make freeze     # 最初の実験の前に、版を requirements.lock.txt に固定
```
- 実行は `.venv/bin/python -m src.data.xxx` のようにモジュールとして行う(`src/` はパッケージ)。
- NLP(torch, transformers)は重いので `requirements-nlp.txt` に分けた(`make env-nlp`)。GPUが必要なら研究室のサーバーで実行する。
- xgboost を使うには、macOSで OpenMP が必要(`brew install libomp`)。
- 乱数シードは設定ファイルに書き、`config.yaml` に残す。

## 4. 実験の記録(`results/runs/`)
`results/runs/YYYYMMDD_<実験名>/` に、`config.yaml`(設定)・`metrics.json`(指標。val/testを分ける)・`notes.md`(目的・結果・次の仮説)・`git_commit.txt`(コミットハッシュ。未コミットの変更があれば dirty と書く)を置く。大きな予測ファイルは `artifacts/`(git管理外)。詳細は `results/runs/README.md`。

## 5. 価格データ(yfinance)の扱い
- yfinanceの調整済み価格は、配当・分割で**後から値が変わる**。取得した時点を固定するため、`data/raw/prices_yfinance/` に保存し、**同じ場所を上書きしない**。
- 取得のたびに、`FETCH_LOG.md` に **取得日・yfinanceの版・銘柄リスト・期間・`auto_adjust` などの引数**を記録する。再取得するときは日付つきの別ディレクトリに保存して、差分を確認する。
- 取れなかった銘柄(上場廃止・社名変更など)は、一覧にして理由を書く。黙って落とさない。

## 6. gitの使い方
- `main` に小さなコミットを積む。コミットは節目ごと(データ取得の実装、特徴量の追加、実験の結果など)。
- 実験のコードを大きく変えるときは、ブランチを切ってよい。
- **データ・鍵・`.venv` は `.gitignore` で除外済み**。`git add -A` の前に `git status` で確認する。
- 100MB超のファイルはGitHubが拒否する。大きなものは入れない。
- push は、あなたが指示したときだけ行う(Claudeは勝手にpushしない)。

## 7. ノートブック
- 探索・可視化専用。確定した処理は `src/` に移し、ノートブックからは `src/` を呼ぶ。
- 出力つきでコミットすると肥大するので、`nbstripout` で出力を消す(`.venv/bin/nbstripout --install` で、このリポジトリのgit設定に登録する)。

## 8. バックアップ
- データ本体はgitに入らない。**生データの原本は `修士研究/検討したデータセット/` に残してある**(`data/raw/` が壊れたらここから戻し、`make verify` で確認)。
- `interim/`・`processed/` は、スクリプトで再生成できる。
- `results/runs/` の重要な成果と、価格の取得結果(`data/raw/prices_yfinance/`)は、定期的に外部(Time Machineなど)へバックアップする。

## 9. Claude Code の使い方
- **1セッション = 1テーマ**(例: 価格取得 / 時刻整合 / A0 / テキスト特徴量 / 執筆)。長くなったら、新しいセッションに切り替える。
- セッションの最初に、「`docs/RESEARCH_PLAN.md` の §8 の次の作業を進めたい」と伝えれば再開できる(方針は `CLAUDE.md` と `docs/` に入っている)。
- 大きなCSVは画面に出さず、集計して結果をファイルに書く。
- 設計に迷う判断は、実装の前に相談して `docs/DECISIONS.md` に記録する。
- 会話の内容は次のセッションに引き継がれない。**残したいことはmdに書く。**
