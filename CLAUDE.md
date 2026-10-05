# CLAUDE.md

修士研究のリポジトリ。応答は日本語で行う。概要は README.md、方針は docs/RESEARCH_PLAN.md、決定の履歴は docs/DECISIONS.md、運用ルールは docs/OPERATIONS.md を参照。

## 研究の前提
- 問い: 金融ニュースのテキストを、価格由来の数値特徴量に加えると、株価の予測は価格のみより改善するか。
- 学部との違い: テキストのみ → テキスト + 数値。目的変数は**未決**(時間幅 × 種類の組み合わせを比較。RESEARCH_PLAN.md §4)。予測時点は取引日の寄り前(提案)。
- 採用データ(US-1): `data/raw/news_price_events`(米国223銘柄、時刻つき見出し、ODC-By)+ `data/raw/nyt_headlines`(NYT「World」欄。補助)+ yfinance価格(**未取得**)。出典は data/raw/README.md。

## ルール
- `data/raw/` は**書き換えない**(読み取り専用)。加工は `src/` のスクリプトで行い、`data/interim/` → `data/processed/` に出力する。
- データ本体と機密(`.env`、鍵JSON)は**コミットしない**。`.gitignore` で除外済み。pushはユーザーの指示があるときだけ行う。
- 大きなCSVは全体を表示しない。`head`・サンプル・集計を使い、結果はファイルに書く。
- 予測の評価は、日単位のwalk-forward、価格のみのベースラインとの比較を必須とする。
- リークを避ける: ニュースの時刻 → 取引日の対応は、引け後・週末を次営業日に寄せる(詳細は RESEARCH_PLAN.md §6)。
- 既存ラベル(`forward_return`、NYTの`Label`)はそのまま使わず、価格から作り直す(理由は data/raw/README.md)。
- 時刻は `original_first_date`(分まで残っている)を使う。`time_window` は1時間に丸められている。
- 重要な決定は docs/DECISIONS.md に日付つきで追記する。

## 操作
- 環境: `make env`(`.venv`)、`make verify`(rawの整合)、`make test`、`make freeze`。実行は `.venv/bin/python -m src.…`。
- 1実験 = `results/runs/YYYYMMDD_名前/`(config・metrics・notes。`results/runs/README.md`)。分割は `configs/split.yaml`、テスト期間は最後まで見ない。
- yfinanceの価格は `data/raw/prices_yfinance/` に、取得日・版を `FETCH_LOG.md` に記録して保存し、上書きしない。
- 時刻整合・リーク防止のコードを書くときは、`tests/` にテストも書く。

## パス
- リポジトリ外: `../検討したデータセット/`(検討データの原本)、`../学部時代のGDELT研究/`(学部のデータ。使わない)。
