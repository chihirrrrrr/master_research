# CLAUDE.md

修士研究のリポジトリ。応答は日本語で行う。概要は README.md、方針は docs/RESEARCH_PLAN.md、決定の履歴は docs/DECISIONS.md、運用ルールは docs/OPERATIONS.md を参照。

## 研究の前提
- 問い: 金融ニュースのテキストを、価格由来の数値特徴量に加えると、株価の予測は価格のみより改善するか。
- 分析は2段階(RESEARCH_PLAN.md §5): 研究1=市場レベル(S&P500、A0m vs +NYT)、研究2=銘柄レベル(A0〜A3)。NYTは銘柄情報を持たないので市場全体で検証する。
- 学部との違い: テキストのみ → テキスト + 数値。目的変数は**リターン・超過リターン(翌営業日から5営業日 t+1〜t+5 の平均。t は含まない)**に決定(RESEARCH_PLAN.md §4、定義は docs/VARIABLES.md)。予測の起点は取引日 t の引け後で、説明変数は当日(t)のニュースと数値(過去の履歴を含む)。主モデルは XGBoost、第2は系列モデルの GRU(後で。LSTM・RNNは使わない)。分割(テスト)は**保留**。
- 採用データ(US-1): `data/raw/news_price_events`(米国223銘柄、時刻つき見出し、ODC-By)+ `data/raw/nyt_headlines`(NYT「World」欄。補助)+ yfinance価格(取得済み: `data/raw/prices_yfinance/20261005/`)。数値の特徴量(A0)と目的変数の候補は `make features` で `data/processed/` に作る(25個。`src/features/numeric.py`)。出典は data/raw/README.md。

## ルール
- `data/raw/` は**書き換えない**(読み取り専用)。加工は `src/` のスクリプトで行い、`data/interim/` → `data/processed/` に出力する。
- データ本体と機密(`.env`、鍵JSON)は**コミットしない**。`.gitignore` で除外済み。pushはユーザーの指示があるときだけ行う。
- 大きなCSVは全体を表示しない。`head`・サンプル・集計を使い、結果はファイルに書く。
- 予測の評価は、時間順の検証と、数値のみ(A0)とのベースライン比較を必須とする(分割・評価指標の細部は保留)。
- リークを避ける: ニュースの時刻 → 取引日の対応は、引け後・週末を次営業日に寄せる(詳細は RESEARCH_PLAN.md §6)。
- 既存ラベル(`forward_return`、NYTの`Label`)はそのまま使わず、価格から作り直す(理由は data/raw/README.md)。
- ニュースは取引日 t の行(9〜15時の窓)を使い、t の16:00以降の437行は除外する(`original_first_date` で判定。分まで残っている。`time_window` は1時間に丸められている)。
- 最終確認(confirm)の期間は、`--final` を付けたときだけ評価する(最後に1回だけ見る)。開発(dev)の期間で、設定を決める。
- 重要な決定は docs/DECISIONS.md に日付つきで追記する。

## 操作
- 環境: `make env`(`.venv`)、`make verify`(rawの整合)、`make test`、`make freeze`。実行は `.venv/bin/python -m src.…`。
- 1実験 = `results/runs/YYYYMMDD_名前/`(config・metrics・notes。`results/runs/README.md`)。分割は保留(`configs/split.yaml` は暫定値で今は使わない)。
- yfinanceの価格は `data/raw/prices_yfinance/` に、取得日・版を `FETCH_LOG.md` に記録して保存し、上書きしない。
- 時刻整合・リーク防止のコードを書くときは、`tests/` にテストも書く。

## パス
- リポジトリ外: `../検討したデータセット/`(検討データの原本)、`../学部時代のGDELT研究/`(学部のデータ。使わない)。
