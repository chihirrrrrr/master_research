# data/raw/ — 採用データ(生データ)

**方針(US-1): 米国市場の「銘柄別ニュース + 市場全体ニュース + yfinance価格」でテキストの予測上の寄与を検証する。**
このディレクトリのファイルは**書き換えない**(ファイルは読み取り専用)。加工は `data/raw/` を入力にして、`data/interim/`、`data/processed/` に出力する。
データ本体はgit管理外(`.gitignore`)。このREADMEと `CHECKSUMS.txt` だけを追跡する。
整合性の確認: `python src/data/verify_raw.py`(`CHECKSUMS.txt` と照合する)。

## 出典(Kaggleのページを2026-10-05に確認)
| データ | Kaggleページ | 作者 | ライセンス |
|---|---|---|---|
| `news_price_events/` | https://www.kaggle.com/datasets/ibktommy/aggregated-financial-news-dataset<br>題名: *Financial News Market Impact & Volatility Dataset* | Anjorin Tomide Quadri (ibktommy) | **ODC Attribution License (ODC-By)** — 出典の明記が必要 |
| `nyt_headlines/` | https://www.kaggle.com/datasets/jonathanpoli/sp500-market-stock-prediction-with-nyt-headlines<br>題名: *S&P500 Market Stock Prediction with NYT headlines* | EPIDEIXX (jonathanpoli) | CC0: Public Domain(作者の表記) |

- 論文では、作者・題名・URL・アクセス日(2026-10-05)・ライセンスを引用する。NYT見出しの権利はNYT側に残る可能性があるので、出典としてNYTを明記する。
- `news_price_events` は、Kaggleの別データセット *Daily Financial News for 6000+ Stocks* の見出しを加工したもの(作者の説明)。**上流のデータのライセンスは未確認。**

## 内容
| パス | 中身 | 役割 |
|---|---|---|
| `news_price_events/news_price_events.csv` | 米国の銘柄別ニュース見出し(1時間窓、米東部時間)+ 終値 + 翌営業日の絶対リターン。219,971行、223銘柄(ETF含む)、2009-04-29〜2020-06-12 | 銘柄別テキスト(主データ) |
| `nyt_headlines/dataset_NYTimes.csv` | NYT「World」欄の日次見出し(人気順に連結)+ Label(S&P500の翌日の上げ=1/下げ=0)。3,522営業日、2008-01-02〜2021-12-30 | 市場全体テキスト(文脈) |
| `nyt_headlines/20XX.csv`(2008〜2021) | 上のカレンダー全日版(News列は同一)。5,109日 | 元データのまま保持 |
| `prices_yfinance/<取得日>/` | yfinanceの日次価格。`stocks.parquet`(220銘柄、743,535行)、`macro.parquet`(市場・マクロ27系列)、`FETCH_LOG.md`(取得日・版・取れなかった銘柄)。**2026-10-05取得**(`python -m src.data.fetch_prices`) | 数値データ(A0) |

## 作者の説明の要点(news_price_events)
- 見出しのタイムスタンプを米東部時間にそろえ、**寄り前のニュースは「当日の寄り付き(9:30)」、引け後のニュースは「次の取引セッション」に寄せる。**
- 同一銘柄の**1時間内の見出しを `|` で連結**する(窓が `time_window`)。完全に同じ見出しは除去済み。
- 価格はyfinanceの日次。各ニュースについて「**1日先の絶対リターン**」を計算し、**3つの等分位(各33.3%)**に分けてクラス0〜2とする。
- 作者が推奨する分割: 学習 2009〜2017 / 検証 2018〜2019 / テスト 2020(時間外)。
- 対象は「報道数の多い米国の流動性の高い上位300銘柄」と説明されている。

## 検証済みの注意点(加工時に必ず考慮)
**news_price_events**
- `forward_return` = `|Close(t+1)/Close(t) − 1|`(`data.csv`の実価格と相関1.000で確認)。**方向の情報はない。** `market_impact` はその三分位。`Close` は配当・分割調整済みで、始値・高値・安値・出来高はない。
- **ニュース当日の反応がラベルに入らない(説明とデータの両方から確認)。** 寄り前のニュースは当日 t の窓に入るが、ラベルは「t の終値 → t+1 の終値」。ニュースが影響する t 日の値動き(前日終値 → t の終値)は含まれない。→ 目的変数は価格から作り直し、「ニュース直後」の窓も試す。
- **説明とデータの不一致**:
  - クラスの境界: 説明は 0.81% / 1.94%、データの実測は約 0.61% / 1.60%。
  - 銘柄数: 説明は上位300、実際は223。AAPL / AMZN / MSFT / META / JPM / XOM / INTC / AMD / WMT は0行(「報道数の多い銘柄」と矛盾するが、理由は不明)。
- 同一(銘柄,日)に複数行(平均1.38窓)があり、ラベルを共有する → 日単位でtrain/testを分ける。
- 約34%(75,237行)の見出しが複数銘柄に共通の汎用見出し(「Stocks Hitting 52-Week Highs」等)。
- 2020年6月で終了。2018年以降を30%でテストにすると、2020年のコロナ急落が入り多数派が0.387になる。
- 前日の変動とラベルの順位相関が約0.32 → 価格のみのベースラインが強い。

**NYT見出し**
- 見出しは「World」欄のみ(経済・市場欄ではない)で、**市場への関連は間接的**。1日約84語、小文字化済み、本文なし。
- `Label` は S&P500 の**翌日**の上げ(1)/下げ(0)(作者の説明)。Kaggle DJIA の翌営業日Labelと約90%一致することを確認した(指数どうしの連動による)。→ `^GSPC` から作り直して照合する。「横ばい」の扱いは不明。

## 次の作業(未着手)
1. ~~yfinanceで価格を取得~~ → 済み(2026-10-05)。`close` は**分割調整済み・配当は未調整**、`adj_close` は配当・分割とも調整済み。取得できなかったのは AET・ESRX・TWX(買収された会社。ニュース3,131行=1.4%)。再取得するときは別の日付ディレクトリに保存し、`FETCH_LOG.md` と比べる。
2. 目的変数の定義を決める。
3. 日付・時刻の整合(引け前後の判定)を設計する。
