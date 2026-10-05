# データセット理解と整理計画

> **【旧メモ・2026-10-05】** 学部のGDELTデータは、この計画では整理せず、リポジトリ外の `修士研究/学部時代のGDELT研究/` に**そのまま移した**(修士では使わない。docs/DECISIONS.md参照)。以下の整理計画と `MANIFEST.csv` の旧パス・新パス案は実行しておらず、**参考資料**として残している。
>
> 対象はCSVのみ(画像・.venv・.DS_Store等は整理対象外)。**この時点でファイルは1つも移動・削除していません。**
> 全件の一覧は `MANIFEST.csv`(旧パス / データセット名 / 処置 / 重複元 / 新パス案 / 行数 / 期間 / サイズ / md5)。
> 以下の「用途・関係」は列名・期間・フォルダ名・コード名からの推定を含みます。違う箇所は指摘してください。

## 1. 数字で見る現状

| | 件数 | サイズ |
|---|---|---|
| CSV総数 | 144 | - |
| 残すもの (keep) | 109 | 約4.1GB |
| 他と完全一致の重複 (duplicate) | 32 | **約3.8GB** |
| 空ファイル (empty) | 3 | 0 |

重複の大半は `192.168.10.119で作成したファイル/` と `makeTextCsv/` の間のコピーです。

## 2. データセットの全体像(7系統)

| # | データセット | 期間 | 中身 | 主なファイル | 行数規模 |
|---|---|---|---|---|---|
| A | **gdelt_2022_2025** (本命) | 2022-11-18〜2025-11 | GDELTニュースをS&P500辞書でセクター(GICS 11)に振り分け、日次×セクターで集約。FinBERT感情・埋め込み・価格特徴量・翌日up/downラベルまで | 下の§3 | 11,189行(日×セクター) / 学習用7,679行 |
| B | gdelt_2025_pilot | 2025-09〜11 | 上記の1か月パイロット版。記事本文、要約、極性統計、ETF別日次tone | `daily_top5_by_sector_with_text/summary`, `sector/*` | 〜1,528記事 / 230行 |
| C | gdelt_2023_prototype | 2023-09〜10 | 最初期のGDELT試作。クラスタリング・セクター付与の実験 | `SandP500/*`, `企業辞書作成/*` | 〜577行 |
| D | dictionaries | 2025-10 | S&P500企業辞書(ticker/社名/別名regex/GICS) | `sp500_aliases_guarded`(2,142行) ほか | 503〜2,142行 |
| E | aapl_newsapi | 2000〜2025-08 | AAPL単銘柄: 株価25年、NewsAPIニュース、VADER/FinBERT感情、step3データセット、XGBoost予測 | `prices_step1`, `dataset_AAPL_step3*` | 230〜6,266行 |
| F | sector_etf_xlk | 2024-12〜2025-09 | XLK(IT)の終値とup/downラベル | `it_prices_with_labels_long` | 196行 |
| G | djia_kaggle | 2008〜2016 | Kaggle「Reddit News × DJIA」(日ごとTop25見出し+Label)と、感情付与版、BERT埋め込み | `Combined_News_DJIA*`, `djia_bert_embeddings*` | 1,989行 |

学部の研究の主軸はA(と、その立ち上げ過程のB・C・D)で、E・F・Gは「手法検証の足がかり」の位置づけに見えます。

## 3. 本命データセットAのパイプライン(段階順)

```
01_articles   daily_top5_by_sector_2022-2025        53,661行  記事URL(日×セクター上位5)
              → ..._with_text_2022-2025              53,611行  本文取得(失敗含む)
              → ..._with_text_en_only                35,908行  英語のみ(最新版)
02_daily_text daily_concat_en_by_sector              11,189行  日×セクターで title/body を連結
03_sentiment  daily_with_finbert_scores / with_labels / sector_sentiment_with_updown   FinBERT positive/negative/neutral
05_embeddings finbert_embedding_with_updown_int      7,679行×776列  タイトルFinBERT埋め込み(emb_0..767)+ラベル
              fintransformer_body_embedding_...      7,679行×772列  本文FinTransformer埋め込み
04_price      daily_concat_..._with_price_features_safe   ret_1d/3d/5d, ma, vol, spy_ret のlag1特徴量
06_model_input embed_with_price_for_xgb_safe         7,679行×786列  XGBoost入力(タイトル埋め込み+価格)
              fintransformer_body_with_price_..._safe 7,679行×782列  XGBoost入力(本文埋め込み+価格)
prices        sector_prices_2022-2025_labeled        8,251行  セクターETF終値・ret・updown_int
results       xgb_*_importance, sector_xgb_pca_results  セクター別精度(accuracy/F1)など
```

ラベル(`updown_int`)について: 卒論§3-6-1では「前日終値→**当日**終値のclose-to-closeリターン > 0 を1」と定義されています。
一方、パイロット版のファイル名に `target_updown_t_plus_1` があり、コード上は翌日定義の可能性もあるため、**実データ(日付とretの対応)で検証が必要**です(§8)。

## 4. 見つかった問題点(データの中身)

1. **`_safe` と無印の2系統**: `*_safe.csv` はlag1特徴量で、未来情報リークを避けた修正版とみられます。無印は旧版→**修士で使うのは `_safe` を標準にするのが安全**。
2. **同名で中身が違うファイル(要判断)**
   - `daily_top5_by_sector_with_text_en_only_2022-2025.csv`: 35,799行(makeTextCsv/11:17)と35,908行(11:20以降)。後者が新しい版。
   - `embed_with_price_for_xgb.csv`: 5,743行・〜2025-02-11(makeTextCsv/1203)と7,679行・〜2025-11-14(.119/1203)。前者は途中までしかない旧版。
   - `gdelt_us_with_text.csv`: `SandP500/`(577行)と `企業辞書作成/`(500行)で別物。
   - `news_AAPL_step2.csv`: 3か所に別内容(161 / 137 / 0行)。
3. **`sector_prices*.csv` の `close` 列が11個重複**: 列名がセクターごとに付かず、`close, close.1 ... close.10` になっています(yfinanceのMultiIndex平坦化ミスとみられる)。どの列がどのETFか分からないので、修士用データ作成前に直したほうが良いです。
4. `finbert_with_updown_int_2022-2025.csv` は `sector` 列が2つあります。
5. `newsAPIdataset/` の2つのCSVは空(ヘッダなし)。`199/data/news_AAPL_step2.csv` もヘッダのみ。
6. `djia_bert_embeddings_separate.csv` は19,201列・467MBで、`news_embeddings_partial.csv`(250行)は途中で止まった残骸のようです。
7. データの終端が日付で揃っていません: テキスト系は〜2025-11-16、価格・埋め込み系は〜2025-11-14。

## 5. 新ディレクトリ構成案(CSVのみ)

```
data/
  gdelt_2022_2025/{01_articles,02_daily_text,03_sentiment,04_price_features,05_embeddings,06_model_input,prices}/
  gdelt_2025_pilot/
  gdelt_2023_prototype/
  dictionaries/
  aapl_newsapi/
  sector_etf_xlk/
  djia_kaggle/
  (修士用に追加するデータはここへ: data/master_* )
results/xgb/
archive/
  duplicates/     ← 完全一致の重複(32件、約3.8GB)を退避。確認後に削除
  empty_files/    ← 空ファイル3件
```
コード(.py)は今回は対象外のため `gdeltを使用した研究/` に残します(必要なら後で整理)。

## 6. 実行時の安全策(確認をもらったら)

- 先に現状を `git commit` してから、`git mv` で移動(いつでも戻せる)。
- 重複は削除せず `archive/duplicates/` へ。**消すのはあなたが確認した後**。
- 移動後に全ファイルのmd5をMANIFESTと照合して欠損が無いことを確認。
- 注意: 既存の `.py` は旧パスをハードコードしている可能性が高く、移動後は動かなくなります(学部の再現用なら、旧パス→新パスの対応表を渡せます)。

## 7. 確認したいこと

1. 重複32件の「正」の選び方: 現案は `192.168.10.119` 側を正、`makeTextCsv/` 側を重複扱いです。逆が良ければ指示してください。
2. §4-2の旧版(5,743行の `embed_with_price_for_xgb` など)は archive に退避でよいか。
3. データセットB(1か月パイロット)・C(2023試作)は残すか、archive行きにするか。
4. 修士で追加するデータは、Aを延長するのか、新しい系統(H)にするのか。

## 8. 卒業論文との対応(卒論PDFを読んで追記)

卒論: 「自然言語処理技術を用いたニューステキストからの株価上昇下降予測の有効性検証」(上智大・高岡研、2025年度)。
GDELTニュース(タイトル/本文)のみから、S&P500セクターETF(GICS 11)の日次up/downを予測。価格特徴量は**意図的に使わない**設計。

| 卒論での名称 | 該当データセット | 根拠 |
|---|---|---|
| 小規模データセット(2025-10-07〜11-05、1,528記事) | `gdelt_2025_pilot` | 記事数1,528が `daily_top5_by_sector_with_text` と一致 |
| 中規模データセット(2022-11-18〜2025-11-16、1,078日、35,799記事) | `gdelt_2022_2025` | 35,799行 = **`11:17に作ったファイル`版の en_only** と一致 |
| 企業辞書(§3-4-3) | `dictionaries` | `sp500_aliases_guarded`ほか |
| 実験の学習・評価(70/30時系列split、テスト2,304件) | `05_embeddings` の7,679行 | 7,679×0.3≒2,304、混同行列の合計と一致 |

### 卒論と照合して判明した重要点
1. **卒論が使った en_only は 35,799行版**(makeTextCsv/11:17)。35,908行版(11:20以降、`.119`側)は卒論後に再生成された別版。→ **どちらを「正」にするかの根拠になる。** 再現性を重視するなら35,799行版を正として残し、35,908行版は別名で併存。
2. **価格特徴量つきファイル(`04_price_features`, `06_model_input`、1203/1204フォルダ)は卒論の結果には出てこない**。卒論提出後(12月)に、課題(6)「数値特徴量とテキストの分離解析」に取り組んだ延長実験とみられる。修士の出発点として重要。
3. **卒論の各手法とデータの対応**: (A)FinBERT感情=`03_sentiment`、(B)埋め込み+PCA+XGBoost=`05_embeddings`、(C)ファインチューニング=`05_embeddings`の元テキスト。
4. 卒論の課題(1)〜(6)は、修士用データで直すべき点そのもの:
   - (1) **timestamp不在**: 現データはGDELTの日付(日単位)のみ。引け後記事の混入がありうる。修士用は公開時刻を保持して「前日引け後〜当日寄り前」で切る設計が必要。
   - (2) 日×セクターで上位k=5・同一ソース上限2に絞っている → 修士では記事単位のまま保持し、集約は後段で選べるようにするのが良い。
   - (3) 入力長切り詰め(512/1024トークン) → 記事単位データがあれば分割・重要文抽出が可能。
   - (6) 目的変数を翌日・異常リターン等にずらす案 → 価格データを生のOHLCVで持っておけばラベルを後から再定義できる。

### 8-1. 修士用データ整理への含意(提案)
- `data/gdelt_2022_2025/` の `01_articles`(記事単位・本文付き)が**最も価値の高い資産**。ここに公開時刻(GDELTのDATE=15分粒度のDATEADDED等)が残っているか、残っていなければBigQueryから再取得が必要か、を最初に確認する。
- ラベル(`updown_int`)・`sector_prices` は列名バグ(closeが11列)があるため、修士用では**yfinanceから生OHLCVを再取得**して `prices/` を作り直す方が確実。
- 卒論の結果を再現できる最小セットを `data/gdelt_2022_2025/` に固定し(35,799行版を使用)、修士の新規データは `data/master_*` に分けて混ざらないようにする。


## 9. 学部データの妥当性評価(実データ検証の結果)

**価格・ラベル側は正しい**: ret再計算との差1e-16、`updown_int == (同日ret>0)` が100%一致(卒論§3-6-1の定義どおり)。ETF対応も正しい。
軽微: 各セクター初日のret=NaNがlabel 0になっている(11行)、ret==0(58行)はDown扱い。

**テキスト側に問題あり(卒論の結論の解釈に影響)**
1. **企業同定の誤検出**: 最頻出tickerのSO(3,057件)は、本文/タイトルに社名(Southern Co等)が実在するのが2.3%のみ。DOW(1,217件)も6.2%。→ Utilities/Materials の記事の相当数がノイズ。`match_score` は全件1.0で、「関連度上位k件」は実質ランダム選択。
2. **週末・祝日のテキストを捨てている**: 日次テキスト11,189行のうち3,202行が週末で、学習用7,679行に入っていない(月曜入力に週末ニュースが入らない)。記事の28%が週末付け。
3. **同日対応付けによる事後報道混入**: ニュース日付(GDELT, 日単位)と同日のリターンを対応させている。
4. **ベースライン未比較**: テスト(2,304件)のUp率=0.5395。卒論の全手法のAccuracy(0.46〜0.54)は多数派ベースライン以下か同等。「FinBERTタイトル微調整 0.539」は全件Up予測そのもの。
5. **サンプルが独立でない**: 同日の11セクターは強く連動(日別up率のstd=0.31、独立なら約0.15)。train/test境界(2024-12-19)が同一日をまたぐ。
6. 軽微: 本文欠損・極短 151件、ボイラープレート疑い 685件、同一title 2,872件、本文完全重複45件。
