# よく使う操作。`make help` で一覧。
PYTHON ?= python3.12
VENV   := .venv
PY     := $(VENV)/bin/python

.PHONY: help env env-nlp verify test freeze fetch features market-base market-features nyt-features study1-dev study1-all

help:
	@echo "make env      .venv を作り、requirements.txt を入れる"
	@echo "make env-nlp  NLP用(torch, transformers)を追加で入れる"
	@echo "make verify   data/raw が CHECKSUMS.txt と一致するか確認"
	@echo "make test     テストを実行(raw の整合・データ契約)"
	@echo "make freeze   実験に使った版を requirements.lock.txt に固定"
	@echo "make fetch    yfinanceから価格を取得して data/raw/prices_yfinance/<日付>/ に保存(上書きしない)"
	@echo "make features 数値の特徴量と目的変数の候補を data/processed/ に作る"
	@echo "make market-base 研究1の土台の表(S&P500のOHLC + 金利。NYTの期間)を data/interim/ に作る"
	@echo "make market-features 研究1の日単位の説明変数18個と目的変数を data/processed/ に作る"
	@echo "make nyt-features NYT見出しをFinBERTで感情の数値にする(torch・transformers・FinBERTが必要)"
	@echo "make study1-dev 研究1を開発期間で実行(数値のみ と +NYT を、reg1・reg2で比較。最終確認は実行しない)"

env:
	$(PYTHON) -m venv $(VENV)
	$(PY) -m pip install -U pip
	$(PY) -m pip install -r requirements.txt

env-nlp:
	$(PY) -m pip install -r requirements-nlp.txt

verify:
	$(PY) src/data/verify_raw.py

test:
	$(PY) -m pytest -q

freeze:
	$(PY) -m pip freeze > requirements.lock.txt

fetch:
	$(PY) -m src.data.fetch_prices

features:
	$(PY) -m src.features.numeric

market-base:
	$(PY) -m src.data.make_market_base

market-features:
	$(PY) -m src.features.market_level

nyt-features:
	$(PY) -m src.features.nyt_text

study1-dev:
	# 数値のみ と +NYT を、同じ行・同じ分割で、正則化の強さ2つ(reg1, reg2)で比べる(開発期間のみ。最終確認は実行しない)
	$(PY) -m src.models.study1 --features numeric --periods dev --model-config configs/study1_xgb_reg1.yaml --name study1_numeric_reg1_dev
	$(PY) -m src.models.study1 --features numeric+nyt --periods dev --model-config configs/study1_xgb_reg1.yaml --name study1_numeric_nyt_reg1_dev
	$(PY) -m src.models.study1 --features numeric --periods dev --model-config configs/study1_xgb_reg2.yaml --name study1_numeric_reg2_dev
	$(PY) -m src.models.study1 --features numeric+nyt --periods dev --model-config configs/study1_xgb_reg2.yaml --name study1_numeric_nyt_reg2_dev

study1-all:
	# 2013〜2021の全部(最終確認の期間を開ける)。事前登録は docs/DECISIONS.md。4つの組み合わせ
	$(PY) -m src.models.study1 --features numeric --periods all --final --model-config configs/study1_xgb_reg1.yaml --name study1_numeric_reg1_all
	$(PY) -m src.models.study1 --features numeric+nyt --periods all --final --model-config configs/study1_xgb_reg1.yaml --name study1_numeric_nyt_reg1_all
	$(PY) -m src.models.study1 --features numeric --periods all --final --model-config configs/study1_xgb_reg2.yaml --name study1_numeric_reg2_all
	$(PY) -m src.models.study1 --features numeric+nyt --periods all --final --model-config configs/study1_xgb_reg2.yaml --name study1_numeric_nyt_reg2_all
