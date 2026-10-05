# よく使う操作。`make help` で一覧。
PYTHON ?= python3.12
VENV   := .venv
PY     := $(VENV)/bin/python

.PHONY: help env env-nlp verify test freeze fetch features

help:
	@echo "make env      .venv を作り、requirements.txt を入れる"
	@echo "make env-nlp  NLP用(torch, transformers)を追加で入れる"
	@echo "make verify   data/raw が CHECKSUMS.txt と一致するか確認"
	@echo "make test     テストを実行(raw の整合・データ契約)"
	@echo "make freeze   実験に使った版を requirements.lock.txt に固定"
	@echo "make fetch    yfinanceから価格を取得して data/raw/prices_yfinance/<日付>/ に保存(上書きしない)"
	@echo "make features 数値の特徴量と目的変数の候補を data/processed/ に作る"

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
