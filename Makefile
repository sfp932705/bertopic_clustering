export PYTHONPATH=$(PWD)/src

format:
	ruff format .
	ruff check . --select I --fix

lint: lint-ruff lint-mypy

lint-ruff:
	ruff check src
	ruff format --check src

lint-mypy:
	mypy src

install:
	pip install -q -r requirements.txt

train dataset:
	python train.py --dataset $(dataset)

