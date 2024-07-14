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

train train-dataset:
	python train.py --dataset $(train-dataset)


infer test-dataset model output:
	python infer.py --dataset $(test-dataset) --model $(model) --output $(output)

