.PHONY: install test lint overfit train eval clean

install:
	pip install -r requirements.txt
	pip install -e .

test:
	pytest -q --cov=star --cov-report=term-missing

lint:
	ruff check src tests scripts
	black --check src tests scripts

# NOTE: on Kaggle use notebooks/kaggle_stage15_best3.ipynb; these targets are for local smoke runs.
overfit:
	python scripts/train.py --config configs/stage1_safe_warmstart.yaml --overfit-one-batch

train:
	python scripts/train.py --config configs/stage1_safe_warmstart.yaml

eval:
	python scripts/evaluate.py --config configs/stage1_safe_warmstart.yaml --ckpt checkpoints/best.pth

clean:
	rm -rf __pycache__ .pytest_cache .coverage htmlcov outputs/tmp
