.PHONY: install data features test

install:
	pip install -r requirements.txt

data:
	python -m src.ingestion.download
	python -m src.ingestion.parse_cricsheet
	python -m src.ingestion.normalise

features:
	python -m src.features.fantasy_points

test:
	pytest tests/
