# Dream11 Next-Gen Team Builder
Milestone 1 - Foundation & Safe Data

## 1. Project setup
Requirements are defined in `pyproject.toml` and pinned in `requirements.txt`.
`make` commands are available to run the pipeline.

## 2. Environment setup
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## 3. How to download data
```bash
make data
```
This triggers `src/ingestion/download.py` which downloads `t20s_json.zip` from cricsheet programmatically into `data/raw`.

## 4. How to parse data
`make data` automatically runs `src/ingestion/parse_cricsheet.py` to parse JSONs into `matches.parquet` and `deliveries.parquet`, strictly excluding matches after 2024-06-30.
Then `src/ingestion/normalise.py` normalizes team names.

## 5. How fantasy scoring works
`make features` runs `src/features/fantasy_points.py` which applies rules from `configs/scoring.yaml` to calculate T20 fantasy points, generating `player_match_stats.parquet`.

## 6. How to run tests
```bash
make test
```
Runs unit tests and leakage tests via pytest.

## 7. Data directory structure
- `data/raw/`: Contains downloaded zips and extracted JSONs.
- `data/processed/`: Contains clean Parquet analytical tables.

## 8. Leakage prevention approach
We use a `FeatureContext` class in `src/features/context.py` which acts as a "leakage wall". For any target match on date D, it structurally asserts and filters so only `date < D` data can pass through. A HARD_CUTOFF of `2024-06-30` is also enforced at the ingestion level.

## 9. Exact commands
- `make install`
- `make data`
- `make features`
- `make test`

## 10. How to launch the UI
The Streamlit app provides an interactive prediction interface and model evaluation insights.
Run the following command:
\\\ash
streamlit run src/ui/app.py
\\\
- **Product UI**: Select match date and teams to get the optimal pre-toss Dream11 Best XI and C/VC.
- **Model UI**: View canonical holdout evaluation metrics, SHAP global feature importances, and the system architecture.
- **Demo Mode**: The interface pre-loads a historical holdout match (2024-07-01, Malawi vs Kenya) as a valid demo out of the box.

