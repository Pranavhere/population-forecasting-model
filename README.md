# Atlanta MSA Population Forecast

Three different approaches to projecting population growth for the Atlanta metropolitan area (2025-2035).

## What's in here

- **popforecast.ipynb** — Main notebook with all three models, detailed explanations, and visualizations
- **Model 1** (`model1_extrapolation.py`) — Simple growth rate extrapolation
- **Model 2** (`model2_arima.py`) — ARIMA time series model
- **Model 3** (`model3_agebasedcohort.py`) — Cohort-component demographic model (recommended)
- **Data** — Historical population, births/deaths/migration, mortality rates

## Quick start

```bash
pip install -r requirements.txt
jupyter notebook popforecast.ipynb
```

## Models compared

| Model | Method | Best for |
|-------|--------|----------|
| 1 | Demographic rate extrapolation | Quick estimates |
| 2 | ARIMA time series | Short-term smoothing |
| 3 | Age-structured cohorts | Long-term planning |

Model 3 is the usual choice for demographic planning since it tracks age structure, fertility, mortality, and migration separately.

## Files

- `Final_Data_for_Modeling.xlsx` — Historical data (2000-2024)
- `population_forecast.xlsx` — Forecast outputs
- `*.html` — Model selection and methodology references
- `model*_forecast.png` — Individual model visualizations
- `model_comparison.png` — All three models side-by-side
