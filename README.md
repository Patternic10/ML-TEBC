# Materials Screening Copilot for Thermal & Environmental Barrier Coatings

## What the app does

This Streamlit app helps screen **rare earth phosphate (REPO4)** candidates for thermal/environmental barrier coating applications using saved machine learning artifacts.  
It predicts thermal conductivity (**kappa**) and thermal expansion (**CTE**), then combines both to generate a final shortlist under EBC/TBC constraints.

## Why I built it

TEBC material selection is multi-objective and time-consuming.  
I built this app to make candidate screening faster, reproducible, and transparent by giving researchers a guided workflow from prediction to shortlist creation.

## How it was built with Codex

This project was iterated and refactored with OpenAI Codex assistance to:
- organize reusable prediction/screening helpers,
- streamline the Streamlit workflow,
- improve validation and user-facing error messages,
- and add contest-focused UX polish.

## Main features

1. **Predict kappa**
   - Accepts `Composition` input via CSV upload and/or typed formula.
   - Predicts `kappa` over **100 K to 2000 K** (or a single temperature).

2. **Screen kappa**
   - Ranks compositions by lowest predicted `kappa` at a selected temperature.
   - Supports Top-N selection and interactive `kappa vs T` plotting.

3. **Check CTE**
   - Upload CTE prediction CSV (`Composition`, `T`, and `cte_pred` or `GBR_Pred`), or
   - generate CTE predictions from saved artifacts.

4. **Final shortlist**
   - Merges kappa-screened candidates with CTE predictions on `Composition` and `T`.
   - Filters for:
     - `EBC`: `CTE <= user max`
     - `TBC`: `CTE >= user min`
   - Ranks passing candidates using a transparent `Screening_Score`.

## Example use case

A researcher ranks **REPO4 candidate compositions** for EBC/TBC design constraints by:
1) predicting kappa,  
2) screening low-kappa candidates at 1500 K,  
3) checking/generating CTE,  
4) producing a final shortlist that satisfies CTE limits.

## Model artifact note

The app uses **saved ML artifacts** from:
- `Prod_Kappa/kappa_rf_models`
- `Prod_CTE/gbr_model`

During prediction/screening, models are **loaded and used directly**; they are **not retrained** in the app.

## How to run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deployment compatibility note

The saved sklearn artifacts were serialized with scikit-learn 1.4.x internals.  
For deployment, use:
- `scikit-learn==1.4.2` (already pinned in `requirements.txt`)
- `numpy<2.0` (already pinned in `requirements.txt` for sklearn 1.4.x compatibility)
- Python 3.11 (set in `runtime.txt`)

This avoids deserialization errors such as:
`AttributeError: ... __pyx_unpickle_CyHalfSquaredError`.
