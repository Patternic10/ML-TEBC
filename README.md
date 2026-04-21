# ML-TEBC

Machine learning workflows for Thermal and Environmental Barrier Coatings, now with a Streamlit interface:

## Materials Screening Copilot for TEBCs

This app provides an end-to-end workflow with four tabs:

1. **Predict kappa**
   - Accepts compositions from:
     - a CSV upload with a `Composition` column, and/or
     - a single typed composition.
   - Uses saved Random Forest artifacts under `Prod_Kappa/kappa_rf_models`.
   - Predicts `kappa` from **100 K to 2000 K** in **100 K** steps.
   - Supports CSV download of prediction results.

2. **Screen kappa**
   - Ranks compositions by **lowest predicted kappa** at a selected temperature.
   - Default screening temperature: **1500 K**.
   - Lets you choose **Top N** candidates.
   - Includes an interactive `kappa vs T` plot for selected compositions.
   - Supports CSV download of ranked output.

3. **Check CTE**
   - Supports either:
     - Uploading a CTE prediction CSV (`Composition`, `T`, and `cte_pred` or `GBR_Pred`), or
     - Generating CTE predictions from saved GBR artifacts under `Prod_CTE/gbr_model`.
   - Supports CSV download of normalized/generated CTE results.

4. **Final shortlist**
   - Merges kappa-screened candidates with CTE results on `Composition` and `T`.
   - Supports mode-based filtering:
     - `EBC`: keep candidates at or below a user-defined maximum acceptable CTE.
     - `TBC`: keep candidates at or above a user-defined minimum acceptable CTE.
   - Supports CSV download of final shortlist.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Notes

- The app includes robust validation/error handling for:
  - missing model artifacts,
  - missing required CSV columns,
  - no predictions available before screening/shortlisting,
  - invalid/failed composition featurization.
- Reusable logic is factored into `tbc_helpers.py`.
