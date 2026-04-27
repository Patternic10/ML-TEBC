from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from tbc_helpers import (
    ArtifactError,
    build_final_shortlist,
    coerce_cte_input,
    load_cte_artifacts,
    load_kappa_artifacts,
    parse_compositions_from_sources,
    predict_cte,
    predict_kappa,
    rank_lowest_kappa_at_temperature,
)

st.set_page_config(page_title="Materials Screening Copilot for TEBCs", layout="wide")
st.title("Materials Screening Copilot for TEBCs")
st.caption("Predict kappa/CTE, screen candidates, and build a final shortlist.")


@st.cache_resource(show_spinner=False)
def _cached_load_kappa_artifacts(model_dir: str):
    return load_kappa_artifacts(model_dir)


@st.cache_resource(show_spinner=False)
def _cached_load_cte_artifacts(model_dir: str):
    return load_cte_artifacts(model_dir)


if "kappa_predictions" not in st.session_state:
    st.session_state.kappa_predictions = None
if "kappa_ranked" not in st.session_state:
    st.session_state.kappa_ranked = None
if "cte_predictions" not in st.session_state:
    st.session_state.cte_predictions = None
if "failed_compositions" not in st.session_state:
    st.session_state.failed_compositions = None

with st.sidebar:
    st.header("About this app")
    st.markdown(
        """
This app predicts thermal conductivity (**kappa**) and thermal expansion (**CTE**) for phosphate-based candidate materials,
then combines both signals to shortlist candidates for thermal/environmental barrier coating design.

Why it matters: TEBC materials must balance low thermal conductivity with suitable thermal expansion to improve durability
under high-temperature service conditions.

Built for the Handshake/OpenAI Codex Creator Challenge: Codex was used to help refactor the workflow into reusable helper
functions, strengthen validation, and streamline the screening UI.
"""
    )

    st.header("Model Artifact Paths")
    kappa_model_dir = st.text_input("Kappa model directory", value="Prod_Kappa/kappa_rf_models")
    cte_model_dir = st.text_input("CTE model directory", value="Prod_CTE/gbr_model")

    st.header("Input Compositions")
    sample_input_df = pd.DataFrame(
        {
            "Composition": [
                "Sc0.2Lu0.2Yb0.2Y0.2Gd0.2PO4",
                "Lu0.5Ho0.5PO4",
                "La0.2Sm0.2Gd0.2Dy0.2Nd0.2PO4",
            ]
        }
    )
    st.download_button(
        label="Download sample input CSV",
        data=sample_input_df.to_csv(index=False),
        file_name="sample_compositions.csv",
        mime="text/csv",
        help="Use this template to upload compositions for prediction.",
    )
    uploaded_comp_file = st.file_uploader("Upload CSV with Composition column", type=["csv"])
    typed_comp = st.text_input("Or type single composition", placeholder="e.g., Lu0.25Y0.25Er0.25Yb0.25PO4")

    uploaded_comp_df = None
    if uploaded_comp_file is not None:
        try:
            uploaded_comp_df = pd.read_csv(uploaded_comp_file)
            if "Composition" not in uploaded_comp_df.columns:
                st.error(
                    "Uploaded composition CSV is missing the required 'Composition' column. "
                    "Please upload a file with a 'Composition' header."
                )
                uploaded_comp_df = None
            else:
                st.write(f"Loaded {len(uploaded_comp_df)} rows from uploaded composition CSV.")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Unable to read uploaded composition CSV: {exc}")

    st.divider()
    st.caption("Built with Codex (GPT-5.3-Codex) for rapid ML app development and refactoring.")


def _download_df(label: str, df: pd.DataFrame, file_name: str):
    st.download_button(
        label=label,
        data=df.to_csv(index=False),
        file_name=file_name,
        mime="text/csv",
    )


tab1, tab2, tab3, tab4 = st.tabs(["Predict kappa", "Screen kappa", "Check CTE", "Final shortlist"])

with tab1:
    st.subheader("Predict kappa from 100 K to 2000 K")
    kappa_temp_mode = st.radio(
        "Kappa prediction temperature mode",
        ["Full range (100–2000 K, step 100 K)", "Single temperature"],
        horizontal=True,
    )
    if kappa_temp_mode == "Single temperature":
        kappa_single_t = st.number_input(
            "Single kappa prediction temperature (K)",
            min_value=100,
            max_value=2000,
            value=1500,
            step=100,
            key="kappa_single_t",
        )
        kappa_tmin = int(kappa_single_t)
        kappa_tmax = int(kappa_single_t)
        kappa_step = 100
    else:
        kappa_tmin = 100
        kappa_tmax = 2000
        kappa_step = 100

    if st.button("Run kappa prediction"):
        try:
            compositions = parse_compositions_from_sources(uploaded_comp_df, typed_comp)
            artifacts = _cached_load_kappa_artifacts(kappa_model_dir)
            pred_df, failed_df = predict_kappa(
                compositions,
                artifacts,
                tmin=kappa_tmin,
                tmax=kappa_tmax,
                step=kappa_step,
            )

            st.session_state.kappa_predictions = pred_df
            st.session_state.failed_compositions = failed_df

            st.success(f"Generated {len(pred_df)} kappa prediction rows for {pred_df['Composition'].nunique()} compositions.")
            st.dataframe(pred_df.head(30), use_container_width=True)
            _download_df("Download kappa predictions CSV", pred_df, "kappa_predictions.csv")

            if failed_df is not None and not failed_df.empty:
                st.warning("Some compositions failed parsing or featurization.")
                st.dataframe(failed_df, use_container_width=True)
        except ArtifactError as exc:
            st.error(
                "Kappa model artifacts are missing or invalid. "
                f"Check the directory '{kappa_model_dir}'. Details: {exc}"
            )
        except ValueError as exc:
            st.error(str(exc))
        except Exception as exc:  # noqa: BLE001
            st.exception(exc)

with tab2:
    st.subheader("Screen lowest kappa at a chosen temperature")
    target_t = st.number_input("Screening temperature (K)", min_value=100, max_value=2000, value=1500, step=100)
    top_n = st.number_input("Top N", min_value=1, max_value=500, value=10, step=1)

    if st.button("Rank lowest-kappa candidates"):
        try:
            if st.session_state.kappa_predictions is None:
                raise ValueError("No kappa predictions available. Run 'Predict kappa' first.")

            ranked_df = rank_lowest_kappa_at_temperature(st.session_state.kappa_predictions, int(target_t), int(top_n))
            st.session_state.kappa_ranked = ranked_df

            st.dataframe(ranked_df, use_container_width=True)
            _download_df("Download ranked kappa CSV", ranked_df, f"screened_kappa_{int(target_t)}K.csv")

            selected = st.multiselect(
                "Select compositions to plot kappa vs T",
                options=ranked_df["Composition"].tolist(),
                default=ranked_df["Composition"].head(min(5, len(ranked_df))).tolist(),
            )
            if selected:
                plot_df = st.session_state.kappa_predictions[
                    st.session_state.kappa_predictions["Composition"].isin(selected)
                ]
                fig = px.line(
                    plot_df,
                    x="T",
                    y="kappa_pred",
                    color="Composition",
                    markers=True,
                    title="kappa vs Temperature",
                )
                st.plotly_chart(fig, use_container_width=True)
        except ValueError as exc:
            st.error(str(exc))
        except Exception as exc:  # noqa: BLE001
            st.exception(exc)

with tab3:
    st.subheader("Check CTE")
    st.caption("Upload existing CTE predictions or generate from saved CTE model artifacts.")
    cte_mode = st.radio("CTE source", ["Use saved CTE model", "Upload CTE predictions CSV"], horizontal=True)

    if cte_mode == "Upload CTE predictions CSV":
        uploaded_cte = st.file_uploader("Upload CTE predictions CSV", type=["csv"], key="cte_upload")
        if uploaded_cte is not None:
            try:
                cte_df_raw = pd.read_csv(uploaded_cte)
                cte_df = coerce_cte_input(cte_df_raw)
                st.session_state.cte_predictions = cte_df
                st.success(f"Loaded {len(cte_df)} CTE rows from file.")
                st.dataframe(cte_df.head(30), use_container_width=True)
                _download_df("Download normalized CTE CSV", cte_df, "cte_predictions_normalized.csv")
            except ValueError as exc:
                st.error(
                    "Invalid CTE CSV format. Required columns are 'Composition', 'T', and one of "
                    "'cte_pred' or 'GBR_Pred'. "
                    f"Details: {exc}"
                )
            except Exception as exc:  # noqa: BLE001
                st.exception(exc)
    else:
        cte_temp_mode = st.radio(
            "CTE prediction temperature mode",
            ["Full range (100–2000 K, step 100 K)", "Single temperature"],
            horizontal=True,
        )
        if cte_temp_mode == "Single temperature":
            cte_single_t = st.number_input(
                "Single CTE prediction temperature (K)",
                min_value=100,
                max_value=2000,
                value=1500,
                step=100,
                key="cte_single_t",
            )
            cte_tmin = int(cte_single_t)
            cte_tmax = int(cte_single_t)
            cte_step = 100
        else:
            cte_tmin = 100
            cte_tmax = 2000
            cte_step = 100

        if st.button("Run CTE prediction"):
            try:
                compositions = parse_compositions_from_sources(uploaded_comp_df, typed_comp)
                artifacts = _cached_load_cte_artifacts(cte_model_dir)
                cte_df, failed_df = predict_cte(
                    compositions,
                    artifacts,
                    tmin=cte_tmin,
                    tmax=cte_tmax,
                    step=cte_step,
                )
                st.session_state.cte_predictions = cte_df

                st.success(f"Generated {len(cte_df)} CTE prediction rows for {cte_df['Composition'].nunique()} compositions.")
                st.dataframe(cte_df.head(30), use_container_width=True)
                _download_df("Download CTE predictions CSV", cte_df, "cte_predictions.csv")

                if failed_df is not None and not failed_df.empty:
                    st.warning("Some compositions failed in CTE prediction.")
                    st.dataframe(failed_df, use_container_width=True)
            except ArtifactError as exc:
                st.error(
                    "CTE model artifacts are missing or invalid. "
                    f"Check the directory '{cte_model_dir}'. Details: {exc}"
                )
            except ValueError as exc:
                st.error(str(exc))
            except Exception as exc:  # noqa: BLE001
                st.exception(exc)

with tab4:
    st.subheader("Final shortlist (kappa + CTE)")
    cte_mode = st.radio(
        "CTE shortlist mode",
        ["EBC", "TBC"],
        horizontal=True,
        help="EBC applies a maximum acceptable CTE. TBC applies a minimum acceptable CTE.",
    )
    cte_threshold_label = "Maximum acceptable CTE" if cte_mode == "EBC" else "Minimum acceptable CTE"
    cte_threshold = st.number_input(cte_threshold_label, value=10.0, step=0.1)
    st.caption(
        "Scoring rule: Screening_Score = predicted kappa at the selected screening temperature "
        "(lower is better). Final list is filtered by CTE threshold, then ranked by Screening_Score."
    )

    if st.button("Build final shortlist"):
        try:
            if st.session_state.kappa_ranked is None:
                raise ValueError("No ranked kappa candidates found. Run 'Screen kappa' first.")
            if st.session_state.cte_predictions is None:
                raise ValueError("No CTE predictions available. Use 'Check CTE' first.")

            shortlist_df = build_final_shortlist(
                st.session_state.kappa_ranked,
                st.session_state.cte_predictions,
                float(cte_threshold),
                cte_mode,
            )

            if shortlist_df.empty:
                st.warning(
                    f"No candidates passed the final {cte_mode} filter at threshold {cte_threshold}. "
                    "Try relaxing the CTE threshold or screening more candidates in 'Screen kappa'."
                )
            else:
                st.success(f"Final shortlist contains {len(shortlist_df)} candidates.")
                st.dataframe(shortlist_df, use_container_width=True)
                _download_df("Download final shortlist CSV", shortlist_df, "final_shortlist.csv")
        except ValueError as exc:
            st.error(str(exc))
        except Exception as exc:  # noqa: BLE001
            st.exception(exc)
