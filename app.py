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

if "kappa_predictions" not in st.session_state:
    st.session_state.kappa_predictions = None
if "kappa_ranked" not in st.session_state:
    st.session_state.kappa_ranked = None
if "cte_predictions" not in st.session_state:
    st.session_state.cte_predictions = None
if "failed_compositions" not in st.session_state:
    st.session_state.failed_compositions = None

with st.sidebar:
    st.header("Model Artifact Paths")
    kappa_model_dir = st.text_input("Kappa model directory", value="Prod_Kappa/kappa_rf_models")
    cte_model_dir = st.text_input("CTE model directory", value="Prod_CTE/gbr_model")

    st.header("Input Compositions")
    uploaded_comp_file = st.file_uploader("Upload CSV with Composition column", type=["csv"])
    typed_comp = st.text_input("Or type single composition", placeholder="e.g., Lu0.25Y0.25Er0.25Yb0.25PO4")

    uploaded_comp_df = None
    if uploaded_comp_file is not None:
        uploaded_comp_df = pd.read_csv(uploaded_comp_file)
        st.write(f"Loaded {len(uploaded_comp_df)} rows from uploaded composition CSV.")


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
            artifacts = load_kappa_artifacts(kappa_model_dir)
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
        except FileNotFoundError as exc:
            st.error(f"Artifact path error: {exc}")
        except ArtifactError as exc:
            st.error(str(exc))
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
                st.error(str(exc))
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
                artifacts = load_cte_artifacts(cte_model_dir)
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
            except FileNotFoundError as exc:
                st.error(f"Artifact path error: {exc}")
            except ArtifactError as exc:
                st.error(str(exc))
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
                st.warning(f"No candidates met the {cte_mode} CTE threshold after merging by Composition and T.")
            else:
                st.success(f"Final shortlist contains {len(shortlist_df)} candidates.")
                st.dataframe(shortlist_df, use_container_width=True)
                _download_df("Download final shortlist CSV", shortlist_df, "final_shortlist.csv")
        except ValueError as exc:
            st.error(str(exc))
        except Exception as exc:  # noqa: BLE001
            st.exception(exc)
