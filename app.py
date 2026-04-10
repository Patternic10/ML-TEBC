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

st.set_page_config(page_title="Materials Screening Copilot for TBCs", layout="wide")

@st.cache_resource(show_spinner=False)
def _cached_kappa_artifacts(model_dir: str):
    return load_kappa_artifacts(model_dir)


@st.cache_resource(show_spinner=False)
def _cached_cte_artifacts(model_dir: str):
    return load_cte_artifacts(model_dir)


@st.cache_data(show_spinner=False, max_entries=128)
def _cached_predict_kappa(compositions_key: tuple[str, ...], model_dir: str, tmin: int, tmax: int, step: int):
    artifacts = _cached_kappa_artifacts(model_dir)
    return predict_kappa(list(compositions_key), artifacts, tmin=tmin, tmax=tmax, step=step)


@st.cache_data(show_spinner=False, max_entries=128)
def _cached_predict_cte(compositions_key: tuple[str, ...], model_dir: str, tmin: int, tmax: int, step: int):
    artifacts = _cached_cte_artifacts(model_dir)
    return predict_cte(list(compositions_key), artifacts, tmin=tmin, tmax=tmax, step=step)


st.title("Materials Screening Copilot for TBCs")
st.caption("Predict kappa/CTE, screen candidates, and build a final shortlist.")

if "kappa_predictions" not in st.session_state:
    st.session_state.kappa_predictions = None
if "kappa_ranked" not in st.session_state:
    st.session_state.kappa_ranked = None
if "cte_predictions" not in st.session_state:
    st.session_state.cte_predictions = None
if "failed_compositions" not in st.session_state:
    st.session_state.failed_compositions = None
if "plot_compositions" not in st.session_state:
    st.session_state.plot_compositions = []

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
    with st.form("kappa_prediction_form", clear_on_submit=False):
        run_kappa = st.form_submit_button("Run Prediction")

    if run_kappa:
        try:
            compositions = parse_compositions_from_sources(uploaded_comp_df, typed_comp)
            with st.spinner("Running kappa featurization + prediction..."):
                pred_df, failed_df = _cached_predict_kappa(tuple(compositions), kappa_model_dir, 100, 2000, 100)

            st.session_state.kappa_predictions = pred_df
            st.session_state.failed_compositions = failed_df

            st.success(f"Generated {len(pred_df)} kappa prediction rows for {pred_df['Composition'].nunique()} compositions.")
        except FileNotFoundError as exc:
            st.error(f"Artifact path error: {exc}")
        except ArtifactError as exc:
            st.error(str(exc))
        except ValueError as exc:
            st.error(str(exc))
        except Exception as exc:  # noqa: BLE001
            st.exception(exc)

    if st.session_state.kappa_predictions is not None:
        st.dataframe(st.session_state.kappa_predictions.head(30), use_container_width=True)
        _download_df("Download kappa predictions CSV", st.session_state.kappa_predictions, "kappa_predictions.csv")

    if st.session_state.failed_compositions is not None and not st.session_state.failed_compositions.empty:
        st.warning("Some compositions failed parsing or featurization.")
        st.dataframe(st.session_state.failed_compositions, use_container_width=True)

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
            st.session_state.plot_compositions = ranked_df["Composition"].head(min(5, len(ranked_df))).tolist()

            st.dataframe(ranked_df, use_container_width=True)
            _download_df("Download ranked kappa CSV", ranked_df, f"screened_kappa_{int(target_t)}K.csv")
        except ValueError as exc:
            st.error(str(exc))
        except Exception as exc:  # noqa: BLE001
            st.exception(exc)

    if st.session_state.kappa_ranked is not None:
        plot_options = st.session_state.kappa_ranked["Composition"].tolist()
        selected = st.multiselect(
            "Select compositions to plot kappa vs T",
            options=plot_options,
            default=[c for c in st.session_state.plot_compositions if c in plot_options],
            key="kappa_plot_select",
        )
        st.session_state.plot_compositions = selected
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
        with st.form("cte_prediction_form", clear_on_submit=False):
            run_cte = st.form_submit_button("Run Prediction")
        if run_cte:
            try:
                compositions = parse_compositions_from_sources(uploaded_comp_df, typed_comp)
                with st.spinner("Running CTE featurization + prediction..."):
                    cte_df, failed_df = _cached_predict_cte(tuple(compositions), cte_model_dir, 100, 2000, 100)
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

    if st.session_state.cte_predictions is not None:
        st.dataframe(st.session_state.cte_predictions.head(30), use_container_width=True)
        _download_df("Download CTE predictions CSV", st.session_state.cte_predictions, "cte_predictions.csv")

with tab4:
    st.subheader("Final shortlist (kappa + CTE)")
    cte_threshold = st.number_input("Maximum acceptable CTE", value=10.0, step=0.1)

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
            )

            if shortlist_df.empty:
                st.warning("No candidates met the CTE threshold after merging by Composition and T.")
            else:
                st.success(f"Final shortlist contains {len(shortlist_df)} candidates.")
                st.dataframe(shortlist_df, use_container_width=True)
                _download_df("Download final shortlist CSV", shortlist_df, "final_shortlist.csv")
        except ValueError as exc:
            st.error(str(exc))
        except Exception as exc:  # noqa: BLE001
            st.exception(exc)
