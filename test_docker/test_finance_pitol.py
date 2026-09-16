#!/usr/bin/env python3
import argparse
import datetime as dt
import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlflow
import numpy as np
import pandas as pd

try:
    from tabpfn import TabPFNRegressor
except ImportError:
    TabPFNRegressor = None

from certain_library.log_basic.log_param import log_param
from certain_library.train_monitor.log_metrics import log_metrics
from certain_library.train_monitor.log_model import (
    log_model_info,
    log_model_hyperparameters,
)
from certain_library.data_analysis.log_dataset import (
    log_dataset,
    log_train_test_dataset,
)
from certain_library.data_analysis.log_whylogs import log_whylogs_profile
from certain_library.data_analysis.log_timeseries import timestamp_analysis
from certain_library.resource_monitor.resource import start_tracker, stop_tracker

from finance_pilot.utils.constants import (
    DEFAULT_TIMESTAMP_COL,
    DEFAULT_ITEM_COL,
    DEFAULT_RATING_COL,
    DEFAULT_USER_COL,
)

from finance_pilot.algorithms.kpi_gen.load_kpi_generator import LoadKPIGenerator
from finance_pilot.algorithms.kpi_gen.ma_kpi_generator import MAKPIGenerator
from finance_pilot.algorithms.profitability_prediction import ProfitabilityPrediction
from finance_pilot.algorithms.rfr_kpi_model import RFRKPIModel

from finance_pilot.data.filter.asset.asset_with_test_price import AssetWithTestPrice
from finance_pilot.data.filter.customer.customer_in_train import CustomerInTrain
from finance_pilot.data.filter.data_filter import DataFilter
from finance_pilot.data.filter.rating.ratings_not_in_train import RatingsNotInTrain
from finance_pilot.data.filter.timeseries.no_filter import NoFilter

from finance_pilot.data.financial_asset_time_series import FinancialAssetTimeSeries
from finance_pilot.data.financial_data_continuous import FinancialContinuousData
from finance_pilot.data.financial_interaction_data import FinancialInteractionData

from finance_pilot.metrics.kpi_ann_evaluation_metric import (
    AnnualizedKPIEvaluationMetric,
)
from finance_pilot.metrics.kpi_evaluation_metric import KPIEvaluationMetric
from finance_pilot.metrics.kpi_monthly_evaluation_metric import (
    MonthlyKPIEvaluationMetric,
)
from finance_pilot.metrics.pure_ndcg import PureNDCG

pd.options.mode.chained_assignment = None

RFR = "rfr"
LGBM = "lgbm"
TABPFN = "tabpfn"
SUPPORTED_MODELS = (RFR, LGBM, TABPFN)
START_TIME = dt.datetime.now()
SCRIPT_DIR = Path(__file__).resolve().parent
FINANCE_PILOT_DIR = SCRIPT_DIR / "finance_pilot"
COUNTERFACTUALS_SCRIPT = FINANCE_PILOT_DIR / "generate_counterfactuals.py"


#### run_dataset_analysis.py
# Dataset-analysis launcher mode integrated from the standalone analysis script.
DATASET_ANALYSIS_MODE = "dataset-analysis"
DATASET_ANALYSIS_PERIODS = (
    # start_date, end_date, num_splits, num_future, summary_suffix
    ("2019-08-01", "2021-02-26", 28, 13, 1), 
)


def run_basic_dataset_analysis(
    dataset_path: str,
    output_directory: str,
    dataset_analysis_script: Optional[str] = None,
    customer_analysis_script: Optional[str] = None,
) -> None:
    """Run the FAR-Trans asset/customer analysis over the two configured periods.

    The asset and customer analyzers are project files shipped alongside this
    launcher. By default they are resolved relative to this file, not the shell's
    current working directory. Optional explicit paths remain available for
    compatibility with custom project layouts.
    """
    dataset_path = os.path.abspath(dataset_path)
    output_directory = os.path.abspath(output_directory)
    dataset_analysis_script = os.path.abspath(
        dataset_analysis_script or os.path.join(FINANCE_PILOT_DIR, "dataset_analysis.py")
    )
    customer_analysis_script = os.path.abspath(
        customer_analysis_script or os.path.join(FINANCE_PILOT_DIR, "customer_analysis.py")
    )
    os.makedirs(output_directory, exist_ok=True)

    interactions_file = os.path.join(dataset_path, "transactions.csv")
    time_series_file = os.path.join(dataset_path, "close_prices.csv")
    min_file = os.path.join(dataset_path, "limit_prices.csv")
    child_env = os.environ.copy()
    pythonpath_entries = ["/app/test_docker", "/app/test_docker/finance_pilot"]
    existing_pythonpath = child_env.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath_entries.append(existing_pythonpath)
    child_env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    required_inputs = (interactions_file, time_series_file, min_file)
    missing_inputs = [path for path in required_inputs if not os.path.isfile(path)]
    if missing_inputs:
        raise FileNotFoundError(
            "Missing required FAR-Trans input file(s): {}".format(
                ", ".join(missing_inputs)
            )
        )

    for script_path in (dataset_analysis_script, customer_analysis_script):
        if not os.path.isfile(script_path):
            raise FileNotFoundError(
                "Required analysis script not found: {}".format(script_path)
            )

    with mlflow.start_run(run_name="dataset_analysis_subrun", nested=True) as run:
        run_id = run.info.run_id

        log_param("dataset_path", dataset_path)
        log_param("output_directory", output_directory)
        log_param("dataset_analysis_script", dataset_analysis_script)

        for start_date, end_date, num_splits, num_future, suffix in DATASET_ANALYSIS_PERIODS:
            print(
                "Starting analysis for period: {} to {}".format(
                    start_date, end_date
                )
            )

            child_env["MLFLOW_RUN_ID"] = run_id

            asset_command = [
                sys.executable,
                "-m",
                "test_docker.finance_pilot.dataset_analysis",
                interactions_file,
                time_series_file,
                "range",
                start_date,
                end_date,
                str(num_splits),
                str(num_future),
                output_directory,
                "assets_{}.csv".format(suffix),
            ]

            log_param("start_subprocess_dataset_analysis", True)
            try:
                subprocess.run(asset_command, check=True, env=child_env)
            except subprocess.CalledProcessError as exc:
                raise RuntimeError(
                    "Asset analysis failed for period {} to {} (exit code {}).".format(
                        start_date, end_date, exc.returncode
                    )
                ) from exc

            log_param("finish_subprocess_dataset_analysis", True)

    with mlflow.start_run(run_name="customer_analysis_subrun", nested=True) as run:
        run_id = run.info.run_id

        log_param("dataset_path", dataset_path)
        log_param("output_directory", output_directory)
        log_param("customer_analysis_script", customer_analysis_script)

        for start_date, end_date, num_splits, num_future, suffix in DATASET_ANALYSIS_PERIODS:
            print(
                "Starting analysis for period: {} to {}".format(
                    start_date, end_date
                )
            )

            child_env["MLFLOW_RUN_ID"] = run_id

            customer_command = [
                sys.executable,
                "-m",
                "test_docker.finance_pilot.customer_analysis",
                interactions_file,
                time_series_file,
                min_file,
                "range",
                start_date,
                end_date,
                str(num_splits),
                str(num_future),
                output_directory,
                "customers_{}.csv".format(suffix),
            ]

            log_param("start_subprocess_customer_analysis", True)
            try:
                subprocess.run(customer_command, check=True, env=child_env)
            except subprocess.CalledProcessError as exc:
                raise RuntimeError(
                    "Customer analysis failed for period {} to {} (exit code {}).".format(
                        start_date, end_date, exc.returncode
                    )
                ) from exc

            log_param("finish_subprocess_customer_analysis", True)
            print("End analysis for period: {} to {}".format(start_date, end_date))


def run_dataset_analysis_mode_if_requested() -> bool:
    """Handle the integrated dataset-analysis subcommand before the legacy CLI.

    Returns True when dataset-analysis mode was requested and completed.
    """
    if len(sys.argv) < 2 or sys.argv[1] != DATASET_ANALYSIS_MODE:
        return False

    parser = argparse.ArgumentParser(
        prog="{} {}".format(os.path.basename(sys.argv[0]), DATASET_ANALYSIS_MODE),
        description=(
            "Run FAR-Trans basic asset/customer dataset analysis over the two "
            "configured evaluation periods."
        ),
    )
    parser.add_argument(
        "dataset_path",
        help=(
            "Directory containing transactions.csv, close_prices.csv, "
            "and limit_prices.csv."
        ),
    )
    parser.add_argument(
        "output_dir",
        help="Directory in which analysis outputs and summary CSVs are stored.",
    )
    parser.add_argument(
        "--dataset-analysis-script",
        default=None,
        help=(
            "Optional path to dataset_analysis.py. By default the copy next to "
            "this launcher is used."
        ),
    )
    parser.add_argument(
        "--customer-analysis-script",
        default=None,
        help=(
            "Optional path to customer_analysis.py. By default the copy next to "
            "this launcher is used."
        ),
    )

    analysis_args = parser.parse_args(sys.argv[2:])

    log_param("dataset_path", analysis_args.dataset_path)
    log_param("output_dir", analysis_args.output_dir)
    log_param("dataset_analysis_script", analysis_args.dataset_analysis_script)
    log_param("customer_analysis_script", analysis_args.customer_analysis_script)

    run_basic_dataset_analysis(
        dataset_path=analysis_args.dataset_path,
        output_directory=analysis_args.output_dir,
        dataset_analysis_script=analysis_args.dataset_analysis_script,
        customer_analysis_script=analysis_args.customer_analysis_script,
    )
    return True

basic_kpis = [
    "past_profitability_63d",
    "past_profitability_126d",
    "past_profitability_189d",
    "volatility_63d",
    "volatility_126d",
    "volatility_189d",
    "avg_price_63d",
    "avg_price_126d",
    "avg_price_189d",
]

full_kpis = [
    "past_profitability_63d",
    "past_profitability_126d",
    "past_profitability_189d",
    "volatility_63d",
    "volatility_126d",
    "volatility_189d",
    "avg_price_63d",
    "avg_price_126d",
    "avg_price_189d",
    "sharpe_63d",
    "sharpe_126d",
    "sharpe_189d",
    "m_63d",
    "m_126d",
    "m_189d",
    "roc_63d",
    "roc_126d",
    "roc_189d",
    "MACD",
    "rsi_14",
    "dco_22",
    "min_63d",
    "min_126d",
    "min_189d",
    "max_63d",
    "max_126d",
    "max_189d",
    "exp_mean_63d",
    "exp_mean_126d",
    "exp_mean_189d",
]

basic_short_kpis = [
    "past_profitability_21d",
    "past_profitability_63d",
    "past_profitability_126d",
    "volatility_21d",
    "volatility_63d",
    "volatility_126d",
    "avg_price_21d",
    "avg_price_63d",
    "avg_price_126d",
]

full_short_kpis = [
    "past_profitability_21d",
    "past_profitability_63d",
    "past_profitability_126d",
    "volatility_21d",
    "volatility_63d",
    "volatility_126d",
    "avg_price_21d",
    "avg_price_63d",
    "avg_price_126d",
    "sharpe_21d",
    "sharpe_63d",
    "sharpe_126d",
    "m_21d",
    "m_63d",
    "m_126d",
    "roc_21d",
    "roc_63d",
    "roc_126d",
    "MACD",
    "rsi_14",
    "dco_22",
    "min_21d",
    "min_63d",
    "min_126d",
    "max_21d",
    "max_63d",
    "max_126d",
    "exp_mean_21d",
    "exp_mean_63d",
    "exp_mean_126d",
]


def safe_log_dataset(df: pd.DataFrame, name: str, output_dir: str) -> None:
    try:
        log_dataset(df, name=name, output_dir=output_dir)
    except Exception as exc:
        print("[WARN] Could not log dataset {}: {}".format(name, exc))


def safe_log_whylogs(df: pd.DataFrame, name: str) -> None:
    try:
        log_whylogs_profile(df, name=name)
    except Exception as exc:
        print("[WARN] Could not log whylogs profile {}: {}".format(name, exc))


def get_feature_list(feature_set: str) -> List[str]:
    if feature_set == "basic":
        return basic_kpis
    if feature_set == "full":
        return full_kpis
    if feature_set == "basic_short":
        return basic_short_kpis
    if feature_set == "full_short":
        return full_short_kpis

    raise ValueError(
        "Invalid feature_set. Use one of: basic, full, basic_short, full_short"
    )


def get_name(rec_model: str, params: List[str]) -> Optional[str]:
    """Build a stable run/model name from the model-specific parameters."""
    if rec_model in (RFR, LGBM):
        if len(params) < 2:
            return None

        try:
            n_estimators = int(params[0])
        except ValueError:
            return None

        feature_set = params[1]
        get_feature_list(feature_set)  # validate early
        return "{}_{}_{}".format(rec_model, n_estimators, feature_set)

    if rec_model == TABPFN:
        if len(params) < 1:
            return None

        feature_set = params[0]
        get_feature_list(feature_set)  # validate early

        if len(params) >= 2:
            try:
                sample_fraction = float(params[1])
            except ValueError:
                return None
            if not 0.0 < sample_fraction <= 1.0:
                return None
            return "{}_{}_sample-{}".format(TABPFN, feature_set, sample_fraction)

        return "{}_{}".format(TABPFN, feature_set)

    return None


def compute_profitability(
    time_series: pd.DataFrame,
    recommendation_date: Any,
    evaluation_date: Any,
    min_values: Optional[pd.DataFrame] = None,
) -> Dict[Any, float]:
    rec_series = time_series[time_series[DEFAULT_TIMESTAMP_COL] == recommendation_date]

    future_series = time_series[time_series[DEFAULT_TIMESTAMP_COL] == evaluation_date]

    aux_series = rec_series.merge(
        future_series,
        on=DEFAULT_ITEM_COL,
        suffixes=("_present", "_future"),
    )

    aux_series["profitability"] = (
        aux_series[DEFAULT_RATING_COL + "_future"]
        - aux_series[DEFAULT_RATING_COL + "_present"]
    ) / aux_series[DEFAULT_RATING_COL + "_present"]

    prof_dict = {}

    for _, row in aux_series.iterrows():
        prof_dict[row[DEFAULT_ITEM_COL]] = row["profitability"]

    if min_values is not None:
        max_series = rec_series.merge(min_values, on=DEFAULT_ITEM_COL)

        max_series["profitability"] = (
            max_series["max_price"] - max_series[DEFAULT_RATING_COL]
        ) / max_series[DEFAULT_RATING_COL]

        for _, row in max_series.iterrows():
            if row[DEFAULT_ITEM_COL] not in prof_dict:
                prof_dict[row[DEFAULT_ITEM_COL]] = row["profitability"]

    return prof_dict


def compute_volatility(
    time_series: pd.DataFrame,
    recommendation_date: Any,
    evaluation_date: Any,
) -> Dict[Any, float]:
    series = time_series[
        time_series[DEFAULT_TIMESTAMP_COL].between(
            recommendation_date,
            evaluation_date,
        )
    ]

    series_asset = {}

    for asset in series[DEFAULT_ITEM_COL].unique().flatten():
        aux_series = series[series[DEFAULT_ITEM_COL] == asset].copy()

        aux_series["profit"] = (
            aux_series[DEFAULT_RATING_COL] - aux_series[DEFAULT_RATING_COL].shift(1)
        ) / aux_series[DEFAULT_RATING_COL].shift(1)

        aux_series = aux_series.dropna()

        if aux_series.empty:
            series_asset[asset] = 0.0
        else:
            series_asset[asset] = aux_series["profit"].std() * np.sqrt(252)

    return series_asset


def test_algorithm(
    algorithm: Any,
    eval_metrics: List[Any],
    file_prefix: str,
    recommendation_date: Any,
    customers: Any,
) -> Optional[Dict[str, Any]]:
    if os.path.exists(file_prefix + "_metrics.csv"):
        print("Skipping {}; metrics already exist.".format(file_prefix))
        return None

    local_start = dt.datetime.now()
    print("Started {}".format(file_prefix))

    algorithm.train(recommendation_date)
    print("Algorithm trained in {}".format(dt.datetime.now() - local_start))

    # finance_pilot.recommend signature changed to (rec_time, target_custs, repeated, only_test_customers)
    recs = algorithm.recommend(recommendation_date, customers, False, True)
    recs = recs.sort_values(
        by=[DEFAULT_USER_COL, DEFAULT_RATING_COL],
        ascending=[False, False],
    )

    recs_file = file_prefix + "_recs.txt"
    recs.to_csv(recs_file, index=False)

    safe_log_dataset(recs, "recommendations", "recommendations")
    safe_log_whylogs(recs, "recommendations_profile")

    cutoffs = [1, 5, 10, 20, 50, 100, 1000]
    metric_res = {}

    for metric_name, metric_obj in eval_metrics:
        print("Started metric {}".format(metric_name))

        metric_dict = metric_obj.evaluate_cutoffs(
            recs,
            cutoffs,
            customers,
            True,
        )

        for cutoff in cutoffs:
            # Use colon separator for metric names to comply with MLflow naming rules
            full_metric_name = "{}:{}".format(metric_name, cutoff)
            metric_res[full_metric_name] = metric_dict[cutoff]
            aggregate_value = metric_dict[cutoff][1]

            try:
                log_metrics({full_metric_name: float(aggregate_value)})
            except Exception as exc:
                print(
                    "[WARN] Could not log metric {}: {}".format(full_metric_name, exc)
                )

    metrics_file = file_prefix + "_metrics.csv"
    with open(metrics_file, "w") as f:
        for key, val in metric_res.items():
            f.write(key + "\t" + str(val[1]) + "\n")

    customer_metric_df = None

    for key, val in metric_res.items():
        if customer_metric_df is None:
            customer_metric_df = val[0].rename(columns={"metric": key})
        else:
            aux_df = val[0].rename(columns={"metric": key})
            customer_metric_df = customer_metric_df.merge(aux_df, on=DEFAULT_USER_COL)

    customers_file = file_prefix + "_customers.csv"
    customer_metric_df.to_csv(customers_file, index=False)

    safe_log_dataset(customer_metric_df, "customer_metrics", "customer_metrics")
    safe_log_whylogs(customer_metric_df, "customer_metrics_profile")

    try:
        mlflow.log_artifact(recs_file, artifact_path="recommendations")
        mlflow.log_artifact(metrics_file, artifact_path="metrics")
        mlflow.log_artifact(customers_file, artifact_path="customer_metrics")
    except Exception as exc:
        print("[WARN] Could not log artifacts: {}".format(exc))

    print("Finished {} in {}".format(file_prefix, dt.datetime.now() - local_start))

    return metric_res


def run_regressor(
    model_id: str,
    params: List[str],
    financial_data: Any,
    recommendation_date: Any,
    eval_metrics: List[Any],
    output_dir: str,
    file_name: str,
    num_months: str,
) -> Optional[Dict[str, Any]]:
    """Instantiate the requested model and run the common profitability pipeline."""

    sample_fraction = None
    model_n_estimators = None

    if model_id in (RFR, LGBM):
        if len(params) < 2:
            raise ValueError(
                "{} requires: <n_estimators> <feature_set>".format(model_id)
            )
        model_n_estimators = int(params[0])
        feature_set = params[1]

    elif model_id == TABPFN:
        if len(params) < 1:
            raise ValueError("tabpfn requires: <feature_set> [sample_fraction]")
        feature_set = params[0]

        if len(params) >= 2:
            sample_fraction = float(params[1])
            if not 0.0 < sample_fraction <= 1.0:
                raise ValueError("TabPFN sample_fraction must be in (0, 1].")
    else:
        raise ValueError("Unsupported model: {}".format(model_id))

    feats = get_feature_list(feature_set)

    if model_id == RFR:
        assert model_n_estimators is not None
        model = RFRKPIModel(
            n_estimators=model_n_estimators,
            k=5,
            kpi_type=feature_set,
            kpi_features=feats,
            random_state=42,
            n_jobs=1,
        )
        model_display_name = "RFRKPIModel"
        framework = "internal-kpi-pipeline"

    elif model_id == LGBM:
        assert model_n_estimators is not None
        try:
            from finance_pilot.algorithms.lgbm_kpi_model import LGBMKPIModel
        except ImportError as exc:
            raise ImportError(
                "LightGBM is not installed. Install it with: pip install lightgbm"
            ) from exc

        model = LGBMKPIModel(
            n_estimators=model_n_estimators,
            k=5,
            kpi_type=feature_set,
            kpi_features=feats,
            random_state=42,
            n_jobs=1,
        )
        model_display_name = "LGBMKPIModel"
        framework = "internal-kpi-pipeline"

    else:  # TABPFN
        if TabPFNRegressor is None:
            raise ImportError(
                "TabPFN is not installed. Install it with: pip install tabpfn"
            )
        model = TabPFNRegressor()
        model_display_name = "TabPFNRegressor"
        framework = "tabpfn"

        # IMPORTANT:
        # The launcher accepts an optional TabPFN sample fraction, but this
        # tracked version of ProfitabilityPrediction only receives the final
        # feature matrix at fit time and does not expose asset IDs here.
        # Therefore proportional per-asset sampling cannot be implemented
        # faithfully at this point without modifying ProfitabilityPrediction.
        if sample_fraction is not None:
            print(
                "[WARN] TabPFN sample_fraction={} was supplied, but proportional "
                "per-asset sampling is not implemented in this tracked "
                "recommendation.py. The value will be logged only."
                .format(sample_fraction)
            )

    algorithm = ProfitabilityPrediction(
        model,
        financial_data,
        num_months,
        feats,
        -1,
        save_for_testing=True,
    )

    # CHANGED: model metadata is now model-aware.
    log_model_info(
        model_information={
            "model_name": model_display_name,
            "model_version": "1.0",
            "framework": framework,
            "task": "financial_asset_recommendation",
            "recommendation_date": str(recommendation_date),
        }
    )

    hyperparams: Dict[str, Any] = {
        "feature_set": feature_set,
        "num_months": num_months,
        "features": ",".join(feats),
    }
    if model_n_estimators is not None:
        hyperparams["n_estimators"] = model_n_estimators
    if sample_fraction is not None:
        hyperparams["sample_fraction"] = sample_fraction

    log_model_hyperparameters(hyperparams)

    file_prefix = os.path.join(output_dir, file_name)

    result = test_algorithm(
        algorithm=algorithm,
        eval_metrics=eval_metrics,
        file_prefix=file_prefix,
        recommendation_date=recommendation_date,
        customers=financial_data.users,
    )

    try:
        artifact_dir = algorithm._artifact_dir()
        if os.path.exists(artifact_dir):
            mlflow.log_artifacts(
                artifact_dir,
                artifact_path="artifacts_for_counterfactuals",
            )

            pipeline_file = algorithm._artifact_path(
                "profitability_recommendation_pipeline",
                recommendation_date,
                "pkl",
            )
            if os.path.exists(pipeline_file):
                mlflow.log_artifact(pipeline_file, artifact_path="model")
    except Exception:
        pass

    return result


def load_financial_data(interactions_file: str, time_series_file: str):
    interaction_data = FinancialInteractionData(interactions_file)
    time_series_data = FinancialAssetTimeSeries(time_series_file)

    data = FinancialContinuousData(interaction_data, time_series_data)
    data.load()

    return interaction_data, time_series_data, data


def load_or_compute_kpis(
    data: Any,
    output_dir: str,
    kpi_type: str = "full_short",
) -> pd.DataFrame:
    kpi_file = os.path.join(output_dir, "kpis.csv")
    print("KPI file: {}".format(kpi_file))

    if os.path.exists(kpi_file):
        kpi_gen = LoadKPIGenerator(kpi_file)
    else:
        kpi_gen = MAKPIGenerator(data.time_series.data, 5, kpi_type)

    kpi_gen.compute()
    kpis = kpi_gen.get_kpis()

    if not os.path.exists(kpi_file):
        kpi_gen.print_kpis(kpi_file)

    data.add_kpis(kpis)

    return kpis


def get_dates_from_args(args: argparse.Namespace, data: Any):
    if args.date_format == "range":
        min_date = dt.datetime.strptime(args.min_date, "%Y-%m-%d")
        max_date = dt.datetime.strptime(args.max_date, "%Y-%m-%d")

        print("Num splits: {} Num future: {}".format(args.num_splits, args.num_future))

        dates, future_dates = data.get_dates(
            min_date,
            max_date,
            args.num_splits,
            args.num_future,
        )

    elif args.date_format == "fixed_dates":
        dates = [pd.to_datetime(x) for x in args.split_dates.split(",")]
        future_dates = [pd.to_datetime(x) for x in args.future_dates.split(",")]

    else:
        raise ValueError("Invalid date format. Use range or fixed_dates.")

    print("Selected dates:")
    for i in range(len(dates)):
        print(
            "\t{} Training date: {}\tFuture date: {}".format(
                i, dates[i], future_dates[i]
            )
        )

    return dates, future_dates


def build_metrics(splitted_data: Any, rec_date: Any, future_date: Any) -> List[Any]:
    profitability_dict = compute_profitability(
        splitted_data.time_series,
        rec_date,
        future_date,
        None,
    )

    volatility_dict = compute_volatility(
        splitted_data.time_series,
        rec_date,
        future_date,
    )

    metrics = [
        (
            "profitability",
            KPIEvaluationMetric(splitted_data, profitability_dict),
        ),
        (
            "annualized_prof",
            AnnualizedKPIEvaluationMetric(
                splitted_data,
                profitability_dict,
                (future_date - rec_date).days,
            ),
        ),
        (
            "monthly_prof",
            MonthlyKPIEvaluationMetric(
                splitted_data,
                profitability_dict,
                (future_date - rec_date).days,
            ),
        ),
        (
            "volatility",
            KPIEvaluationMetric(splitted_data, volatility_dict),
        ),
        (
            "ndcg",
            PureNDCG(splitted_data),
        ),
    ]

    return metrics


def _build_counterfactual_model_param_tag(model_name: str, params: List[str]) -> str:
    if len(params) < 2:
        raise ValueError(
            "{} requires <n_estimators> <feature_set> to generate counterfactuals".format(
                model_name
            )
        )

    return "n-{}_kpi-{}_internal_kpis".format(int(params[0]), params[1])


def _run_counterfactual_generation(
    model_name: str,
    params: List[str],
    recommendation_date: Any,
) -> None:
    if model_name not in (RFR, LGBM):
        return

    if not COUNTERFACTUALS_SCRIPT.is_file():
        raise FileNotFoundError(
            "Counterfactual generator not found: {}".format(COUNTERFACTUALS_SCRIPT)
        )

    with mlflow.start_run(run_name="counterfactual_generation_sudrun", nested=True) as run:
        if importlib.util.find_spec("dice_ml") is None:
            print("[WARN] dice_ml is not installed; skipping counterfactual generation.")
            return

        artifact_dir = Path("artifacts_for_counterfactuals") / "{}_{}".format(
            model_name,
            _build_counterfactual_model_param_tag(model_name, params),
        )
        if not artifact_dir.is_dir():
            raise FileNotFoundError(
                "Counterfactual artifact directory not found: {}".format(artifact_dir)
            )

        date_tag = pd.to_datetime(recommendation_date).strftime("%Y-%m-%d")
        pkl_candidates = sorted(
            artifact_dir.glob("profitability_recommendation_pipeline_{}_*.pkl".format(date_tag))
        )
        if not pkl_candidates:
            raise FileNotFoundError(
                "No counterfactual pipeline PKL found in {} for date {}".format(
                    artifact_dir,
                    date_tag,
                )
            )

        env = os.environ.copy()
        pythonpath_entries = [str(FINANCE_PILOT_DIR)]
        existing_pythonpath = env.get("PYTHONPATH")
        if existing_pythonpath:
            pythonpath_entries.append(existing_pythonpath)
        env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)

        command = [
            sys.executable,
            str(COUNTERFACTUALS_SCRIPT),
            "--model-pkl",
            str(pkl_candidates[0]),
        ]

        print("Generating counterfactuals from {}".format(pkl_candidates[0]))
        subprocess.run(command, cwd=str(SCRIPT_DIR.parent), env=env, check=True)


def run_single_experiment(
    args: argparse.Namespace,
    data: Any,
    interaction_data: Any,
    time_series_data: Any,
    kpis: pd.DataFrame,
    rec_date: Any,
    future_date: Any,
    model_name: str,
    params: List[str],
) -> Optional[Dict[str, Any]]:
    output_dir = args.output_dir
    months_term = args.months

    delta = dt.timedelta(days=36525)
    min_split_date = rec_date - delta

    alg_name = model_name + "_" + rec_date.strftime("%Y-%m-%d")

    if os.path.exists(os.path.join(output_dir, alg_name + "_metrics.csv")):
        print("Skipped {}; metrics already exist.".format(alg_name))
        return None

    with mlflow.start_run(run_name=alg_name, nested=True) as run:
        print("Run started: {}".format(run.info.run_id))

        log_param("algorithm_name", alg_name)
        log_param("model", args.model)
        log_param("recommendation_date", str(rec_date))
        log_param("future_date", str(future_date))
        log_param("months", months_term)
        if args.model in (RFR, LGBM):
            log_param("n_estimators", int(params[0]))
            log_param("feature_set", params[1])
        else:
            log_param("feature_set", params[0])
            if len(params) >= 2:
                log_param("sample_fraction", float(params[1]))
        log_param("output_dir", output_dir)

        safe_log_dataset(interaction_data.data, "transactions", "datasets")
        safe_log_whylogs(interaction_data.data, "transactions_profile")

        safe_log_dataset(time_series_data.data, "close_prices", "datasets")
        safe_log_whylogs(time_series_data.data, "close_prices_profile")

        safe_log_dataset(kpis, "kpis", "features")
        safe_log_whylogs(kpis, "kpis_profile")

        splitted_data = data.split(
            min_split_date,
            rec_date,
            future_date,
            DataFilter(
                CustomerInTrain(),
                AssetWithTestPrice(),
                RatingsNotInTrain(),
                NoFilter(),
                False,
                True,
                False,
            ),
        )

        print("Dataset split completed: {}".format(dt.datetime.now() - START_TIME))

        try:
            log_train_test_dataset(splitted_data.train, splitted_data.test)
        except Exception as exc:
            print("[WARN] Could not log train/test dataset: {}".format(exc))

        safe_log_whylogs(
            splitted_data.train,
            "train_profile_{}".format(rec_date.strftime("%Y%m%d")),
        )
        safe_log_whylogs(
            splitted_data.test,
            "test_profile_{}".format(rec_date.strftime("%Y%m%d")),
        )

        if (
            DEFAULT_TIMESTAMP_COL in splitted_data.train.columns
            and DEFAULT_TIMESTAMP_COL in splitted_data.test.columns
        ):
            try:
                timestamp_analysis(
                    train_timestamps=splitted_data.train[DEFAULT_TIMESTAMP_COL],
                    test_timestamps=splitted_data.test[DEFAULT_TIMESTAMP_COL],
                    output_dir="timestamps",
                )
            except Exception as exc:
                print("[WARN] Could not log timestamp analysis: {}".format(exc))

        log_param("num_train_rows", splitted_data.train.shape[0])
        log_param("num_test_rows", splitted_data.test.shape[0])
        log_param("num_users", len(splitted_data.users))
        log_param("num_assets", len(splitted_data.assets))

        metrics = build_metrics(splitted_data, rec_date, future_date)

        tracker_data, output_location = start_tracker(
            output_file_name="emissions_{}".format(alg_name)
        )
        print("Resource monitoring started")

        try:
            metric_res = run_regressor(
                model_id=args.model,
                params=params,
                financial_data=splitted_data,
                recommendation_date=rec_date,
                eval_metrics=metrics,
                output_dir=output_dir,
                file_name=alg_name,
                num_months=months_term,
            )
        finally:
            stop_tracker(tracker_data, output_location)
            print("Resource monitoring stopped")

        _run_counterfactual_generation(args.model, params, rec_date)

        return metric_res


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="finance_pilot_tracked",
        description="Tracked Finance Pilot recommendation experiment.",
    )

    parser.add_argument("interactions", help="Customer-asset transaction CSV file.")
    parser.add_argument("time_series", help="Asset price time series CSV file.")

    subparsers = parser.add_subparsers(dest="date_format")
    subparsers.required = True

    parser_range = subparsers.add_parser("range")
    parser_range.add_argument("min_date")
    parser_range.add_argument("max_date")
    parser_range.add_argument("num_splits", type=int)
    parser_range.add_argument("num_future", type=int)
    parser_range.add_argument("output_dir")
    parser_range.add_argument("months")
    parser_range.add_argument("model", choices=SUPPORTED_MODELS)
    parser_range.add_argument("params", nargs="*")

    parser_fixed = subparsers.add_parser("fixed_dates")
    parser_fixed.add_argument("split_dates")
    parser_fixed.add_argument("future_dates")
    parser_fixed.add_argument("output_dir")
    parser_fixed.add_argument("months")
    parser_fixed.add_argument("model", choices=[RFR])
    parser_fixed.add_argument("params", nargs="*")

    return parser.parse_args()


def main() -> None:
    experiment_name = "finance_pilot_recommendation_tracked_v2"
    # Ensure MLflow is pointed at the REST tracking server (prefer HTTP URI).
    _env_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not _env_uri or not _env_uri.startswith("http"):
        mlflow.set_tracking_uri("http://certain_mlflow:5001")
    else:
        mlflow.set_tracking_uri(_env_uri)

    mlflow.set_experiment(experiment_name)
    print("Experiment: {}".format(experiment_name))

    if len(sys.argv) > 1 and sys.argv[1] == DATASET_ANALYSIS_MODE:
        with mlflow.start_run(run_name="customer+dataset_analysis"):
            if run_dataset_analysis_mode_if_requested():
                return

    args = parse_arguments()

    os.makedirs(args.output_dir, exist_ok=True)

    model_name = get_name(args.model, args.params)

    if model_name is None:
        sys.stderr.write("ERROR: Invalid model or parameters\n")
        sys.exit(1)

    interaction_data, time_series_data, data = load_financial_data(
        args.interactions,
        args.time_series,
    )

    print("Dataset loaded: {}".format(dt.datetime.now() - START_TIME))

    # CHANGED: derive KPI/feature set from the model-specific parameter layout.
    if args.model in (RFR, LGBM):
        kpi_type = args.params[1]
    else:
        kpi_type = args.params[0]

    kpis = load_or_compute_kpis(
        data,
        args.output_dir,
        kpi_type=kpi_type,
    )

    print("Technical indicators computed: {}".format(dt.datetime.now() - START_TIME))

    dates, future_dates = get_dates_from_args(args, data)

    with mlflow.start_run(run_name=model_name) as parent_run:
        if args.model in (RFR, LGBM):
            if len(args.params) < 2:
                sys.stderr.write(
                    "ERROR: {} requires <n_estimators> <feature_set>\n".format(args.model)
                )
                sys.stderr.write(
                    "feature_set: basic, full, basic_short, full_short\n"
                )
                sys.exit(1)
        elif args.model == TABPFN:
            if len(args.params) < 1:
                sys.stderr.write(
                    "ERROR: tabpfn requires <feature_set> [sample_fraction]\n"
                )
                sys.stderr.write(
                    "feature_set: basic, full, basic_short, full_short\n"
                )
                sys.exit(1)

            if len(args.params) >= 2:
                try:
                    sample_fraction = float(args.params[1])
                except ValueError:
                    sys.stderr.write("ERROR: TabPFN sample_fraction must be numeric.\n")
                    sys.exit(1)

                if not 0.0 < sample_fraction <= 1.0:
                    sys.stderr.write(
                        "ERROR: TabPFN sample_fraction must be in (0, 1].\n"
                    )
                    sys.exit(1)

        log_param("model", args.model)
        log_param("params", args.params)
        # CHANGED: parent run logging is model-aware.
        if args.model in (RFR, LGBM):
            log_param("feature_set", args.params[1])
            log_param("n_estimators", int(args.params[0]))
        else:
            log_param("feature_set", args.params[0])
            if len(args.params) >= 2:
                log_param("sample_fraction", float(args.params[1]))

        log_param("months", args.months)

        for i in range(len(dates)):
            run_single_experiment(
                args=args,
                data=data,
                interaction_data=interaction_data,
                time_series_data=time_series_data,
                kpis=kpis,
                rec_date=dates[i],
                future_date=future_dates[i],
                model_name=model_name,
                params=args.params,
            )

    print("Workflow complete.")



if __name__ == "__main__":
    main()
