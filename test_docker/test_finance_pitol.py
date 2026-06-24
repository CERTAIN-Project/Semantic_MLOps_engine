# GitHub repo: https://github.com/ALi-KORDiA/counterfactualFAR/tree/main

#!/usr/bin/env python3
import argparse
import datetime as dt
import os
import sys
from typing import Any, Dict, List, Optional

import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

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
START_TIME = dt.datetime.now()


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
    if rec_model != RFR:
        return None

    if len(params) < 2:
        return None

    n_estimators = int(params[0])
    feature_set = params[1]
    return "{}_{}_{}".format(RFR, n_estimators, feature_set)


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

    recs = algorithm.recommend(recommendation_date, False, True)
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
            full_metric_name = "{}@{}".format(metric_name, cutoff)
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
    params: List[str],
    financial_data: Any,
    recommendation_date: Any,
    eval_metrics: List[Any],
    output_dir: str,
    file_name: str,
    num_months: str,
) -> Optional[Dict[str, Any]]:
    n_estimators = int(params[0])
    feature_set = params[1]
    feats = get_feature_list(feature_set)

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=42,
        n_jobs=-1,
    )

    algorithm = ProfitabilityPrediction(
        model,
        financial_data,
        num_months,
        feats,
        -1,
    )

    log_model_info(
        model_information={
            "model_name": "RandomForestRegressor",
            "model_version": "1.0",
            "framework": "scikit-learn",
            "task": "financial_asset_recommendation",
            "recommendation_date": str(recommendation_date),
        }
    )

    log_model_hyperparameters(
        {
            "n_estimators": n_estimators,
            "feature_set": feature_set,
            "num_months": num_months,
            "features": ",".join(feats),
        }
    )

    file_prefix = os.path.join(output_dir, file_name)

    return test_algorithm(
        algorithm=algorithm,
        eval_metrics=eval_metrics,
        file_prefix=file_prefix,
        recommendation_date=recommendation_date,
        customers=financial_data.users,
    )


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
        log_param("n_estimators", int(params[0]))
        log_param("feature_set", params[1])
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
    parser_range.add_argument("model", choices=[RFR])
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
    args = parse_arguments()

    if len(args.params) < 2:
        sys.stderr.write("ERROR: Invalid arguments for Random Forest\n")
        sys.stderr.write("Usage params: <n_estimators> <feature_set>\n")
        sys.stderr.write("feature_set: basic, full, basic_short, full_short\n")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    experiment_name = "finance_pilot_recommendation_tracked_v2"
    mlflow.set_experiment(experiment_name)
    print("Experiment: {}".format(experiment_name))

    model_name = get_name(args.model, args.params)

    if model_name is None:
        sys.stderr.write("ERROR: Invalid model or parameters\n")
        sys.exit(1)

    interaction_data, time_series_data, data = load_financial_data(
        args.interactions,
        args.time_series,
    )

    print("Dataset loaded: {}".format(dt.datetime.now() - START_TIME))

    kpis = load_or_compute_kpis(
        data,
        args.output_dir,
        kpi_type="full_short",
    )

    print("Technical indicators computed: {}".format(dt.datetime.now() - START_TIME))

    dates, future_dates = get_dates_from_args(args, data)

    with mlflow.start_run(run_name=model_name) as parent_run:
        log_param("model", args.model)
        log_param("params", args.params)
        log_param("feature_set", args.params[1])
        log_param("n_estimators", int(args.params[0]))
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
