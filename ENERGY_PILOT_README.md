# CERTAIN Logging Functions Guide

This guide explains the CERTAIN logging functions demonstrated in the complete ML workflow example. It describes what each function records, when to use it, and how to call it.

## 0. Which function should I use?

You should adapt the docker file to copy your code inside the docker and 

| Need | Use |
|---|---|
| Create or select an experiment | `tracker.set_experiment` |
| Start a parent or nested run | `tracker.start_run` |
| Add searchable run metadata | `tracker.set_tags` |
| Log fixed configuration | `log_params` |
| Log changing numeric results | `log_metrics` |
| Log CPU/RAM/GPU measurements | `log_resources` |
| Monitor emissions continuously | `start_tracker` / `stop_tracker` |
| Record hyperparameter search definition | `log_search_space` |
| Log a transformed DataFrame | `log_dataset` |
| Log final train/test inputs | `log_train_test_dataset` |
| Profile dataset quality | `log_whylogs_profile` |
| Record data file location and size | `save_dataset_manifest` |
| Analyze temporal splits | `timestamp_analysis` |
| Measure dataset drift | `log_drift_metrics` |
| Document preprocessing | `log_data_techniques` |
| Record model identity | `log_model_info` |
| Record model design | `log_model_architecture` |
| Record model-specific hyperparameters | `log_model_hyperparameters` |
| Record input/output schema | `log_model_signature` |
| Record Git provenance | `log_git_metadata` |
| Record runtime environment | `collect_runtime_environment` + `save_runtime_env_as_artifact` |
| Record risks and oversight | `log_ai_actors` + `log_labeling_procedures` + `log_risk` + `log_human_oversight` + `log_transparency_measure` + `log_change` |
| Record explainability and documentation | `log_declarations_of_conformity` + `log_visual_documentations` + `log_explainable_ai` |
| Record deployment lifecycle | `log_model_packaging` + `log_build_testing` + `log_standards` + `log_interface` + `log_decommissioning` |
| Upload an arbitrary file | `tracker.log_artifact` |

---

## 1. Recommended workflow

A typical CERTAIN-tracked workflow follows this order:

The following two functions replace the mlflow set_experiment and start_run functions
```python
from certain_library.tracking.tracker import tracker

# set the experiment name and the experiment tags
tracker.set_experiment(
    experiment_name="energy_forecasting",
    tags={
        "team": "energy",
        "project": "load_forecasting",
        "owner": "data-science",
    },
)

# start the run
with tracker.start_run(
    run_name="xgboost_training",
    tags={"run_type": "training"},
) as run:
    # Log provenance, data, parameters, metrics, resources,
    # model information, compliance, and artifacts here.
    ...
```

In case you need parent and trails: use one parent run for the complete workflow and use nested runs for e.g hyperparameter-search trials:

```python
with tracker.start_run(run_name="parent_training") as parent_run:
    for trial in study.trials:
        with tracker.start_run(
            nested=True,
            run_name=f"Trial_{trial.number}",
        ):
            log_params({"trial_number": trial.number})
            log_metrics({"trial_mse": float(trial.value)}, step=trial.number)
```

## 2. Core experiment and run functions

### `tracker.set_experiment(...)`

Creates or activates an MLflow experiment.

Use it once before starting the parent run.

```python
tracker.set_experiment(
    experiment_name="complete_ml_workflow_demo_v2",
    tags={
        "team": "energy",
        "project": "opsd_demo",
        "owner": "dimitrios",
    },
)
```

Use experiment tags for metadata shared by all runs, such as team, project, owner, business domain, or cost center.

---

### `tracker.start_run(...)`

Starts an MLflow run and initializes CERTAIN run metadata.

```python
with tracker.start_run(
    run_name="random_forest_classifier",
    tags={
        "project": "opsd_demo",
        "test_run": "1",
    },
) as run:
    print(run.info.run_id)
```

For nested trial runs:

```python
with tracker.start_run(
    nested=True,
    run_name=f"Trial_{trial.number}",
):
    ...
```

Use a parent run for the complete workflow and nested runs for Optuna trials, cross-validation folds, candidate models, or repeated evaluations.

---

### `tracker.set_tags(tags)`

Adds or updates tags on the active run.

```python
tracker.set_tags(
    {
        "run_stage": "data_preprocessing",
        "run_type": "demo",
    }
)
```

Use tags for searchable metadata that can be different among runs. Examples include stage, owner, model family, data version, validation status, or deployment target.

Do not use tags for numeric time-series measurements. Use metrics instead.

---

### `tracker.log_artifact(local_path, artifact_path=None)`

Uploads any file to the active MLflow run and mirrors it through CERTAIN.

```python
tracker.log_artifact(
    "reports/feature_importance.png",
    artifact_path="explainability",
)
```

Use it for files that do not have a dedicated CERTAIN helper, including plots, reports, logs, configuration files, serialized outputs, and generated documents.

---

## 3. Parameters, metrics, resources, and search space

Import:

```python
from certain_library.log_basic.log_params import log_params
from certain_library.train_monitor.log_metrics import (
    log_metrics,
    log_resources,
    log_search_space,
)
```

### `log_params(params)`

Logs configuration values that describe how the run was executed.

```python
log_params(
    {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.05,
    }
)
```

Use parameters for values that should remain fixed within a run:

- hyperparameters;
- dataset paths or versions;
- random seeds;
- preprocessing configuration;
- row and column counts;
- algorithm choices.

MLflow parameters are effectively immutable within a run. Do not log the same parameter key later with a different value.

---

### `log_metrics(metrics, step=None, timestamp=None, keep_best=False)`

Logs numeric measurements.

```python
log_metrics(
    {
        "mse": float(mse),
        "r2_score": float(r2),
    },
    step=step,
)
```

Use metrics for values that can vary over time or steps:

- loss;
- MSE, RMSE, R², accuracy, precision, recall;
- validation scores;
- latency;
- throughput;
- drift statistics.

Use `step` for epochs, boosting rounds, trials, batches, or evaluation checkpoints.

```python
for step in range(1, max_steps + 1):
    log_metrics({"mse": float(mse)}, step=step)
```

Use `keep_best=True` only when you want to save the best-value that your code detects.

---

### `log_resources(resources, step=None, timestamp=None)`

Logs system-resource measurements under the MLflow `system_metrics/` namespace and records their history in `run_resources.json`.

```python
process = psutil.Process(os.getpid())

log_resources(
    {
        "trial_memory_usage_mb": process.memory_info().rss / 1024 / 1024,
        "trial_cpu_usage_percent": psutil.cpu_percent(interval=1),
    },
    step=trial.number,
)
```

Use it for:

- CPU usage;
- RAM usage;
- GPU usage;
- disk or network utilization;
- process-level resource measurements.

Do not mix model-quality metrics and resource metrics. Use `log_metrics` for model results and `log_resources` for infrastructure measurements.

---

### `log_search_space(search_space)`

Records the hyperparameter search definition.

```python
search_space = {
    "n_estimators": {
        "type": "int",
        "low": 10,
        "high": 100,
    },
    "max_depth": {
        "type": "int",
        "low": 3,
        "high": 10,
    },
    "learning_rate": {
        "type": "float",
        "low": 0.01,
        "high": 0.3,
        "log": True,
    },
}

log_search_space(search_space)
```

Call it once on the parent run before starting the optimization study. Trial-specific selected values belong in each nested run through `log_params`.

---

## 4. Resource and emissions monitoring

Import:

```python
from certain_library.resource_monitor.resource import (
    start_tracker,
    stop_tracker,
)
```

### `start_tracker(output_file_name=...)`

Starts continuous energy and emissions monitoring, typically through CodeCarbon.

```python
    tracker_data, output_location = start_tracker(output_file_name="emissions_train")
```

Use it around a meaningful stage such as:

- data preparation;
- hyperparameter optimization;
- final model training;
- inference benchmarking.

---

### `stop_tracker(tracker_instance, output_location)`

Stops the corresponding resource tracker and saves its output.

```python
stop_tracker(tracker_data, output_location)
```

Always pair every `start_tracker` with `stop_tracker`.


Use `start_tracker`/`stop_tracker` for continuous emissions monitoring. Use `log_resources` only for explicit point-in-time CPU, RAM, or GPU measurements.

---

## 5. Dataset logging

Import:

```python
from certain_library.data_analysis.log_dataset import (
    log_dataset,
    log_train_test_dataset,
)
from certain_library.data_analysis.log_whylogs import log_whylogs_profile
from certain_library.metadata.artifact_metadata import save_dataset_manifest
```

### `log_dataset(dataset, name, output_dir=...)`

Logs a named dataset or transformed dataset.

```python
log_dataset(
    df_cleaned,
    name="df_cleaned",
    output_dir="data_cleaning",
)
```

Call it after important transformations, such as:

- cleaning;
- filtering;
- feature engineering;
- augmentation;
- aggregation.

Choose stable, descriptive names. Avoid logging every temporary intermediate DataFrame.

---

### `log_train_test_dataset(train_dataset, test_dataset)`

Logs the train/test split as MLflow inputs.

```python
log_train_test_dataset(
    train_combined,
    test_combined,
)
```

Call it after creating the final split used by the model. This creates lineage between the run and its training/evaluation data.

---

### `log_whylogs_profile(dataset, name)`

Creates and logs a whylogs statistical profile.

```python
log_whylogs_profile(
    df_cleaned,
    name="cleaned",
)
```

Use it to record distributions, missingness, cardinality, ranges, and other dataset-quality characteristics.

Create profiles for meaningful stages, for example:

```python
log_whylogs_profile(raw_df, name="raw")
log_whylogs_profile(cleaned_df, name="cleaned")
log_whylogs_profile(training_df, name="training")
```

Ensure the dataset passed to the function matches the name. Do not label a `df_cleaned` profile as `"filtered"` or `"augmented"` unless that is the actual DataFrame.

---

### `save_dataset_manifest(run_id, files_or_path, write_manifest=False)`

Stores lightweight dataset location and the size of dataset.

```python
save_dataset_manifest(
    run_id=run.info.run_id,
    files_or_path="data/train_test_combined.csv",
    write_manifest=False,
)
```

Use it when downstream services need a deterministic data location, file size, or manifest.

Use it mainly with `write_manifest=False`.

---

## 6. Time-series and drift logging

Import:

```python
from certain_library.data_analysis.log_timeseries import timestamp_analysis
from certain_library.data_analysis.log_drift_metrics import log_drift_metrics
```

### `timestamp_analysis(...)`

Records the temporal characteristics of train and test datasets.

```python
timestamp_analysis(
    train_timestamps=train_timestamps,
    test_timestamps=test_timestamps,
    output_dir="timestamps",
)
```

Use it for time-series workflows to document:

- train/test time ranges;
- ordering;
- gaps;
- overlap;
- temporal split characteristics.

---

### `log_drift_metrics(train_df, test_df, run_id, model_info=None)`

Computes and logs drift statistics between reference and comparison datasets.

```python
drift_artifact = log_drift_metrics(
    train_df,
    test_df,
    run_id=run.info.run_id,
    model_info={
        "experiment_id": run.info.experiment_id,
    },
)
```

Use it when comparing:

- train versus test;
- training versus production;
- historical versus current data;
- baseline versus candidate datasets.

Avoid creating a second custom drift artifact from an undefined variable. Use the object returned by `log_drift_metrics`, or explicitly construct the artifact from known data.

---

## 7. Data-technique logging

Import:

```python
from certain_library.data_analysis.log_data_techniques import (
    log_data_techniques,
)
```

### `log_data_techniques(definition)`

Documents preprocessing and augmentation methods.

```python
log_data_techniques(
    {
        "data_technique_stage": "preprocessing",
        "techniques": {
            "imputation": {
                "method": "ffill",
                "parameters": {},
                "stage": "preprocessing",
            },
            "noise_injection": {
                "method": "additive_gaussian",
                "parameters": {"noise_factor": 0.01},
                "stage": "augmentation",
            },
        },
    }
)
```

Use it once the applied data pipeline is known. Log only techniques actually used by the workflow.

---

## 8. Model logging

Import:

```python
from certain_library.train_monitor.log_model import (
    log_model_info,
    log_model_architecture,
    log_model_hyperparameters,
    log_model_signature,
)
```

### `log_model_info(model_information=...)`

Records general model identity.

```python
log_model_info(
    model_information={
        "model_name": "XGBRegressor",
        "model_version": "1.0",
        "framework": "xgboost",
        "framework_version": xgb.__version__,
    }
)
```

---

### `log_model_architecture(...)`

Documents the training architecture and optimization setup.

```python
log_model_architecture(
    losses=["rmse"],
    optimizer={
        "name": "gbdt",
        "learning_rate": 0.05,
        "n_estimators": 100,
        "max_depth": 6,
    },
    regularization="none",
    early_stopping=False,
)
```

Use it for structured model-design metadata that is richer than flat parameters.

---

### `log_model_hyperparameters(...)`

Records model hyperparameters through the model-specific logging layer.

Example:

```python
log_model_hyperparameters(
    {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.05,
    }
)
```

Use this when consumers expect model hyperparameters in the model metadata schema. Use `log_params` as the general MLflow-searchable record.

This helper is imported in the example but not called.

---

### `log_model_signature(model, X, y)`

Infers and records model input/output schema.

```python
log_model_signature(
    final_model,
    X_train,
    y_train,
)
```

Use it after fitting the final model. Pass representative data with the same column names and types expected during inference.

---

## 9. Git and runtime provenance

Import:

```python
from certain_library.git_tracking import log_git_metadata
from certain_library.metadata.artifact_metadata import (
    collect_runtime_environment,
    save_runtime_env_as_artifact,
)
```

### `log_git_metadata()`

Logs Git commit, branch, author, remote, and dirty-state metadata.

```python
git_metadata = log_git_metadata()
```

Call it once near the beginning of the parent run.

A dirty repository means the run may not be exactly reproducible from the recorded commit, so preserve the dirty-state tag.

---

### `collect_runtime_environment()`

Collects information about the execution environment.

```python
environment = collect_runtime_environment()
```

Typical content can include Python, platform, package, hardware, or environment details.

---

### `save_runtime_env_as_artifact(environment)`

Saves the collected environment as an MLflow artifact.

```python
environment = collect_runtime_environment()
save_runtime_env_as_artifact(environment)
```

Call these functions together near the start of the parent run.

---

## 10. Experiment governance

Import:

```python
from certain_library.compliance.log_experiment_governance import (
    log_ai_actors,
    log_labeling_procedures,
)
```

### `log_ai_actors(...)`

Records organizations and people responsible for the AI system.

```python
log_ai_actors(
    auditor="CERTAIN Project Consortium",
    organization="Open Power System Data Initiative",
    use_manual_info=True,
    ai_provider_name="Model Development Team",
    ai_provider_role="model development and training",
    ai_deployer_name="Energy Operations Team",
    ai_deployer_role="operational deployment and monitoring",
)
```

Use it once per parent run or experiment when responsibility information changes.

---

### `log_labeling_procedures(...)`

Documents data-labeling and quality-assurance processes.

```python
log_labeling_procedures(
    quality_assurance_methods=[
        "automated sensor validation",
        "cross-source verification",
    ],
    procedures=[
        {
            "description": "Description of the labeling process",
            "annotation_tool": "data pipeline",
            "annotators": ["automated reporting system"],
            "link": "https://example.com/documentation",
        }
    ],
)
```

Use it for supervised labels, automatically generated targets, sensor measurements, human annotations, or externally sourced labels.

---

## 11. Governance and oversight

Import:

```python
from certain_library.compliance.log_governance import (
    log_risk,
    log_human_oversight,
    log_transparency_measure,
    log_change,
)
```

### `log_risk(risks)`

Logs identified AI and data risks.

```python
log_risk(
    [
        {
            "risk_type": "data_drift",
            "risk_description": "Consumption patterns may change over time.",
            "risk_level": 0.6,
        }
    ]
)
```

Use it for risks such as bias, drift, privacy, robustness, safety, security, and misuse.

---

### `log_human_oversight(mechanisms)`

Documents human review and intervention processes.

```python
log_human_oversight(
    [
        {
            "oversight_type": "human-in-the-loop",
            "description": "Experts review predictions before operational use.",
            "implementation_details": "Weekly review and approval process.",
        }
    ]
)
```

Use it when human review, override, escalation, or auditing is part of the system.

---

### `log_transparency_measure(measures)`

Records transparency and disclosure mechanisms.

```python
log_transparency_measure(
    [
        {
            "measure_type": ["model_card", "data_sheet"],
            "measure_value": [
                "https://example.com/model-card",
                "https://example.com/data-sheet",
            ],
            "description": "Documentation for model and data.",
        }
    ]
)
```

Use it for model cards, data sheets, explainability reports, user notices, and public documentation.

---

### `log_change(changes)`

Records meaningful lifecycle changes.

```python
log_change(
    [
        {
            "change_description": "Retrained model with updated data.",
            "changed_by": "data_science_team",
        }
    ]
)
```

Use it for retraining, model replacement, dataset changes, policy updates, architecture changes, or deployment modifications.

---

## 12. Documentation and explainability

Import:

```python
from certain_library.compliance.log_documentation import (
    log_declarations_of_conformity,
    log_visual_documentations,
    log_explainable_ai,
)
```

### `log_declarations_of_conformity(...)`

Records declarations and standards references.

```python
log_declarations_of_conformity(
    issuer="CERTAIN Project Consortium",
    version="v1.0",
    standard_references=[
        "ISO/IEC 42001:2023",
        "EU AI Act Art. 13",
    ],
    declarations=[
        {
            "filename": "declaration.pdf",
            "file_type": "pdf",
            "mime_type": "application/pdf",
            "description": "Declaration of conformity.",
        }
    ],
)
```

Use it when compliance declarations exist or are generated.

---

### `log_visual_documentations(...)`

Registers visual documentation and links to its artifacts.

```python
log_visual_documentations(
    stage="evaluation",
    generated_by="matplotlib",
    model_version="1.0",
    documents=[
        {
            "filename": "feature_importance.png",
            "file_type": "png",
            "description": "Feature importance chart.",
            "tags": ["feature_importance"],
            "link_to_artifacts": "mlflow://artifacts/feature_importance.png",
        }
    ],
)
```

Use it for plots, diagrams, dashboards, screenshots, and evaluation graphics.

Upload the actual file with `tracker.log_artifact` as well; this function records its documentation metadata.

---

### `log_explainable_ai(...)`

Records explainability outputs.

```python
log_explainable_ai(
    feature_names=["temperature", "hour", "wind"],
    feature_values=["0.42", "0.31", "0.17"],
    implementation_details=(
        "XGBoost built-in gain-based feature importance."
    ),
)
```

Use it for feature importance, SHAP, LIME, counterfactuals, rule explanations, or other model-explanation methods.

---

## 13. Deployment logging

Import:

```python
from certain_library.compliance.log_deployment import (
    log_model_packaging,
    log_build_testing,
    log_standards,
    log_interface,
    log_decommissioning,
)
```

### `log_model_packaging(...)`

Documents how the model is packaged and containerized.

```python
log_model_packaging(
    deployment_id="dep-energy-xgb-prod",
    model_id="energy-load-xgb-v1",
    packaging_format="mlflow_model",
    dependencies=["xgboost", "scikit-learn", "fastapi"],
    containerization_details={
        "base_image": "python:3.11-slim",
        "cpu": "2",
        "memory": "4GB",
        "port": 8090,
    },
)
```

Use it during build or deployment preparation.

---

### `log_build_testing(...)`

Records build and test results.

```python
log_build_testing(
    deployment_id="dep-energy-xgb-prod",
    model_id="energy-load-xgb-v1",
    build_status="success",
    build_logs="Container image built successfully.",
    test_type="integration",
    test_results={
        "total": 3,
        "passed": 3,
        "failed": 0,
        "skipped": 0,
        "coverage_pct": 92.0,
    },
)
```

Use it after CI/CD builds, unit tests, integration tests, security scans, or acceptance tests.

---

### `log_standards(...)`

Records applicable standards and regulations.

```python
log_standards(
    deployment_id="dep-energy-xgb-prod",
    model_id="energy-load-xgb-v1",
    standards=[
        {
            "name": "ISO/IEC 42001:2023",
            "description": "AI management system standard.",
            "version": "2023",
        }
    ],
)
```

Use it when standards applicability or compliance status is known.

---

### `log_interface(...)`

Documents the deployed model interface.

```python
log_interface(
    deployment_id="dep-energy-xgb-prod",
    model_id="energy-load-xgb-v1",
    interface_type="REST API",
    specifications=(
        "POST /predict\n"
        "GET /health"
    ),
    version="v1.0",
    documentation_link="http://localhost:8090/docs",
)
```

Use it for REST, gRPC, batch, event-stream, command-line, or embedded interfaces.

---

### `log_decommissioning(...)`

Records the end of a deployment lifecycle.

```python
log_decommissioning(
    deployment_id="dep-energy-xgb-prod",
    model_id="energy-load-xgb-v1",
    decommissioning_actions=[
        "stop inference service",
        "archive model artifacts",
        "notify operations team",
    ],
    reason="Demo completed.",
    procedure_details="Service was stopped after validation.",
)
```

Use it when a model or endpoint is retired, replaced, disabled, or archived.

---
