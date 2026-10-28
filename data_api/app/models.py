from sqlalchemy import (
    Column,
    String,
    Float,
    Integer,
    ForeignKey,
    Boolean,
    JSON,
    BigInteger,
    PrimaryKeyConstraint,
    ForeignKeyConstraint,
)
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy import Numeric
import uuid

Base = declarative_base()


class Experiments(Base):
    __tablename__ = "experiments"

    experiment_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    experiment_name = Column(String, nullable=False, default="default")
    experiment_stage = Column(String, nullable=False, default="Active")
    lifecycle_stage = Column(String, nullable=False, default="data_processing")
    description = Column(String, nullable=True)
    creation_time = Column(Numeric, nullable=False, default=1672531200)
    last_update_time = Column(Numeric, nullable=False, default=1672531200)

    runs = relationship(
        "Runs", back_populates="experiments", cascade="all, delete-orphan"
    )
    tags = relationship(
        "ExperimentTag", back_populates="experiments", cascade="all, delete-orphan"
    )
    deployment = relationship(
        "ModelDeployed", back_populates="experiments", cascade="all, delete-orphan"
    )
    ai_actors = relationship(
        "AIActors", back_populates="experiments", cascade="all, delete-orphan"
    )
    labeling_procedures = relationship(
        "LabelingProcedures", back_populates="experiments", cascade="all, delete-orphan"
    )
    human_oversightmechanism = relationship(
        "HumanOversightMechanism",
        back_populates="experiments",
        cascade="all, delete-orphan",
    )
    transparency_measure = relationship(
        "TransparencyMeasure",
        back_populates="experiments",
        cascade="all, delete-orphan",
    )
    risks = relationship(
        "Risk", back_populates="experiments", cascade="all, delete-orphan"
    )


class Runs(Base):
    __tablename__ = "runs"

    run_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    run_name = Column(String)
    parent_id = Column(String)
    source_type = Column(String)
    source_name = Column(String)
    user_id = Column(String)
    status = Column(String)
    start_time = Column(Numeric)  # Changed from BigInteger to Numeric
    end_time = Column(Numeric)  # Changed from BigInteger to Numeric
    source_version = Column(String)

    experiment_id = Column(
        String,
        ForeignKey("experiments.experiment_id"),
        nullable=False,
        default=lambda: str(uuid.uuid4()),
    )

    experiment = relationship("Experiments", back_populates="runs")
    deployment = relationship("ModelDeployed", back_populates="run")
    tags = relationship("RunTags", back_populates="run", cascade="all, delete-orphan")
    code = relationship(
        "RunCode", back_populates="run", uselist=False, cascade="all, delete-orphan"
    )
    logs = relationship("RunLogs", back_populates="run", cascade="all, delete-orphan")
    model_architecture = relationship(
        "ModelArchitecture",
        back_populates="run",
        uselist=False,
        cascade="all, delete-orphan",
    )
    declaration_of_conformity = relationship(
        "DeclarationOfConformity",
        back_populates="run",
        uselist=False,
        cascade="all, delete-orphan",
    )
    visual_documentation = relationship(
        "VisualDocumentation",
        back_populates="run",
        uselist=False,
        cascade="all, delete-orphan",
    )
    explainable_ai_features = relationship(
        "ExplainableAIFeatures",
        back_populates="run",
        uselist=False,
        cascade="all, delete-orphan",
    )
    # Link to a runtime environment record (one-per-run when available)
    runtime_environment = relationship(
        "RuntimeEnvironment", back_populates="run", uselist=False
    )


class ExperimentsTags(Base):
    __tablename__ = "experiments_tags"

    experiment_id = Column(
        String,
        ForeignKey("experiments.experiment_id"),
        primary_key=True,
        nullable=False,
        default=lambda: str(uuid.uuid4()),
    )
    key = Column(String, primary_key=True, nullable=False)
    value = Column(String)

    experiment = relationship("Experiments", back_populates="tags")


class ModelDeployed(Base):
    __tablename__ = "model_deployed"

    experiment_id = Column(
        String,
        ForeignKey("experiments.experiment_id"),
        primary_key=True,
        nullable=False,
        default=lambda: str(uuid.uuid4()),
    )
    model_id = Column(
        String, primary_key=True, nullable=False, default=lambda: str(uuid.uuid4())
    )
    deployment_id = Column(
        String, primary_key=True, nullable=False, default=lambda: str(uuid.uuid4())
    )

    deployed_time = Column(BigInteger, nullable=False, default=1672531200)
    model_version = Column(String)
    endpoint = Column(String)
    model_format = Column(String)
    size = Column(String)
    description = Column(String)
    user_id = Column(String)
    current_stage = Column(String)
    run_id = Column(
        String, ForeignKey("runs.run_id"), default=lambda: str(uuid.uuid4())
    )
    location = Column(String)
    status = Column(String)
    model_cateory = Column(String)

    experiment = relationship("Experiment", back_populates="deployment")
    run = relationship("Run", back_populates="deployment")
    runtime_environment = relationship(
        "RuntimeEnvironment",
        back_populates="deployment",
        uselist=False,
        cascade="all, delete-orphan",
    )
    drift_metric = relationship(
        "DriftMetric",
        back_populates="deployment",
        cascade="all, delete-orphan",
    )
    monitor_logs = relationship(
        "MonitorLog", back_populates="deployment", cascade="all, delete-orphan"
    )
    build_and_integration_testing = relationship(
        "BuildAndIntegrationTesting",
        back_populates="deployment",
        uselist=False,
        cascade="all, delete-orphan",
    )
    interface = relationship(
        "Interface", back_populates="experiments", cascade="all, delete-orphan"
    )
    standard = relationship(
        "Standard", back_populates="experiments", cascade="all, delete-orphan"
    )
    decomissioning = relationship(
        "Decomissioning", back_populates="experiments", cascade="all, delete-orphan"
    )


class RuntimeEnvironment(Base):
    __tablename__ = "runtime_environment"

    deployment_id = Column(
        String, primary_key=True, nullable=False, default=lambda: str(uuid.uuid4())
    )
    experiment_id = Column(
        String, primary_key=True, nullable=False, default=lambda: str(uuid.uuid4())
    )
    model_id = Column(
        String, primary_key=True, nullable=False, default=lambda: str(uuid.uuid4())
    )
    # Optional run_id to link a runtime environment to the run that produced it
    run_id = Column(String, ForeignKey("runs.run_id"), nullable=True)

    server_name = Column(String)
    performance = Column(Integer)
    # Combined runtime/python environment details captured from artifacts:
    #   - certain/metadata/runtime_env.json (python_version, platform, env_vars, in_docker, ...)
    #   - certain/model/python_env.yaml (python version + pip build_dependencies/dependencies)
    details = Column(JSON)

    __table_args__ = (
        ForeignKeyConstraint(
            ["experiment_id", "deployment_id", "model_id"],
            [
                "model_deployed.experiment_id",
                "model_deployed.deployment_id",
                "model_deployed.model_id",
            ],
        ),
    )

    deployment = relationship(
        "ModelDeployed", back_populates="runtime_environment", uselist=False
    )

    # Back-reference to the run that produced the runtime environment artifact
    run = relationship("Runs", back_populates="runtime_environment", uselist=False)


class MonitorLog(Base):
    __tablename__ = "monitor_logs"

    deployment_id = Column(
        String, primary_key=True, nullable=False, default=lambda: str(uuid.uuid4())
    )
    experiment_id = Column(
        String, primary_key=True, nullable=False, default=lambda: str(uuid.uuid4())
    )
    model_id = Column(
        String, primary_key=True, nullable=False, default=lambda: str(uuid.uuid4())
    )
    log_id = Column(
        String,
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )

    message = Column(String)

    __table_args__ = (
        ForeignKeyConstraint(
            ["experiment_id", "deployment_id", "model_id"],
            [
                "model_deployed.experiment_id",
                "model_deployed.deployment_id",
                "model_deployed.model_id",
            ],
        ),
    )

    deployment = relationship("ModelDeployed", back_populates="monitor_logs")


class DriftMetric(Base):
    __tablename__ = "drift_metrics"

    experiment_id = Column(
        String, primary_key=True, nullable=False, default=lambda: str(uuid.uuid4())
    )
    # Make deployment_id optional (nullable). Some artifacts may not include
    # a deployment identifier; keep the column but do not require it in the
    # primary key so rows can be inserted without a deployment_id.
    deployment_id = Column(String, nullable=True)
    model_id = Column(
        String, primary_key=True, nullable=False, default=lambda: str(uuid.uuid4())
    )
    value = Column(Integer)
    timestamp = Column(Numeric, nullable=False, default=1672531200)

    __table_args__ = (
        ForeignKeyConstraint(
            ["experiment_id", "deployment_id", "model_id"],
            [
                "model_deployed.experiment_id",
                "model_deployed.deployment_id",
                "model_deployed.model_id",
            ],
        ),
    )

    deployment = relationship(
        "ModelDeployed", back_populates="drift_metric", uselist=False
    )


class RunTags(Base):
    __tablename__ = "runs_tags"

    run_id = Column(
        String,
        ForeignKey("runs.run_id"),
        primary_key=True,
        nullable=False,
        default=lambda: str(uuid.uuid4()),
    )
    key = Column(String, primary_key=True, nullable=False)
    value = Column(String, nullable=False)

    run = relationship("Run", back_populates="tags")


class RunCode(Base):
    __tablename__ = "runs_code"

    run_id = Column(
        String,
        ForeignKey("runs.run_id"),
        primary_key=True,
        nullable=False,
        default=lambda: str(uuid.uuid4()),
    )
    git_commit_hash = Column(String, nullable=True)
    git_commit_short = Column(String, nullable=True)
    git_branch = Column(String, nullable=True)
    git_message = Column(String, nullable=True)
    git_author = Column(String, nullable=True)
    git_author_email = Column(String, nullable=True)
    name = Column(String)

    run = relationship("Run", back_populates="code", uselist=False)


class RunLogs(Base):
    __tablename__ = "runs_logs"

    run_id = Column(
        String,
        ForeignKey("runs.run_id"),
        primary_key=True,
        nullable=False,
        default=lambda: str(uuid.uuid4()),
    )
    log_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    log_type = Column(String)
    log_message = Column(String)
    log_creation_time = Column(Numeric, nullable=False, default=1672531200)

    run = relationship("Run", back_populates="logs")


class Data(Base):
    __tablename__ = "data"

    run_id = Column(
        String,
        ForeignKey("runs.run_id"),
        primary_key=True,
        nullable=False,
        default=lambda: str(uuid.uuid4()),
    )
    data_id = Column(
        String,
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    data_stage = Column(String)
    data_type = Column(String)
    data_source = Column(String)
    data_version = Column(String)
    data_location = Column(String)
    data_acquisition_method = Column(String)
    data_size = Column(Float)
    data_format = Column(String)
    creation_time = Column(Numeric, nullable=False, default=1672531200)
    last_update_time = Column(Numeric, nullable=False, default=1672531200)

    run = relationship("Run", back_populates="data")
    data_resources = relationship(
        "DataResources", back_populates="data", cascade="all, delete-orphan"
    )
    data_hyperparameters = relationship(
        "DataTechniquesHyperparameters",
        back_populates="data",
        cascade="all, delete-orphan",
    )
    data_techniques = relationship(
        "DataTechniques", back_populates="data", cascade="all, delete-orphan"
    )
    data_metrics = relationship(
        "DataMetrics", back_populates="data", cascade="all, delete-orphan"
    )


class DataResources(Base):
    __tablename__ = "data_resources"

    run_id = Column(String, nullable=False, default=lambda: str(uuid.uuid4()))
    data_id = Column(String, nullable=False, default=lambda: str(uuid.uuid4()))
    key = Column(String, nullable=False)
    stage = Column(String)
    value = Column(String)
    timestamp = Column(Numeric, nullable=False, default=1672531200)

    __table_args__ = (
        PrimaryKeyConstraint("run_id", "data_id", "key"),
        ForeignKeyConstraint(["run_id", "data_id"], ["data.run_id", "data.data_id"]),
    )

    data = relationship("Data", back_populates="data_resources")


class DataTechniquesHyperparameters(Base):
    __tablename__ = "data_hyperparameters"

    run_id = Column(
        String, primary_key=True, nullable=False, default=lambda: str(uuid.uuid4())
    )
    data_id = Column(
        String, primary_key=True, nullable=False, default=lambda: str(uuid.uuid4())
    )
    technique_name = Column(String, primary_key=True, nullable=False)
    technique_parameter_name = Column(String, primary_key=True, nullable=False)
    technique_parameter_value = Column(String, nullable=False)
    value = Column(String)

    __table_args__ = (
        PrimaryKeyConstraint(
            "run_id", "data_id", "technique_name", "technique_parameter_name"
        ),
        ForeignKeyConstraint(
            ["run_id", "data_id"],
            ["data.run_id", "data.data_id"],
        ),
    )

    data = relationship("Data", back_populates="data_hyperparameters")


class DataTechniques(Base):
    __tablename__ = "data_techniques"

    run_id = Column(String, nullable=False, default=lambda: str(uuid.uuid4()))
    data_id = Column(String, nullable=False, default=lambda: str(uuid.uuid4()))
    technique_name = Column(ARRAY(String), nullable=False)
    data_technique_stage = Column(String)
    # JSON blob containing method, library, parameters, notes, etc.
    technique_details = Column(JSON)

    __table_args__ = (
        PrimaryKeyConstraint("run_id", "data_id", "technique_name"),
        ForeignKeyConstraint(
            ["run_id", "data_id"],
            ["data.run_id", "data.data_id"],
        ),
    )

    data = relationship("Data", back_populates="data_techniques")


class DataMetrics(Base):
    __tablename__ = "data_metrics"

    run_id = Column(String, nullable=False, default=lambda: str(uuid.uuid4()))
    data_id = Column(String, nullable=False, default=lambda: str(uuid.uuid4()))
    key = Column(String, nullable=False)
    value = Column(String)
    timestamp = Column(Numeric, nullable=False, default=1672531200)
    data_stage = Column(String)
    is_NaN = Column(Boolean)

    __table_args__ = (
        PrimaryKeyConstraint("run_id", "data_id", "key"),
        ForeignKeyConstraint(
            ["run_id", "data_id"],
            ["data.run_id", "data.data_id"],
        ),
    )

    data = relationship("Data", back_populates="data_metrics")


class DataSignatures(Base):
    __tablename__ = "data_signatures"

    run_id = Column(String, nullable=False, default=lambda: str(uuid.uuid4()))
    data_id = Column(String, nullable=False, default=lambda: str(uuid.uuid4()))
    signature = Column(JSON)

    __table_args__ = (
        PrimaryKeyConstraint("run_id", "data_id"),
        ForeignKeyConstraint(
            ["run_id", "data_id"],
            ["data.run_id", "data.data_id"],
        ),
    )

    run = relationship("Runs", back_populates="data_signatures")


class ModelArchitecture(Base):
    __tablename__ = "model_architecture"

    run_id = Column(
        String,
        ForeignKey("runs.run_id"),
        primary_key=True,
        nullable=False,
        default=lambda: str(uuid.uuid4()),
    )
    model_id = Column(
        String, primary_key=True, nullable=False, default=lambda: str(uuid.uuid4())
    )
    model_version = Column(Integer, nullable=False, default=1)
    architecture_name = Column(String)
    activation_function = Column(String)
    loss_function = Column(String)
    optimizer = Column(String)
    layer_structure = Column(JSON)
    framework = Column(String)
    metrics = Column(ARRAY(String))

    input_shape = Column(String)
    output_shape = Column(String)

    number_of_layers = Column(Integer)
    number_of_total_parameters = Column(Integer)
    number_of_trainable_parameters = Column(Integer)
    number_of_non_trainable_parameters = Column(Integer)

    creation_time = Column(Numeric, nullable=False, default=1672531200)

    run = relationship("Runs", back_populates="model_architecture")

    checkpoints = relationship(
        "Checkpoints", back_populates="model_architecture", cascade="all, delete-orphan"
    )
    model_hyperparameters = relationship(
        "ModelHyperparameters",
        back_populates="model_architecture",
        cascade="all, delete-orphan",
    )
    model_metrics = relationship(
        "ModelMetrics",
        back_populates="model_architecture",
        cascade="all, delete-orphan",
    )
    last_model_metrics = relationship(
        "LastModelMetrics",
        back_populates="model_architecture",
        cascade="all, delete-orphan",
    )
    weight_distribution = relationship(
        "WeightDistribution",
        back_populates="model_architecture",
        cascade="all, delete-orphan",
    )
    resources = relationship(
        "Resources", back_populates="model_architecture", cascade="all, delete-orphan"
    )
    examples = relationship(
        "Examples", back_populates="model_architecture", cascade="all, delete-orphan"
    )


class Checkpoints(Base):
    __tablename__ = "checkpoints"

    checkpoint_id = Column(String, default=lambda: str(uuid.uuid4()))
    run_id = Column(String, nullable=False, default=lambda: str(uuid.uuid4()))
    model_id = Column(String, nullable=False, default=lambda: str(uuid.uuid4()))
    checkpoint_name = Column(String)
    checkpoint_location = Column(String)
    creation_time = Column(Numeric, nullable=False, default=1672531200)

    __table_args__ = (
        PrimaryKeyConstraint("run_id", "checkpoint_id", "model_id"),
        ForeignKeyConstraint(
            ["run_id", "model_id"],
            ["model_architecture.run_id", "model_architecture.model_id"],
        ),
    )

    model_architecture = relationship("ModelArchitecture", back_populates="checkpoints")


class ModelHyperparameters(Base):
    __tablename__ = "model_hyperparameters"

    run_id = Column(String, nullable=False, default=lambda: str(uuid.uuid4()))
    model_id = Column(String, nullable=False, default=lambda: str(uuid.uuid4()))
    key = Column(String, nullable=False)
    value = Column(String)

    __table_args__ = (
        PrimaryKeyConstraint("run_id", "key", "model_id"),
        ForeignKeyConstraint(
            ["run_id", "model_id"],
            ["model_architecture.run_id", "model_architecture.model_id"],
        ),
    )

    model_architecture = relationship(
        "ModelArchitecture", back_populates="model_hyperparameters"
    )


class ModelMetrics(Base):
    __tablename__ = "model_metrics"

    run_id = Column(String, nullable=False, default=lambda: str(uuid.uuid4()))
    model_id = Column(String, nullable=False, default=lambda: str(uuid.uuid4()))
    key = Column(String, nullable=False)
    value = Column(String, nullable=False)
    step = Column(Integer, nullable=False)
    stage = Column(String, nullable=False)
    is_NaN = Column(Boolean, nullable=False)
    timestamp = Column(Numeric, nullable=False, default=1672531200)

    __table_args__ = (
        PrimaryKeyConstraint("run_id", "key", "model_id", "step", "stage"),
        ForeignKeyConstraint(
            ["run_id", "model_id"],
            ["model_architecture.run_id", "model_architecture.model_id"],
        ),
    )

    model_architecture = relationship(
        "ModelArchitecture", back_populates="model_metrics"
    )


class LastModelMetrics(Base):
    __tablename__ = "last_model_metrics"

    run_id = Column(String, nullable=False, default=lambda: str(uuid.uuid4()))
    model_id = Column(String, nullable=False, default=lambda: str(uuid.uuid4()))
    key = Column(String, nullable=False)
    value = Column(String)
    step = Column(Integer)
    stage = Column(String)
    is_NaN = Column(Boolean)
    timestamp = Column(Numeric, nullable=False, default=1672531200)

    __table_args__ = (
        PrimaryKeyConstraint("run_id", "key", "model_id"),
        ForeignKeyConstraint(
            ["run_id", "model_id"],
            ["model_architecture.run_id", "model_architecture.model_id"],
        ),
    )

    model_architecture = relationship(
        "ModelArchitecture", back_populates="last_model_metrics"
    )


class WeightDistribution(Base):
    __tablename__ = "weight_distribution"

    run_id = Column(String, nullable=False, default=lambda: str(uuid.uuid4()))
    model_id = Column(String, nullable=False, default=lambda: str(uuid.uuid4()))
    layer_name = Column(String, nullable=False)
    std = Column(Float, nullable=False)
    mean = Column(Float, nullable=False)
    step = Column(Integer, nullable=False)
    stage = Column(String, nullable=False)
    is_NaN = Column(Boolean, nullable=False)
    timestamp = Column(Numeric, nullable=False, default=1672531200)

    __table_args__ = (
        PrimaryKeyConstraint("run_id", "model_id", "step", "stage"),
        ForeignKeyConstraint(
            ["run_id", "model_id"],
            ["model_architecture.run_id", "model_architecture.model_id"],
        ),
    )

    model_architecture = relationship(
        "ModelArchitecture", back_populates="weight_distribution"
    )


class Resources(Base):
    __tablename__ = "resources"

    run_id = Column(String, nullable=False, default=lambda: str(uuid.uuid4()))
    model_id = Column(String, nullable=False, default=lambda: str(uuid.uuid4()))
    key = Column(String, nullable=False)
    value = Column(String, nullable=False)
    step = Column(Integer, nullable=False)
    stage = Column(String, nullable=False)
    timestamp = Column(Numeric, nullable=False, default=1672531200)

    __table_args__ = (
        PrimaryKeyConstraint("run_id", "model_id", "key", "step", "stage"),
        ForeignKeyConstraint(
            ["run_id", "model_id"],
            ["model_architecture.run_id", "model_architecture.model_id"],
        ),
    )

    model_architecture = relationship("ModelArchitecture", back_populates="resources")


class Examples(Base):
    __tablename__ = "examples"

    run_id = Column(String, nullable=False, default=lambda: str(uuid.uuid4()))
    model_id = Column(String, nullable=False, default=lambda: str(uuid.uuid4()))
    input = Column(String, nullable=False)
    prediction = Column(String, nullable=False)
    ground_truth = Column(String, nullable=False)

    step = Column(Integer, nullable=False)
    stage = Column(String, nullable=False)
    timestamp = Column(Numeric, nullable=False, default=1672531200)

    __table_args__ = (
        PrimaryKeyConstraint("run_id", "model_id", "input", "step", "stage"),
        ForeignKeyConstraint(
            ["run_id", "model_id"],
            ["model_architecture.run_id", "model_architecture.model_id"],
        ),
    )

    model_architecture = relationship("ModelArchitecture", back_populates="examples")


class IdMapping(Base):
    __tablename__ = "id_mapping"

    run_id = Column(
        String,
        ForeignKey("runs.run_id"),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    model_id = Column(String, default=lambda: str(uuid.uuid4()), nullable=False)
    data_id = Column(String, default=lambda: str(uuid.uuid4()), nullable=False)
    deployment_id = Column(String, default=lambda: str(uuid.uuid4()), nullable=False)


class AIActors(Base):
    __tablename__ = "ai_actors"

    ai_actors_id = Column(
        String,
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    # Use run_id (runs.run_id) instead of experiment_id as the parent reference.
    # This allows ai_actors rows to be associated with a specific run.
    run_id = Column(String, primary_key=True, nullable=False)
    ai_provider = Column(String)
    ai_deployer = Column(String)
    auditor = Column(String)
    organization = Column(String)

    __table_args__ = (
        PrimaryKeyConstraint("ai_actors_id", "run_id"),
        ForeignKeyConstraint(["run_id"], ["runs.run_id"]),
    )


class LabelingProcedures(Base):
    __tablename__ = "labeling_procedures"

    labeling_id = Column(
        String,
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    experiment_id = Column(String, primary_key=True, nullable=False)
    procedure_description = Column(String)
    quality_assurance_methods = Column(String)
    annotator_details = Column(String)
    annotation_tools = Column(ARRAY(String))

    __table_args__ = (
        PrimaryKeyConstraint("labeling_id", "experiment_id"),
        ForeignKeyConstraint(["experiment_id"], ["experiments.experiment_id"]),
    )


class ModelPackaging(Base):
    __tablename__ = "model_packaging"

    packaging_id = Column(
        String,
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    experiment_id = Column(
        String, primary_key=True, nullable=False, default=lambda: str(uuid.uuid4())
    )
    deployment_id = Column(
        String, primary_key=True, nullable=False, default=lambda: str(uuid.uuid4())
    )
    model_id = Column(
        String, primary_key=True, nullable=False, default=lambda: str(uuid.uuid4())
    )
    packaging_format = Column(String)
    dependencies = Column(ARRAY(String))
    containerization_details = Column(String)

    __table_args__ = (
        PrimaryKeyConstraint(
            "packaging_id", "experiment_id", "deployment_id", "model_id"
        ),
        ForeignKeyConstraint(
            ["experiment_id", "deployment_id", "model_id"],
            [
                "model_deployed.experiment_id",
                "model_deployed.deployment_id",
                "model_deployed.model_id",
            ],
        ),
    )

    deployment = relationship("ModelDeployed", back_populates="model_packaging")


class BuildAndIntegrationTesting(Base):
    __tablename__ = "build_and_integration_testing"

    test_id = Column(
        String,
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    experiment_id = Column(String, primary_key=True, nullable=False)
    deployment_id = Column(String, primary_key=True, nullable=False)
    model_id = Column(String, primary_key=True, nullable=False)
    build_status = Column(String)
    build_logs = Column(String)
    build_timestamp = Column(Numeric, nullable=False, default=1672531200)
    test_type = Column(String)
    test_results = Column(String)

    __table_args__ = (
        PrimaryKeyConstraint("test_id", "experiment_id", "deployment_id", "model_id"),
        ForeignKeyConstraint(
            ["experiment_id", "deployment_id", "model_id"],
            [
                "model_deployed.experiment_id",
                "model_deployed.deployment_id",
                "model_deployed.model_id",
            ],
        ),
    )


class DeclarationOfConformity(Base):
    __tablename__ = "declaration_of_conformity"

    declaration_id = Column(
        String,
        default=lambda: str(uuid.uuid4()),
    )
    run_id = Column(String, nullable=False)

    # Metadata
    filename = Column(String, nullable=False)
    file_type = Column(String, nullable=False)
    mime_type = Column(String, nullable=False)
    file_size = Column(Integer)

    link_to_artifacts = Column(String, nullable=True)

    description = Column(String)
    creation_time = Column(Numeric, nullable=False, default=1672531200)
    # Optional deployment_id: some artifacts may include an explicit deployment
    # identifier. Keep it nullable to avoid breaking existing rows.
    deployment_id = Column(String, nullable=True)

    __table_args__ = (
        PrimaryKeyConstraint("declaration_id", "run_id"),
        ForeignKeyConstraint(["run_id"], ["runs.run_id"]),
    )


class VisualDocumentation(Base):
    __tablename__ = "visual_documentation"

    document_id = Column(
        String,
        default=lambda: str(uuid.uuid4()),
    )
    run_id = Column(String, nullable=False)

    # Metadata
    filename = Column(String, nullable=False)
    file_type = Column(String, nullable=False)
    file_size = Column(Integer)

    link_to_artifacts = Column(String, nullable=True)

    description = Column(String)
    tags = Column(ARRAY(String))
    creation_time = Column(Numeric, nullable=False, default=1672531200)

    __table_args__ = (
        PrimaryKeyConstraint("document_id", "run_id"),
        ForeignKeyConstraint(["run_id"], ["runs.run_id"]),
    )


class Standard(Base):
    __tablename__ = "standards"

    # Follow the same shape as DeclarationOfConformity: one artifact row per
    # run, with optional deployment_id when provided by the artifact.
    standard_id = Column(
        String,
        default=lambda: str(uuid.uuid4()),
    )
    run_id = Column(String, nullable=False)

    # Metadata
    name = Column(String, nullable=False)
    description = Column(String, nullable=False)
    version = Column(String, nullable=False)
    publication_date = Column(Integer)

    creation_time = Column(Numeric, nullable=False, default=1672531200)
    # Optional deployment_id: some artifacts may include an explicit deployment
    # identifier. Keep it nullable to avoid breaking existing rows.
    deployment_id = Column(String, nullable=True)
    model_id = Column(String, nullable=True)

    __table_args__ = (
        PrimaryKeyConstraint("standard_id", "run_id", "model_id"),
        ForeignKeyConstraint(["run_id"], ["runs.run_id"]),
    )

class Interface(Base):
    __tablename__ = "interfaces"

    interface_id = Column(
        String,
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    experiment_id = Column(
        String, primary_key=True, nullable=False, default=lambda: str(uuid.uuid4())
    )
    run_id = Column(String, ForeignKey("runs.run_id"), nullable=True)
    deployment_id = Column(
        String, primary_key=True, nullable=False, default=lambda: str(uuid.uuid4())
    )
    model_id = Column(
        String, primary_key=True, nullable=False, default=lambda: str(uuid.uuid4())
    )
    interface_type = Column(String, nullable=False)
    specifications = Column(String)
    version = Column(String)
    documentation_link = Column(String)

    __table_args__ = (
        PrimaryKeyConstraint(
            "interface_id", "experiment_id", "deployment_id", "model_id"
        ),
        ForeignKeyConstraint(
            ["experiment_id", "deployment_id", "model_id"],
            [
                "model_deployed.experiment_id",
                "model_deployed.deployment_id",
                "model_deployed.model_id",
            ],
        ),
    )

    run = relationship("Runs")


class ExplainableAIFeature(Base):
    __tablename__ = "explainable_ai_features"

    feature_id = Column(
        String,
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    run_id = Column(String, nullable=False)
    feature_name = Column(ARRAY(String), nullable=False)
    feature_values = Column(ARRAY(String), nullable=False)
    implementation_details = Column(String)
    # Optional deployment_id stored when artifact contains it
    deployment_id = Column(String, nullable=True)

    __table_args__ = (
        PrimaryKeyConstraint("feature_id", "run_id"),
        ForeignKeyConstraint(["run_id"], ["runs.run_id"]),
    )


class HumanOversightMechanism(Base):
    __tablename__ = "human_oversight_mechanisms"

    mechanism_id = Column(
        String,
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    experiment_id = Column(String, nullable=False)
    run_id = Column(String, ForeignKey("runs.run_id"), nullable=True)
    oversight_type = Column(String, nullable=False)
    description = Column(String)
    implementation_details = Column(String)
    # Optional deployment_id stored when artifact contains it
    deployment_id = Column(String, nullable=True)

    __table_args__ = (
        PrimaryKeyConstraint("mechanism_id", "experiment_id"),
        ForeignKeyConstraint(["experiment_id"], ["experiments.experiment_id"]),
    )

    run = relationship("Runs")


class TransparencyMeasure(Base):
    __tablename__ = "transparency_measures"

    measure_id = Column(
        String,
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    experiment_id = Column(String, nullable=False)
    measure_type = Column(ARRAY(String), nullable=False)
    measure_value = Column(ARRAY(String), nullable=False)
    description = Column(String)
    implementation_details = Column(String)

    __table_args__ = (
        PrimaryKeyConstraint("measure_id", "experiment_id"),
        ForeignKeyConstraint(["experiment_id"], ["experiments.experiment_id"]),
    )


class Risk(Base):
    __tablename__ = "risks"

    risk_id = Column(
        String,
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    experiment_id = Column(String, nullable=False)
    risk_description = Column(String, nullable=False)
    risk_type = Column(String, nullable=False)
    risk_level = Column(Float, nullable=False)

    __table_args__ = (
        PrimaryKeyConstraint("risk_id", "experiment_id"),
        ForeignKeyConstraint(["experiment_id"], ["experiments.experiment_id"]),
    )


class Decomissioning(Base):
    __tablename__ = "decomissioning"
    decomissioning_id = Column(
        String,
        default=lambda: str(uuid.uuid4()),
    )
    experiment_id = Column(String, nullable=False, default=lambda: str(uuid.uuid4()))
    deployment_id = Column(String, nullable=False, default=lambda: str(uuid.uuid4()))
    model_id = Column(String, nullable=False, default=lambda: str(uuid.uuid4()))
    system_name = Column(String)
    decommissioning_plan = Column(String)
    approvals = Column(ARRAY(String))
    data_retention_archive = Column(String)
    migration = Column(String)
    access_removal = Column(String)
    infrastructure_shutdown = Column(String)
    evidence_documentation = Column(ARRAY(String))
    audit_trail = Column(String)
    decomissioning_date = Column(Numeric, nullable=False, default=1672531200)
    decomissioning_actions = Column(ARRAY(String), nullable=False)
    reason = Column(String, nullable=False)
    procedure_details = Column(String)

    __table_args__ = (
        PrimaryKeyConstraint(
            "decomissioning_id", "experiment_id", "deployment_id", "model_id"
        ),
        ForeignKeyConstraint(
            ["experiment_id", "deployment_id", "model_id"],
            [
                "model_deployed.experiment_id",
                "model_deployed.deployment_id",
                "model_deployed.model_id",
            ],
        ),
    )


class ChangeLog(Base):
    __tablename__ = "change_logs"

    log_id = Column(
        String,
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    run_id = Column(String, nullable=False)
    change_description = Column(String, nullable=False)
    changed_by = Column(String, nullable=False)
    change_timestamp = Column(Numeric, nullable=False, default=1672531200)

    __table_args__ = (
        PrimaryKeyConstraint("log_id", "run_id"),
        ForeignKeyConstraint(["run_id"], ["runs.run_id"]),
    )


class TokenizerConfig(Base):
    """Configuration of the tokenizer used for an LLM/NLP run.

    One row per run — stores the tokenizer type, vocabulary details, and all
    encoding settings so the exact tokenisation behaviour can be reproduced.
    """

    __tablename__ = "tokenizer_config"

    tokenizer_id = Column(
        String,
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    run_id = Column(String, nullable=False)
    # Tokenizer identity
    tokenizer_type = Column(
        String, nullable=False
    )  # e.g. "BPE", "WordPiece", "SentencePiece"
    model_name_or_path = Column(String)  # e.g. "bert-base-uncased"
    vocab_size = Column(Integer)
    # Encoding settings
    max_length = Column(Integer)
    padding = Column(String)  # "max_length", "longest", "do_not_pad"
    truncation = Column(Boolean)
    stride = Column(Integer)
    # Special tokens stored as a JSON object {token_name: token_string}
    special_tokens = Column(JSON)
    creation_time = Column(Numeric, nullable=False, default=1672531200)

    __table_args__ = (
        PrimaryKeyConstraint("tokenizer_id", "run_id"),
        ForeignKeyConstraint(["run_id"], ["runs.run_id"]),
    )


class TokenizationStats(Base):
    """Per-split tokenization statistics for an LLM/NLP run.

    One row per (run, split) — e.g. ``split="train"`` and ``split="test"``.
    Captures sequence length distribution, out-of-vocabulary rate, and
    truncation/padding rates so data quality can be tracked over time.
    """

    __tablename__ = "tokenization_stats"

    stats_id = Column(
        String,
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    run_id = Column(String, nullable=False)
    split = Column(String, nullable=False)  # "train", "val", "test", "full", …
    total_sequences = Column(Integer)
    total_tokens = Column(Integer)
    avg_token_length = Column(Float)
    max_token_length = Column(Integer)
    min_token_length = Column(Integer)
    truncation_rate = Column(Float)  # fraction of sequences that were truncated
    padding_rate = Column(Float)  # fraction of sequences that were padded
    oov_rate = Column(Float)  # fraction of tokens that are [UNK]
    creation_time = Column(Numeric, nullable=False, default=1672531200)

    __table_args__ = (
        PrimaryKeyConstraint("stats_id", "run_id"),
        ForeignKeyConstraint(["run_id"], ["runs.run_id"]),
    )
