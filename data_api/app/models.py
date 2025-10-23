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
    lifecycle_stage = Column(String, nullable=False)
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


class Runs(Base):
    __tablename__ = "runs"

    run_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    run_name = Column(String)
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

    server_name = Column(String)
    performance = Column(Integer)

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
    deployment_id = Column(
        String, primary_key=True, nullable=False, default=lambda: str(uuid.uuid4())
    )
    model_id = Column(
        String, primary_key=True, nullable=False, default=lambda: str(uuid.uuid4())
    )
    value = Column(Integer, primary_key=True)
    timestamp = Column(Numeric, primary_key=True, nullable=False, default=1672531200)

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
    git_commit_hash = Column(String, nullable=False)
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

    run_id = Column(String, nullable=False, default=lambda: str(uuid.uuid4()))
    data_id = Column(String, nullable=False, default=lambda: str(uuid.uuid4()))
    key = Column(String, nullable=False)
    technical_name = Column(String, nullable=False)
    value = Column(String)

    __table_args__ = (
        PrimaryKeyConstraint("run_id", "data_id", "key", "technical_name"),
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
    technique_name = Column(String, nullable=False)
    value = Column(String)

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
    nummber_of_non_trainable_parameters = Column(Integer)

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
        PrimaryKeyConstraint(
            "run_id", "key", "model_id", "step", "stage", "timestamp", "is_NaN", "value"
        ),
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
        PrimaryKeyConstraint(
            "run_id",
            "model_id",
            "layer_name",
            "std",
            "mean",
            "step",
            "stage",
            "timestamp",
            "is_NaN",
        ),
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
        PrimaryKeyConstraint(
            "run_id",
            "model_id",
            "key",
            "value",
            "step",
            "stage",
            "timestamp",
        ),
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
        PrimaryKeyConstraint(
            "run_id",
            "model_id",
            "input",
            "prediction",
            "ground_truth",
            "step",
            "stage",
            "timestamp",
        ),
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
