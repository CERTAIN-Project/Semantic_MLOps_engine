from app.db import SessionLocal
from app.models import Experiments

session = SessionLocal()

print("Creating tables...")

# Example creation
exp = Experiments(
    experiment_id=1,
    experiment_name="Test Experiment",
    lifecycle_stage="active",
    creation_time=1713440000000,
    last_update_time=1713440000000,
)
