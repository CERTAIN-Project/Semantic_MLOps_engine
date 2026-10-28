"""Cleanup child-run `data` rows, keeping only parent-run entries.

This script performs a dry-run by default: it reports how many `data` rows
and dependent rows (data_resources, data_metrics, etc.) would be deleted if
we remove entries where the owning run has a non-null `parent_id`.

Run with --apply to actually perform deletions (use with caution).

Usage:
    python -m data_api.scripts.cleanup_child_data --apply

"""

import argparse
from app.target_connector import target_engine
from sqlalchemy import MetaData, Table, select


def main(apply: bool = False):
    metadata = MetaData()
    data_table = Table("data", metadata, autoload_with=target_engine)
    runs_table = Table("runs", metadata, autoload_with=target_engine)

    with target_engine.connect() as conn:
        # find runs that have a parent_id
        q = select(runs_table.c.run_id, runs_table.c.parent_id).where(
            runs_table.c.parent_id != None
        )
        child_runs = [r[0] for r in conn.execute(q).fetchall()]

        if not child_runs:
            print("No child runs found. Nothing to do.")
            return

        # count data rows owned by child runs
        q2 = select(data_table.c.run_id, data_table.c.data_id).where(
            data_table.c.run_id.in_(child_runs)
        )
        rows = conn.execute(q2).fetchall()
        print(f"Found {len(rows)} data rows belonging to child runs.")

        if not apply:
            print("Dry run: no changes made. Re-run with --apply to delete these rows.")
            return

        # perform deletions in dependent tables first (cascade not assumed)
        dependent_tables = [
            "data_resources",
            "data_metrics",
            "data_signatures",
            "data_hyperparameters",
            "data_techniques",
        ]
        for dt in dependent_tables:
            t = Table(dt, metadata, autoload_with=target_engine)
            del_q = t.delete().where(t.c.run_id.in_(child_runs))
            res = conn.execute(del_q)
            print(f"Deleted {res.rowcount} rows from {dt}")

        # delete data rows
        del_q = data_table.delete().where(data_table.c.run_id.in_(child_runs))
        res = conn.execute(del_q)
        print(f"Deleted {res.rowcount} rows from data table")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Apply deletions")
    args = parser.parse_args()
    main(apply=args.apply)
