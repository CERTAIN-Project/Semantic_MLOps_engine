import os
from sqlalchemy import create_engine, MetaData, Table
from sqlalchemy.dialects.postgresql import insert
from dotenv import load_dotenv
import psycopg2

load_dotenv()
TARGET_DB = os.getenv("TARGET_DB")
if TARGET_DB is None:
    raise ValueError("TARGET_DB is not set in the environment or .env file")
target_engine = create_engine(TARGET_DB)


def insert_dataframe(df, table_name: str, batch_size: int = 1000):
    """
    Insert or update rows in the database table using upsert with batch processing.

    Args:
        df (pd.DataFrame): The DataFrame containing the data to insert.
        table_name (str): The name of the target database table.
        batch_size (int): The number of rows to process in each batch.
    """
    metadata = MetaData()
    table = Table(table_name, metadata, autoload_with=target_engine)

    # Determine primary key and updateable columns once before looping
    pk_cols = [c.name for c in table.primary_key.columns]
    update_cols = [c.name for c in table.columns if not c.primary_key]

    with target_engine.begin() as conn:
        for i in range(0, len(df), batch_size):
            # Process the DataFrame in batches
            batch = df.iloc[i : i + batch_size]
            rows = batch.to_dict(
                orient="records"
            )  # Convert batch to list of dictionaries

            # Create an insert statement
            stmt = insert(table).values(rows)

            if update_cols:
                # If non-PK columns exist, perform an UPSERT (update on conflict)
                update_dict = {c: stmt.excluded[c] for c in update_cols}
                stmt = stmt.on_conflict_do_update(
                    index_elements=pk_cols,
                    set_=update_dict,
                )
            else:
                # If all columns are part of the PK, do nothing on conflict
                stmt = stmt.on_conflict_do_nothing(index_elements=pk_cols)

            # Execute the statement
            conn.execute(stmt)


def _to_python(value):
    # convert numpy/pandas scalars to native python
    try:
        import numpy as np
        if isinstance(value, (np.integer, np.floating, np.bool_)):
            return value.item()
    except Exception:
        pass
    # convert NaN to None
    try:
        import math
        if isinstance(value, float) and math.isnan(value):
            return None
    except Exception:
        pass
    return value

def bulk_upsert_metrics(engine, rows, chunk_size: int = 500):
    """
    rows: DataFrame or iterable of dicts/tuples with keys/order:
      run_id, data_id, key, value, timestamp, data_stage, is_NaN
    Performs chunked upsert using psycopg2.extras.execute_values for performance.
    """
    import pandas as pd
    import psycopg2.extras

    cols = ("run_id", "data_id", "key", "value", "timestamp", "data_stage", "is_nan")

    # reflect table and map expected logical names (case-insensitive) to actual names
    metadata = MetaData()
    table = Table("data_metrics", metadata, autoload_with=engine)
    available = [c.name for c in table.columns]

    actual_cols = []
    for expected in cols:
        matches = [c for c in available if c.lower() == expected.lower()]
        if not matches:
            raise ValueError(f"Expected column '{expected}' not found in data_metrics. Got: {available}")
        actual_cols.append(matches[0])

    quoted_cols = ", ".join(f'"{c}"' for c in actual_cols)
    pk_cols = actual_cols[:3]  # run_id, data_id, key
    insert_sql = f"""
    INSERT INTO data_metrics ({quoted_cols})
    VALUES %s
    ON CONFLICT ({', '.join(f'"{c}"' for c in pk_cols)})
      DO UPDATE SET {', '.join(f'"{c}" = EXCLUDED."{c}"' for c in actual_cols[3:])}
    """

    # Accept DataFrame directly
    if isinstance(rows, pd.DataFrame):
        rows = rows.to_dict(orient="records")

    # Ensure we have a concrete list (accept generators)
    rows = list(rows)

    if not rows:
        return

    # normalize values into tuples matching cols, deduplicate by primary key (last-wins)
    dedup = {}
    for r in rows:
        if isinstance(r, dict):
            lower_map = {str(k).lower(): v for k, v in r.items()}
            tup = tuple(_to_python(lower_map.get(c)) for c in cols)
        elif hasattr(r, "to_dict"):
            d = r.to_dict()
            lower_map = {str(k).lower(): v for k, v in d.items()}
            tup = tuple(_to_python(lower_map.get(c)) for c in cols)
        elif isinstance(r, (list, tuple)):
            if len(r) != len(cols):
                raise ValueError("Row tuple length doesn't match expected columns")
            tup = tuple(_to_python(v) for v in r)
        else:
            raise TypeError(f"Unsupported row type for bulk_upsert_metrics: {type(r)}")

        pk = tuple(tup[:3])  # run_id, data_id, key
        dedup[pk] = tup  # overwrite duplicates; last occurrence wins

    norm_rows = list(dedup.values())

    # chunk and execute
    for i in range(0, len(norm_rows), chunk_size):
        chunk = norm_rows[i : i + chunk_size]
        conn = engine.raw_connection()
        try:
            cur = conn.cursor()
            psycopg2.extras.execute_values(
                cur,
                insert_sql,
                chunk,
                template=None,  # default: (%s,%s,...)
                page_size=len(chunk),
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cur.close()
            conn.close()