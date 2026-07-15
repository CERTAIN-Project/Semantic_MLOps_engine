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
            # Normalize values and deduplicate by primary key within the batch
            # (last occurrence wins). This prevents Postgres errors when the
            # same primary key appears multiple times inside a single
            # multi-row INSERT ... ON CONFLICT statement.
            if rows:
                dedup = {}
                new_rows = []
                for idx, r in enumerate(rows):
                    # ensure dict-like
                    if not isinstance(r, dict):
                        # defensive: convert to dict if possible
                        try:
                            r = dict(r)
                        except Exception:
                            pass

                    # Build primary-key tuple; if any PK component is None
                    # include the row index to avoid collapsing truly distinct
                    # rows that don't specify PK values.
                    if pk_cols:
                        pk_vals = tuple(_to_python(r.get(col)) for col in pk_cols)
                        if any(v is None for v in pk_vals):
                            key = (idx,) + pk_vals
                        else:
                            key = pk_vals
                    else:
                        # no primary key defined — fallback to index
                        key = (idx,)

                    # Normalize all values using _to_python to convert numpy
                    # scalars and NaN to native Python types / None.
                    norm = {k: _to_python(v) for k, v in r.items()}
                    dedup[key] = norm

                # Preserve insertion order of last-occurrence wins
                rows = list(dedup.values())

            # Create an insert statement
            # Filter each row to include only columns that exist on the target
            # table (case-insensitive). Also handle a few common aliases
            # produced by artifact exporters (plural -> singular names).
            available_cols = [c.name for c in table.columns]
            lower_to_actual = {c.lower(): c for c in available_cols}
            aliases = {
                "ai_providers": "ai_provider",
                "ai_deployers": "ai_deployer",
                "is_nan": "is_nan",
                "is_nan": "is_nan",
            }

            filtered_rows = []
            for r in rows:
                if not isinstance(r, dict):
                    try:
                        r = dict(r)
                    except Exception:
                        # fallback: skip malformed row
                        continue

                lower_map = {str(k).lower(): v for k, v in r.items()}
                out = {}
                # take direct matches first
                for lk, v in lower_map.items():
                    if lk in lower_to_actual:
                        out[lower_to_actual[lk]] = _to_python(v)
                    elif lk in aliases and aliases[lk] in lower_to_actual:
                        # map plural/list values to singular column; join lists
                        val = v
                        if isinstance(val, (list, tuple)):
                            try:
                                val = ",".join(str(x) for x in val)
                            except Exception:
                                val = str(val)
                        out[lower_to_actual[aliases[lk]]] = _to_python(val)

                # Also attempt simple singularization for unknown plural keys
                for lk, v in list(lower_map.items()):
                    if lk not in lower_to_actual and lk.endswith("s"):
                        singular = lk[:-1]
                        if singular in lower_to_actual and singular not in (
                            k.lower() for k in out.keys()
                        ):
                            out[lower_to_actual[singular]] = _to_python(v)

                filtered_rows.append(out)

            # Build a consistent set of columns to pass to SQLAlchemy: union of
            # allowed table columns intersected with keys we produced. For any
            # missing columns in a row, fill with None so all rows share the
            # same shape. This avoids SQLAlchemy compile errors about
            # unconsumed/unknown column names.
            allowed = set(available_cols)
            # union of keys we intend to send (map actual names)
            keys_union = set()
            for r in filtered_rows:
                for k in r.keys():
                    if k in allowed:
                        keys_union.add(k)

            keys_list = list(keys_union) if keys_union else []
            normalized_rows = []
            for r in filtered_rows:
                nr = {k: r.get(k, None) for k in keys_list}
                normalized_rows.append(nr)
            stmt = insert(table).values(normalized_rows)

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
    # Convert lists/tuples to tuples (hashable) and dicts to stable JSON strings
    try:
        if isinstance(value, (list, tuple)):
            return tuple(_to_python(v) for v in value)
        if isinstance(value, dict):
            import json

            # stable string representation
            try:
                return json.dumps(value, sort_keys=True)
            except Exception:
                return str(value)
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
            raise ValueError(
                f"Expected column '{expected}' not found in data_metrics. Got: {available}"
            )
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
