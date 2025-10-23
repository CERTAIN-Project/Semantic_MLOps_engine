-- Initialize databases for MLflow and certain_library
-- Note: PostgreSQL doesn't support CREATE DATABASE IF NOT EXISTS
-- These commands will only run during initial database setup

CREATE DATABASE mlflow;

CREATE DATABASE certain_db;

-- Grant permissions (optional, since we're using postgres user)
GRANT ALL PRIVILEGES ON DATABASE mlflow TO postgres;

GRANT ALL PRIVILEGES ON DATABASE certain_db TO postgres;