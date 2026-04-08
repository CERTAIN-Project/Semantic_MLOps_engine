# Certain Library - MLflow Logging Utilities

A comprehensive Python library for logging machine learning experiments, data analysis, and resource monitoring to MLflow. This library provides convenient functions to track datasets, model information, metrics, hyperparameters, data techniques, timestamps, data profiles, and carbon emissions.

## 🚀 Quick Start with Docker

Prerequisites
- Docker Engine (or Docker Desktop) installed and running
- docker-compose v2 (usually included with Docker Desktop)
- Clone repository and run commands from the project root (where docker-compose.yml lives)

1. Create required host folders (Docker volumes cannot be empty)
```bash
# from project root
mkdir docker_data/mlflow_artifacts
mkdir docker_data/postgres
```

3. Build and start all services
```bash
docker compose up --build -d
```
4. Verify services are running

```bash
docker container list
```

```bash
# List running containers
(base) user_name@Mac certain_library % docker container list
CONTAINER ID   IMAGE                             COMMAND                  CREATED             STATUS                       PORTS                              NAMES
f49f98decf2d   certain_library-data_transfer     "uvicorn main:app --…"   About an hour ago   Up About an hour (healthy)   5432/tcp, 0.0.0.0:8001->8001/tcp   certain_data_transfer_api
cbcadc9ece68   certain_library-library_tracker   "/app/start.sh"          About an hour ago   Up About an hour             5432/tcp, 0.0.0.0:8002->8002/tcp   certain_library_tracker
d37978106638   python:3.11-slim                  "sh -c ' pip install…"   About an hour ago   Up About an hour (healthy)   0.0.0.0:5001->5001/tcp             certain_mlflow
1dcdea45ab70   postgres:13                       "docker-entrypoint.s…"   About an hour ago   Up About an hour (healthy)   0.0.0.0:5432->5432/tcp             certain_databases
```

```bash
docker compose ps
```

```bash
# Compose service status
(base) user_name@Mac certain_library % docker compose ps
NAME                        IMAGE                             COMMAND                  SERVICE           CREATED             STATUS                       PORTS
certain_data_transfer_api   certain_library-data_transfer     "uvicorn main:app --…"   data_transfer     About an hour ago   Up About an hour (healthy)   5432/tcp, 0.0.0.0:8001->8001/tcp
certain_databases           postgres:13                       "docker-entrypoint.s…"   postgres          About an hour ago   Up About an hour (healthy)   0.0.0.0:5432->5432/tcp
certain_library_tracker     certain_library-library_tracker   "/app/start.sh"          library_tracker   About an hour ago   Up About an hour             5432/tcp, 0.0.0.0:8002->8002/tcp
certain_mlflow              python:3.11-slim                  "sh -c ' pip install…"   mlflow            About an hour ago   Up About an hour (healthy)   0.0.0.0:5001->5001/tcp
```

5. Run the integration test inside the library tracker container
```bash
docker exec certain_library_tracker python test_docker/test_complete_workflow.py
```

```bash
...
⚡ Resource monitoring stopped for model training
======================================================================
✅ Workflow Complete!
======================================================================

All data has been logged to PostgreSQL database:
  • Experiment metadata
  • Run information
  • Training & test datasets
  • Model hyperparameters
  • Model information
  • Training metrics per epoch
  • Final evaluation metrics
  • Resource usage metrics
  • Hyperparameter search space

Query the database to explore your ML experiments! 🎉

```

6. Access MLflow UI
- Open http://localhost:5001 in your browser

7. Check Data Transfer API health and docs
```bash
curl -X POST "http://localhost:8001/sync/all" 
```

```bash
{"status":"all data synced successfully"}
```

8. Inspect database contents (examples)
```bash
# Row counts
docker exec certain_databases psql -U postgres -d certain_db -c "
SELECT COUNT(*) as total_rows, 'experiments' as table_name FROM experiments 
UNION ALL SELECT COUNT(*), 'runs' FROM runs 
UNION ALL SELECT COUNT(*), 'data_metrics' FROM data_metrics 
UNION ALL SELECT COUNT(*), 'model_metrics' FROM model_metrics 
UNION ALL SELECT COUNT(*), 'id_mapping' FROM id_mapping 
ORDER BY table_name;"

 total_rows |  table_name   
------------+---------------
       1665 | data_metrics
          2 | experiments
        127 | id_mapping
       1908 | model_metrics
        127 | runs
(5 rows)

# Recent experiments
docker exec certain_databases psql -U postgres -d certain_db -c "
SELECT experiment_id, experiment_name, lifecycle_stage, creation_time 
FROM experiments 
ORDER BY creation_time DESC 
LIMIT 5;"

 experiment_id | experiment_name | lifecycle_stage | creation_time 
---------------+-----------------+-----------------+---------------
 1             | 1               | active          | 1761147276257
 0             | 0               | active          | 1761147257252
```

### Next steps (TODO)
- Publish the library to PyPI (or a private index) to simplify deployments and remove the need to build images that include the source.

## 🏗️ Architecture

The project uses a multi-service Docker architecture:

- **PostgreSQL Database** (Port 5432): Stores MLflow tracking data and custom database
- **MLflow Server** (Port 5001): MLflow tracking server with web UI
- **Data Transfer API** (Port 8001): FastAPI service for programmatic access
- **Library Tracker** (Port 8002): Interactive development container
- **Data Lineage API** (Port 3000): PostgREST service for data lineage queries
- **Ontop SPARQL Endpoint** (Port 8080): Virtual knowledge graph — exposes PostgreSQL data as RDF via the AIDOC-AP ontology
- **Database Migrations**: Alembic migrations for database schema

## 📁 Directory Structure (previous info retained)
```
certain_library/
├── __init__.py
├── data_analysis/
│   ├── __init__.py
│   ├── log_data_techniques.py    # Log data preprocessing techniques
│   ├── log_dataset.py           # Log training/testing datasets
│   ├── log_timeseries.py        # Log timestamp analysis
│   └── log_whylogs.py           # Log WhyLogs data profiles
├── resource_monitor/
│   ├── __init__.py
│   └── resource.py              # Track carbon emissions
└── train_monitor/
│   ├── __init__.py
│   ├── log_metrics.py           # Log training metrics
│   └── log_model.py             # Log model information
├── data_api/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── models.py                 # SQLAlchemy models (Base)
│   │   └── mlflow_connector.py       # MLflow & artifacts helpers
│   └── alembic/
│       ├── env.py                    # Alembic environment bootstrap (handles context.config)
│       └── versions/
├── README.md
└── setup.py / pyproject.toml / requirements.txt
```

## Testing

### Option 1: Docker Development (Recommended)

Use the containerized development environment:

```bash
# Start the development environment
docker compose up -d

# Access interactive Python shell
docker exec -it certain_library_tracker bash

# Run the test
python test_docker/test_complete_workflow.py
```

### Environment Variables

The Docker setup handles environment variables automatically. For local development, create a `.env` file:

```env
MLFLOW_DB=postgresql://postgres:postgres@localhost:5432/mlflow_db
TARGET_DB=postgresql://postgres:postgres@localhost:5432/certain_db
MLFLOW_ARTIFACTS=file:///path/to/your/mlflow/artifacts
```

## Usage Examples

### Using the Data Transfer API

The FastAPI service provides programmatic access to data operations:

```bash
# Check API health
curl http://localhost:8001/health

# View API documentation
open http://localhost:8001/docs

# Example API endpoints (adjust based on your implementation)
curl -X POST http://localhost:8001/sync/all

# If needed to run some sync seperatly
curl -X GET http://localhost:8001/experiments
curl -X GET http://localhost:8001/runs
```

### MLflow Integration

Access the MLflow UI and tracking server:

```bash
# MLflow UI is available at browser:
open http://localhost:5001
```

## 📊 Data Structure

The expected MLflow artifacts structure:

```
mlruns/
└── {experiment_id}/
    └── {run_id}/
        └── artifacts/
            ├── code_carbon/
            │   ├── emissions_data.csv
            │   └── emissions_train.csv
            ├── data_techniques/
            │   └── techniques.json
            ├── dataset/
            │   ├── X_test.csv
            │   └── X_train.csv
            ├── model/
            │   ├── MLmodel
            │   └── model files...
            ├── timestamps/
            │   └── all_timestamps.txt
            └── whylogs/
                └── profiles_augmented.csv ...
```

## 🔧 MLflow Information

- **Version**: MLflow 2.21.2
- **Database**: PostgreSQL backend for tracking
- **Artifacts**: File-based storage in Docker volumes

MLflow ORM models reference:
- [Tracking models](https://github.com/mlflow/mlflow/blob/master/mlflow/store/tracking/dbmodels/models.py)
- [Registry models](https://github.com/mlflow/mlflow/blob/master/mlflow/store/model_registry/dbmodels/models.py)

## 🐳 Docker Operations

```bash
# Start all services
docker compose up -d

# Start specific services
docker compose up -d postgres mlflow data_transfer

# View logs
docker compose logs -f data_transfer

# Rebuild specific service
docker compose build data_transfer

# Stop all services
docker compose down

# Clean rebuild
docker compose down && docker compose up --build -d
```

## 📚 Documentation

- **[DOCKER_SETUP.md](docs/DOCKER_SETUP.md)** - Comprehensive Docker setup guide
- **[DEV_CONTAINER_GUIDE.md](docs/DEV_CONTAINER_GUIDE.md)** - Development container usage
- **[ARTIFACTS_FIX_SUMMARY.md](docs/ARTIFACTS_FIX_SUMMARY.md)** - MLflow artifacts and sync guide

## 🧠 Ontop — SPARQL Endpoint & Virtual Knowledge Graph

[Ontop](https://ontop-vkg.org/) exposes the PostgreSQL data as a **virtual knowledge graph** using the [AIDOC-AP ontology](https://w3id.org/aidoc-ap#) (AI Documentation Application Profile). Instead of duplicating data into a triple store, Ontop translates SPARQL queries into SQL on-the-fly against the live database.

### How it works

```
┌──────────────┐   R2RML mapping    ┌──────────────┐   SPARQL    ┌──────────┐
│  PostgreSQL  │  ──────────────►   │    Ontop     │  ◄────────  │  Client  │
│  certain_db  │                    │  (port 8080) │  ────────►  │          │
│              │   ontology.ttl     │              │   RDF/JSON  │          │
│  experiments │  (AIDOC-AP vocab)  │  Virtual KG  │             │          │
│  runs, data  │                    │              │             │          │
│  models ...  │   input.properties │              │             │          │
│              │  (JDBC connection) │              │             │          │
└──────────────┘                    └──────────────┘             └──────────┘
```

### Configuration files

All Ontop configuration lives in `ontop/ontop/input/`:

| File | Purpose |
|---|---|
| `input.properties` | JDBC connection to PostgreSQL + references to ontology/mapping files |
| `ontology.ttl` | The AIDOC-AP ontology — defines OWL classes (`ModelEngineering`, `AIActivity`, `Dataset`, etc.) and properties |
| `mapping.ttl` | R2RML mapping — maps each PostgreSQL table/column to ontology classes and properties |

### Verify Ontop is running

```bash
# Check container status
docker compose ps | grep ontop

# Check health endpoint
curl -s http://localhost:8080/actuator/health
# Expected: {"status":"UP"}

# Check logs
docker compose logs ontop --tail 20

# Verify JDBC driver is loaded
docker exec ontop_service ls -la /opt/ontop/jdbc/
# Expected: postgresql-42.7.7.jar (~1 MB)
```

### Access the SPARQL UI

Open **http://localhost:8080** in your browser — Ontop provides a built-in query editor where you can write and execute SPARQL queries interactively.

### Example SPARQL queries

**List all experiments:**
```bash
curl -s -X POST "http://localhost:8080/sparql" \
  -H "Accept: application/json" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  --data-urlencode "query=
    PREFIX aidoc: <https://w3id.org/aidoc-ap#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX dcterms: <http://purl.org/dc/terms/>

    SELECT ?experiment ?id ?name ?lifecycleStage WHERE {
      ?experiment a aidoc:ModelEngineering ;
                  dcterms:identifier ?id ;
                  rdfs:label ?name ;
                  aidoc:hasLifecycleStage ?lifecycleStage .
    } LIMIT 10
  " | python3 -m json.tool
```

**List runs with their status and linked experiment:**
```bash
curl -s -X POST "http://localhost:8080/sparql" \
  -H "Accept: application/json" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  --data-urlencode "query=
    PREFIX aidoc: <https://w3id.org/aidoc-ap#>
    PREFIX dcterms: <http://purl.org/dc/terms/>
    PREFIX prov: <http://www.w3.org/ns/prov#>
    PREFIX schema: <https://schema.org/>

    SELECT ?run ?runId ?status ?experiment WHERE {
      ?run a aidoc:AIActivity ;
           dcterms:identifier ?runId ;
           schema:status ?status ;
           prov:wasInfluencedBy ?experiment .
    } LIMIT 5
  " | python3 -m json.tool
```

### Ontop key mapping concepts

The R2RML mapping in `mapping.ttl` translates PostgreSQL tables to AIDOC-AP ontology classes:

| PostgreSQL Table | AIDOC-AP Class | Description |
|---|---|---|
| `experiments` | `aidoc:ModelEngineering` / `mls:Experiment` | ML experiments |
| `runs` | `aidoc:AIActivity` / `prov:Activity` | Individual training/evaluation runs |
| `data` | `aidoc:Dataset` / `mls:Dataset` | Training/test datasets |
| `runs_code` | `aidoc:SoftwareImplementation` | Code snapshots |
| `runs_logs` | `aidoc:Log` | Run log entries |
| `data_hyperparameters` | `mls:HyperParameter` | Data processing hyperparameters |
| `data_metrics` | `aidoc:PerformanceMetric` | Data quality metrics |

### Troubleshooting Ontop

```bash
# If Ontop won't start — check for port conflicts
lsof -i :8080

# If queries return empty — verify the mapping loads correctly
docker compose logs ontop | grep -i "error\|exception\|mapping"

# Restart Ontop after changing mapping/ontology files
docker compose restart ontop
```

---


## ....

# 🎯 Complete Docker Tutorial & Operations Guide

This comprehensive tutorial covers everything you need to know about building, running, and managing the Certain Library Docker environment.

## 🏗️ Building and Starting the Docker Environment

### Step 1: Build and Start All Services

```bash
# Start all services (builds automatically if needed)
docker compose up -d

# Or build explicitly first, then start
docker compose build
docker compose up -d

# Start with logs visible (useful for debugging)
docker compose up

# Start only specific services
docker compose up -d postgres mlflow data_transfer
```

### Step 2: Verify All Services Are Running

```bash
# Check status of all services
docker compose ps

# Expected output:
# NAME                        COMMAND                  SERVICE        STATUS         PORTS
# certain_databases           "docker-entrypoint.s…"  postgres       Up (healthy)   0.0.0.0:5432->5432/tcp
# certain_data_transfer_api   "/app/start.sh"          data_transfer  Up (healthy)   0.0.0.0:8001->8001/tcp
# certain_library_tracker     "tail -f /dev/null"      library_tracker Up            0.0.0.0:8002->8002/tcp
# certain_mlflow              "sh -c 'cd /mlflow &&…" mlflow         Up (healthy)   0.0.0.0:5001->5001/tcp

# Check logs for specific service
docker compose logs data_transfer
docker compose logs mlflow
docker compose logs postgres
```

## 🧪 Running the Complete Workflow Test

### Execute the Main Test

```bash
# Run the complete workflow test
docker exec certain_library_tracker python /workspace/test_docker/test_complete_workflow.py
```

### Expected Test Output

The test should show:
- MLflow tracking setup
- Library integration verification
- Data logging operations
- Resource monitoring
- Successful completion message

## 🔗 Using the Data Transfer API for Syncing

### API Health Check

```bash
# Check if API is running
curl http://localhost:8001/health
# Expected: {"status":"healthy"}

# Check API root
curl http://localhost:8001/
# Expected: {"status":"up"}
```

### View API Documentation

```bash
# Open API documentation in browser
open http://localhost:8001/docs

# Or get OpenAPI spec
curl http://localhost:8001/openapi.json | jq '.'
```

### Sync All Data

```bash
# Sync all MLflow data to the target database
curl -X POST "http://localhost:8001/sync/all"
# Expected: {"status":"all data synced successfully"}
```

### Individual Sync Operations

```bash
# Sync specific data types
curl -X POST "http://localhost:8001/sync/data_metrics"
curl -X POST "http://localhost:8001/sync/data_resources"

# Get all available data
curl -X GET "http://localhost:8001/all/data"
```

## 🐳 Docker Management & Monitoring

### Container Status and Logs

```bash
# Check which containers are running
docker ps

# Check resource usage
docker stats

# View logs with follow
docker compose logs -f data_transfer
docker compose logs -f postgres

# View logs for all services
docker compose logs
```

### Container Shell Access

```bash
# Access the library tracker container (most useful for development)
docker exec -it certain_library_tracker bash

# Access the data transfer API container
docker exec -it certain_data_transfer_api bash

# Access the database container
docker exec -it certain_databases bash

# Access MLflow container
docker exec -it certain_mlflow bash

```

### Service Management

```bash
# Stop all services
docker compose down

# Stop specific service
docker compose stop data_transfer

# Restart specific service
docker compose restart data_transfer

# Restart all services
docker compose restart
```

## 🔧 Troubleshooting & Maintenance

### Volume Management

```bash
# List Docker volumes
docker volume ls

# Inspect specific volume
docker volume inspect certain_library_postgres_data

# **DANGER**: Remove all volumes (deletes all data!)
docker compose down -v

# **SAFER**: Remove only specific volumes
docker volume rm certain_library_postgres_data

# To remove any left volume
docker volume prune -f   
```

### Clean Rebuild

```bash
# Complete clean rebuild (removes everything)
docker compose down -v
docker system prune -f
docker compose build --no-cache
docker compose up -d

# Rebuild specific service
docker compose build --no-cache data_transfer
docker compose up -d data_transfer
```

## 🗄️ Database Inspection & Management

### Quick Database Overview

```bash
# List all databases
docker exec certain_databases psql -U postgres -c "\l"

# List all tables in certain_db
docker exec certain_databases psql -U postgres -d certain_db -c "\dt"

# List all tables in mlflow_db  
docker exec certain_databases psql -U postgres -d mlflow_db -c "\dt"
```

### Inspect Synced Data Content

```bash
# Check row counts in main tables
docker exec certain_databases psql -U postgres -d certain_db -c "
SELECT COUNT(*) as total_rows, 'experiments' as table_name FROM experiments 
UNION ALL SELECT COUNT(*), 'runs' FROM runs 
UNION ALL SELECT COUNT(*), 'data_metrics' FROM data_metrics 
UNION ALL SELECT COUNT(*), 'model_metrics' FROM model_metrics 
UNION ALL SELECT COUNT(*), 'id_mapping' FROM id_mapping 
ORDER BY table_name;"

# View recent experiments
docker exec certain_databases psql -U postgres -d certain_db -c "
SELECT experiment_id, experiment_name, lifecycle_stage, creation_time 
FROM experiments 
ORDER BY creation_time DESC 
LIMIT 5;"

# View recent runs
docker exec certain_databases psql -U postgres -d certain_db -c "
SELECT run_id, experiment_id, status, start_time, end_time 
FROM runs 
ORDER BY start_time DESC 
LIMIT 5;"

# Check ID mapping
docker exec certain_databases psql -U postgres -d certain_db -c "
SELECT run_id, data_id, model_id 
FROM id_mapping 
LIMIT 5;"
```

### Data Metrics Inspection

```bash
# View data metrics (WhyLogs profiles)
docker exec certain_databases psql -U postgres -d certain_db -c "
SELECT data_id, key, value, data_stage 
FROM data_metrics 
WHERE key LIKE '%cardinality%' 
LIMIT 10;"

# Count metrics by type
docker exec certain_databases psql -U postgres -d certain_db -c "
SELECT 
    CASE 
        WHEN key LIKE '[drift_metrics]%' THEN 'drift_metrics'
        WHEN key LIKE '%cardinality%' THEN 'cardinality'
        WHEN key LIKE '%distribution%' THEN 'distribution'
        ELSE 'other'
    END as metric_type,
    COUNT(*) as count
FROM data_metrics 
GROUP BY metric_type 
ORDER BY count DESC;"
```

### Resource Monitoring Data

```bash
# View resource consumption data
docker exec certain_databases psql -U postgres -d certain_db -c "
SELECT * FROM resources 
ORDER BY timestamp DESC 
LIMIT 10;"

# View emissions data
docker exec certain_databases psql -U postgres -d certain_db -c "
SELECT * FROM data_resources 
WHERE key LIKE '%emissions%' 
ORDER BY timestamp DESC 
LIMIT 10;"
```

### Interactive Database Session

```bash
# Start interactive PostgreSQL session
docker exec -it certain_databases psql -U postgres -d certain_db

# Once inside psql:
# \dt                    -- List tables
# \d table_name          -- Describe table structure
# \q                     -- Quit
# SELECT * FROM experiments LIMIT 5;
```


## 📋 Quick Reference Commands

```bash
# Essential commands for daily use:

# Build
docker compose up --build -d

# Start environment
docker compose up -d

# Check status
docker compose ps

# Run tests
docker exec certain_library_tracker python test_docker/test_complete_workflow.py

# Sync data
curl -X POST "http://localhost:8001/sync/all"

# Check database
docker exec certain_databases psql -U postgres -d certain_db -c "\dt"

# View logs
docker compose logs -f data_transfer

# Shell access
docker exec -it certain_library_tracker bash

# Stop environment
docker compose down
```