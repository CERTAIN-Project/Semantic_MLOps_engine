# Use Python 3.9.16 to match MLOps environment exactly
FROM python:3.9.16-slim-bullseye

# Install PostgreSQL 13 and required packages
RUN apt-get update && apt-get install -y \
    wget \
    ca-certificates \
    gnupg2 \
    build-essential \
    curl \
    sudo \
    && wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | apt-key add - \
    && echo "deb http://apt.postgresql.org/pub/repos/apt/ bullseye-pgdg main" > /etc/apt/sources.list.d/pgdg.list \
    && apt-get update && apt-get install -y \
    postgresql-13 \
    postgresql-client-13 \
    postgresql-contrib-13 \
    && rm -rf /var/lib/apt/lists/*

# Create PostgreSQL data directory and set permissions
RUN mkdir -p /var/lib/postgresql/data \
    && chown -R postgres:postgres /var/lib/postgresql \
    && chmod 700 /var/lib/postgresql/data

# Switch to postgres user for initialization
USER postgres

# Initialize PostgreSQL database
RUN /usr/lib/postgresql/13/bin/initdb -D /var/lib/postgresql/data

# Configure PostgreSQL
RUN echo "host all all 0.0.0.0/0 md5" >> /var/lib/postgresql/data/pg_hba.conf \
    && echo "listen_addresses='*'" >> /var/lib/postgresql/data/postgresql.conf

# Switch back to root user
USER root

# Install Python dependencies
COPY requirements.txt /tmp/requirements.txt
# Use virtual environment to avoid system package conflicts
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip
RUN pip install -r /tmp/requirements.txt

# Set environment variables for PostgreSQL
ENV POSTGRES_USER=postgres
ENV POSTGRES_PASSWORD=postgres
ENV POSTGRES_DB=postgres

# Set working directory
WORKDIR /app

COPY test_docker/ /app/test_docker/
COPY certain_library/ /app/certain_library/

# Copy the entire data_api directory structure properly
COPY data_api/ /app/data_api/
# Copy main.py to the data_api directory
COPY data_api/main.py /app/data_api/main.py
# Ensure alembic.ini is in the correct locations
COPY data_api/alembic.ini /app/data_api/alembic.ini

# Create startup script for PostgreSQL
COPY <<EOF /app/start.sh
#!/bin/bash
set -e

# Start PostgreSQL
sudo -u postgres /usr/lib/postgresql/13/bin/pg_ctl -D /var/lib/postgresql/data -l /var/lib/postgresql/data/postgresql.log start

# Wait for PostgreSQL to start
sleep 5

# Create database and user if needed
sudo -u postgres createdb postgres || true
sudo -u postgres psql -c "ALTER USER postgres PASSWORD 'postgres';" || true

# Keep container running
tail -f /var/lib/postgresql/data/postgresql.log
EOF

RUN chmod +x /app/start.sh

# Expose the default PostgreSQL port
EXPOSE 5432

# Default command
CMD ["/app/start.sh"]