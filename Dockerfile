# Use Python 3.12 as the container runtime
FROM python:3.12-slim-bookworm

# Install curl and certificate support
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Download and install uv
ADD https://astral.sh/uv/install.sh /uv-installer.sh

RUN sh /uv-installer.sh \
    && rm /uv-installer.sh

# Make uv available as a command
ENV PATH="/root/.local/bin:$PATH"

# Set the container working directory
WORKDIR /code

# Copy dependency definitions first
COPY pyproject.toml uv.lock /code/

# Install the exact locked dependencies
RUN uv sync --frozen

# Copy application code, CNN architecture, and weights
COPY ./app /code/app
COPY ./helper_lib /code/helper_lib
COPY ./models /code/models

# Start FastAPI on port 80 inside the container
CMD ["uv", "run", "fastapi", "run", "app/main.py", "--port", "80"]