# Use an official Python runtime as a parent image
FROM python:3.12-slim-bookworm

# Install curl and certificates
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates

# Download the uv installer
ADD https://astral.sh/uv/install.sh /uv-installer.sh

# Install uv and remove the installer
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Add uv to PATH
ENV PATH="/root/.local/bin/:$PATH"

# Set the working directory
WORKDIR /code

# Copy dependency files first
COPY pyproject.toml uv.lock /code/

# Install the exact locked dependencies
RUN uv sync --frozen

# Copy the FastAPI application
COPY ./app /code/app

# Start FastAPI on port 80 inside the container
CMD ["uv", "run", "fastapi", "run", "app/main.py", "--port", "80"]