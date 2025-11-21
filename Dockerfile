# Use uv Python 3.12 image for modern package management
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Set environment variables
ENV CONTAINER_HOME=/app

# Set work directory
WORKDIR $CONTAINER_HOME

# Copy dependency files for better caching
COPY pyproject.toml uv.lock ./

# Install dependencies using uv with cache mount
RUN --mount=type=cache,target=/tmp/uv-cache \
    uv sync --frozen --no-dev

# Copy application code
COPY app ./app

# Expose port
EXPOSE 8000

# Run the application
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]