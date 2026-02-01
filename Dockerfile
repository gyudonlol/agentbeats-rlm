FROM python:3.13-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy project files
COPY pyproject.toml uv.lock* ./
COPY src/ ./src/
COPY main.py ./

# Sync dependencies
RUN uv sync --frozen --no-dev

# Run the green agent
ENTRYPOINT ["uv", "run", "main.py", "green"]
