# 1. Base Image: Ubuntu 24.04
FROM ubuntu:24.04

# 2. Install uv (Copy binary method)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# 3. Global Settings
# Install venv to /opt so it doesn't conflict with code mounted at /app
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
# Compile bytecode for speed
ENV UV_COMPILE_BYTECODE=1
# Fix hardlink warnings
ENV UV_LINK_MODE=copy

WORKDIR /app

# 4. Copy ONLY dependency files
COPY pyproject.toml uv.lock ./

# 5. Install Dependencies (No Project Code)
# --frozen: Use lockfile exactly
# --no-install-project: Install pandas/torch, but NOT the flight-predictor package
# --no-editable: Ensure clean install
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-editable

# 6. Set Path
# This makes 'python' and 'uv' use the installed libraries automatically
ENV PATH="/opt/venv/bin:$PATH"

CMD ["bash"]

RUN apt-get update && apt-get install -y git
