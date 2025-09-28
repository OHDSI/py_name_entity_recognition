# Stage 1: Builder
# This stage installs dependencies and prepares the application
FROM python:3.12-slim as builder

# Install Poetry
RUN pip install poetry

# Set the working directory
WORKDIR /app

# Copy only the files needed for dependency installation
COPY pyproject.toml poetry.lock ./

# Install dependencies, excluding development ones
# --no-root is required because we are in a non-interactive environment
RUN poetry install --no-dev --no-root

# Stage 2: Runtime
# This stage creates the final, lean production image
FROM python:3.12-slim as runtime

# Set the working directory
WORKDIR /app

# Create a non-root user and group
RUN addgroup --system nonroot && adduser --system --ingroup nonroot nonroot

# Copy installed dependencies from the builder stage
COPY --from=builder /app/.venv /.venv

# Set the PATH to include the virtual environment's binaries
ENV PATH="/app/.venv/bin:$PATH"

# Copy the application source code
COPY ./py_name_entity_recognition ./py_name_entity_recognition

# Switch to the non-root user
USER nonroot

# Set the entrypoint for the application
# This is a placeholder; adjust if your application has a specific entrypoint
CMD ["python"]
