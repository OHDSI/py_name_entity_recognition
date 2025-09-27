# ---- Builder Stage ----
# This stage installs poetry, exports the dependencies to a requirements.txt file.
FROM python:3.12-slim as builder

# Install poetry
RUN pip install poetry

# Set the working directory
WORKDIR /app

# Copy the dependency files
COPY pyproject.toml poetry.lock ./

# Export the dependencies to a requirements.txt file
# --without-hashes is used for broader compatibility
RUN poetry export -f requirements.txt --output requirements.txt --without-hashes

# ---- Final Stage ----
# This stage builds the final, lean image.
FROM python:3.12-slim

# Create a non-root user and group for security
# and create a home directory for the user.
RUN addgroup --system app && adduser --system --ingroup app --home /home/app app

# Set the working directory
WORKDIR /home/app

# Copy the requirements.txt from the builder stage
COPY --from=builder /app/requirements.txt .

# Install the dependencies using pip.
# We are not using --user as we want the packages to be in the system's
# site-packages, which is the default behavior and is in the PATH.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the user's home directory
COPY . .

# Change the ownership of the application files to the app user
RUN chown -R app:app /home/app

# Set the user
USER app

# Set the default command (optional, can be overridden)
# CMD ["python", "-m", "py_name_entity_recognition"]