# CI/CD Strategy

## 1. Overview

The goal of this CI/CD pipeline is to establish a robust, secure, and efficient automated workflow for the `py_name_entity_recognition` repository. This document outlines the architecture and rationale behind the implemented solution. The pipeline automates linting, testing, and containerization to ensure code quality, maintainability, and security.

The key objectives achieved are:
- **Clarity:** A single, unambiguous dependency management strategy.
- **Efficiency:** Fast feedback loops, optimal caching, and minimized Docker build contexts.
- **Security:** Adherence to the principle of least privilege, vulnerability scanning, and other hardening practices.
- **Comprehensiveness:** Full test coverage across multiple platforms and Python versions.

## 2. Technology Stack and Rationale

The following tools have been selected and configured to form the backbone of the CI/CD pipeline:

- **Dependency Management:** **Poetry** was chosen as the definitive dependency manager. The repository already contained a `pyproject.toml` configured for Poetry and a `poetry.lock` file, indicating it as the established standard. No other managers were in use, so this choice represents a confirmation of the existing strategy.
- **Linting & Formatting:** **Ruff** is used for both linting and formatting. It is an extremely fast, all-in-one tool that can replace multiple other tools like `black`, `isort`, and `flake8`. This simplifies the configuration and speeds up the linting process significantly.
- **Type Checking:** **Mypy** is used for static type checking, ensuring type safety and improving code quality.
- **Orchestration:** **pre-commit** is used to manage and run all static analysis hooks (`ruff`, `mypy`, etc.). This ensures that checks are consistent between local development environments and the CI pipeline.
- **CI/CD Platform:** **GitHub Actions** is used to automate the entire workflow. It is tightly integrated with the source code repository and provides a flexible and powerful platform for building, testing, and deploying applications.
- **Containerization:** **Docker** is used to containerize the application, providing a consistent and isolated environment for execution.
- **Vulnerability Scanning:** **Trivy** is used to scan the Docker image for known vulnerabilities, adding a critical security layer to the pipeline.

## 3. Proactive Improvements Made

Several key files and configurations were missing or incomplete. The following improvements were made to establish a best-in-class CI/CD pipeline:

- **`.pre-commit-config.yaml` Created:** A comprehensive configuration file was created to manage all static analysis tools. It includes hooks for `ruff` (linting and formatting), `mypy` (type checking), and general repository hygiene (e.g., `check-yaml`, `end-of-file-fixer`). This centralizes quality checks and makes them easy to run locally.
- **`Dockerfile` Created:** A multi-stage `Dockerfile` was created from scratch.
    - It uses a slim Python base image to minimize size.
    - A `builder` stage installs Poetry and exports dependencies to a `requirements.txt` file.
    - The final stage copies only the necessary application code and `requirements.txt`, installing dependencies with `pip`. This keeps the final image lean and free of development tools.
    - A non-root user (`app`) is created and used to run the application, adhering to security best practices.
- **`.dockerignore` Created:** A `.dockerignore` file was added to exclude unnecessary files (e.g., `.git`, `.venv`, `__pycache__`) from the Docker build context. This improves build speed and prevents sensitive information from being included in the image.
- **GitHub Actions Workflows Created:** Two new workflow files were created in `.github/workflows/`: `ci.yml` and `docker.yml`.

## 4. Workflow Architecture

The CI/CD process is split into two distinct, parallel workflows:

### `ci.yml` (Linting and Testing)

This workflow is designed for fast feedback. It consists of two jobs in a dependency chain:

1.  **`lint` Job:** This job runs first on a single OS (`ubuntu-latest`) and Python version. It uses `pre-commit/action` to execute all checks defined in the `.pre-commit-config.yaml`. If this job fails, the entire workflow stops, providing immediate feedback on code quality issues without wasting resources on running tests.
2.  **`test` Job:** This job only runs if the `lint` job succeeds (`needs: [lint]`). It runs a test matrix across multiple operating systems (`ubuntu-latest`, `macos-latest`, `windows-latest`) and Python versions (`3.10`, `3.11`, `3.12`) to ensure broad compatibility.

### `docker.yml` (Container Build and Scan)

This workflow runs in parallel to `ci.yml` and focuses on the containerization aspect:

1.  **Build:** It builds the Docker image using the `Dockerfile`. For pull requests, the image is only built for verification and not pushed. Build caching is enabled to speed up subsequent runs.
2.  **Scan:** After the image is built, it is scanned for vulnerabilities using **Trivy**. The workflow is configured to fail if any `CRITICAL` or `HIGH` severity vulnerabilities are found.

## 5. Testing Strategy

- **Framework:** **Pytest** is used as the testing framework.
- **Matrix Testing:** The `test` job in `ci.yml` runs tests across a matrix of 3 operating systems and 3 Python versions, for a total of 9 different environments. This ensures that the code works as expected across all supported platforms. The matrix is set to `fail-fast: false` so that a failure on one platform does not cancel the others, providing a complete picture of compatibility.
- **Code Coverage:** Test coverage is collected during the `pytest` run and uploaded to **Codecov**. Unique flags are attached to each coverage report (`${{ matrix.os }}-py${{ matrix.python-version }}`) to allow Codecov to distinguish between the different environments in the test matrix.

## 6. Dependency Management and Caching

- **Installation:** In the CI environment, **Poetry** is installed using `pipx`. This is the recommended approach for installing Python applications in isolated environments, which prevents dependency conflicts and ensures Poetry is available on the `PATH`.
- **Caching:** The `actions/setup-python` action is configured with `cache: 'poetry'`. This enables native caching of the Poetry virtual environment, which significantly speeds up the `poetry install` step in subsequent workflow runs.

## 7. Security Hardening

The following security measures have been implemented across all workflows:

- **Principle of Least Privilege (PoLP):** All workflows are configured with `permissions: contents: read` at the top level. This ensures that the jobs have only the minimum permissions required to perform their tasks.
- **Action Pinning:** All third-party GitHub Actions are pinned to their full commit SHA (e.g., `actions/checkout@a5ac7e51b41094c92402da3b24376905380afc29`). This prevents malicious or unexpected updates to actions from being automatically used, mitigating supply chain attacks.
- **Non-Root Docker Container:** The `Dockerfile` creates and switches to a non-root user (`app`) before running the application. This is a critical security practice that limits the potential impact of a container compromise.
- **Vulnerability Scanning:** The `docker.yml` workflow integrates **Trivy** to scan the final Docker image for known vulnerabilities, failing the build if high or critical issues are found.
- **Docker Hub Authentication:** The `docker.yml` workflow includes a step to log in to Docker Hub using secrets. This is important to avoid anonymous pull restrictions and rate-limiting from Docker Hub, which can cause CI failures.

## 8. Docker Strategy

- **Multi-Stage Builds:** The `Dockerfile` uses a multi-stage build pattern. This separates the build-time dependencies (like Poetry) from the runtime dependencies, resulting in a final image that is significantly smaller and more secure.
- **Build Caching:** The `docker/build-push-action` is configured to use the GitHub Actions cache (`type=gha`). This caches Docker layers between workflow runs, dramatically speeding up image builds.
- **Verification Builds:** On pull requests, the `docker/build-push-action` is configured with `push: false` and `load: true`. This builds the image and loads it into the local Docker daemon on the runner, allowing it to be scanned by Trivy without needing to push it to a registry.

## 9. How to Run Locally

The CI checks can be replicated locally to catch issues before pushing code.

1.  **Install `pre-commit`:**
    ```bash
    pip install pre-commit
    ```

2.  **Install the Git hooks:**
    ```bash
    pre-commit install
    ```

3.  **Run all checks:**
    To run all configured linting and formatting checks on all files, use the following command:
    ```bash
    pre-commit run --all-files
    ```
    This will execute the same checks that the `lint` job runs in the CI pipeline.

4.  **Run tests:**
    To run the test suite locally, use Poetry:
    ```bash
    poetry install
    poetry run pytest
    ```