# Repository Guidelines

## Project Structure & Module Organization
- `backend/src` hosts FastAPI modules (`config/`, `algorithms/`, `services/`, `api/`, `cli/`); keep domain logic here and place unit tests in `backend/tests`.
- `frontend/src` contains modular JavaScript components, services, and utilities; ship browser assets from `frontend/static` and document UX flows in `docs/`.
- Contract and regression suites live in `tests/contract`; they expect a running backend and real sample data under `data/`.
- Infrastructure assets such as `Dockerfile`, `docker-compose.yml`, and environment scripts (`setup.sh`, `setup.bat`) enable containerized runs and onboarding.

## Build, Test, and Development Commands
- `uvicorn backend.main:app --reload --app-dir backend` launches the API with auto-reload; set `LOG_LEVEL=DEBUG` when diagnosing pipelines.
- `npm --prefix frontend run dev` serves the static frontend on port 3000 for local UI work.
- `docker-compose up` builds the backend container (plus optional Redis cache) and binds `./data` and `./logs` for persistence.
- Use CLI helpers such as `python -m backend.src.cli.tsne_command --input data/sample.csv --output temp/tsne.json` for batch experiments.

## Coding Style & Naming Conventions
- Run `black backend/src backend/tests` (88-character lines) and `mypy backend/src` before committing; favor snake_case modules, PascalCase classes, and explicit type hints.
- JavaScript is formatted via `npm --prefix frontend run format` (Prettier) and linted with `npm --prefix frontend run lint`; colocate UI logic in `components/` and API calls in `services/`.
- Follow SDD principles: keep scientific algorithms pure, surface side effects via services, and name datasets/pipelines descriptively (`tsne_mof_batch2024`).

## Testing Guidelines
- Backend suites run with `cd backend && pytest`; use markers like `@pytest.mark.integration` or `@pytest.mark.slow` per `pyproject.toml`.
- Contract tests: `pytest tests/contract --maxfail=1` against a live server; refresh fixtures in `data/` to keep expectations reproducible.
- Frontend tests execute with `npm --prefix frontend test`, leveraging Jest and Testing Library; place specs in `frontend/src/tests`.
- Aim for ≥80% backend coverage (`coverage run -m pytest && coverage report`) and document any intentionally skipped paths.

## Commit & Pull Request Guidelines
- Craft concise, imperative commit subjects (`Add websocket health logging`) with optional bodies summarizing multi-module changes.
- Ensure `pytest`, `npm test`, and container builds succeed before review; call out skipped markers or failing suites in the PR description.
- Link relevant specs/issues, attach UI screenshots or data samples when applicable, and request reviewers versed in affected domains.
