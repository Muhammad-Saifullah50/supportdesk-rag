.PHONY: setup setup-backend setup-frontend run test test-backend test-frontend eval eval-retrieval compile

setup: setup-backend setup-frontend

setup-backend:
	cd backend && uv sync --frozen

setup-frontend:
	cd frontend && npm ci

run:
	cd backend && .venv/bin/uvicorn main:app --reload --host 127.0.0.1 --port 8000

test: test-backend test-frontend

test-backend:
	cd backend && .venv/bin/python -m unittest discover -s tests

test-frontend:
	cd frontend && npm run test

eval:
	python evals/run_live_eval.py --base-url $${EVAL_BASE_URL:-http://127.0.0.1:8000} --output evals/runs/$$(date -u +%Y-%m-%dT%H%M%SZ).json

eval-retrieval:
	python evals/run_retrieval_eval.py --output evals/runs/$$(date -u +%Y-%m-%dT%H%M%SZ)-retrieval.json

compile:
	cd backend && .venv/bin/python -m compileall -q *.py api tests
