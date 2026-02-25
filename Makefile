.PHONY: install synth deploy destroy test lint local

# === Setup ===
install:
	pip install -e ".[dev,cdk]"

# === CDK ===
synth:
	cd cdk && cdk synth

deploy:
	cd cdk && cdk deploy --all --require-approval broadening

destroy:
	cd cdk && cdk destroy --all

diff:
	cd cdk && cdk diff

# === Testing ===
test:
	python -m pytest tests/ -v

test-unit:
	python -m pytest tests/unit/ -v

# === Code quality ===
lint:
	ruff check src/ tests/ cdk/
	ruff format --check src/ tests/ cdk/

format:
	ruff format src/ tests/ cdk/
	ruff check --fix src/ tests/ cdk/

# === Docker (local testing) ===
docker-build:
	docker build -t claudestreet:latest .

docker-run-sentinel:
	docker run --rm -e AWS_DEFAULT_REGION=us-east-1 \
		claudestreet:latest claudestreet.handlers.sentinel.handler
