# Convenience targets (Linux/macOS/Colab). Windows users: run the python commands directly.
.PHONY: install test smoke quick full report verify
install:        ; pip install -r requirements.txt && pip install -e .
test:           ; pytest -q
smoke:          ; python -m spsbert synthetic --out data/synthetic_issues.csv && python -m spsbert run --config config/smoke_test.yaml
quick:          ; python -m spsbert run --config config/quick.yaml
full:           ; python -m spsbert run --config config/default.yaml
report:         ; python -m spsbert report --github-url $(URL)
verify:         ; python -m spsbert verify --project "$(PROJECT)"
