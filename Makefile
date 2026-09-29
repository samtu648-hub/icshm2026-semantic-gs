.PHONY: verify data-check gpu-check ubuntu-check

verify:
	bash scripts/repo_health.sh

data-check:
	python3 scripts/check_data.py --root "$${ICSHM_DATA_ROOT:-data/Contest Dataset}"

gpu-check:
	bash scripts/check_gpu.sh

ubuntu-check:
	bash scripts/bootstrap_ubuntu.sh --check-only

