#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

echo "[1/7] Repository upload safety"
python3 scripts/check_repo.py

echo "[2/7] Secret patterns"
python3 scripts/scan_secrets.py

echo "[3/7] Markdown links"
python3 scripts/check_markdown_links.py

echo "[4/7] Configuration syntax"
python3 scripts/validate_configs.py

echo "[5/7] Shell syntax"
for script in scripts/*.sh; do
  bash -n "$script"
done

echo "[6/7] Unit tests"
python3 -m unittest discover -s tests -v

echo "[7/7] Optional local dataset"
data_root="${ICSHM_DATA_ROOT:-data/Contest Dataset}"
if [[ -d "$data_root" ]]; then
  python3 scripts/check_data.py --root "$data_root"
else
  echo "Dataset not present; skipped. This is expected in a clean Git clone."
fi

echo "Repository health check passed."

