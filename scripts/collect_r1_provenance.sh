#!/usr/bin/env bash
# Archive exact R1 upstream provenance and local patches outside Git.
# This never copies a third-party work into this repository.
set -euo pipefail

: "${R1_THIRD_PARTY_ROOT:?set R1_THIRD_PARTY_ROOT to the pinned external sources}"
: "${R1_OUTPUT_ROOT:?set R1_OUTPUT_ROOT to the Git-ignored R1 output root}"

destination="$R1_OUTPUT_ROOT/r1/provenance"
mkdir -p "$destination"

for candidate in standard-3dgs mip-splatting scaffold-gs 3dgs-mcmc; do
  source_root="$R1_THIRD_PARTY_ROOT/$candidate"
  if [ ! -d "$source_root/.git" ] && [ ! -f "$source_root/.git" ]; then
    echo "missing pinned Git source: $source_root" >&2
    exit 1
  fi

  evidence="$destination/$candidate.txt"
  patch="$destination/$candidate-local.patch"
  {
    printf 'candidate=%s\n' "$candidate"
    printf 'source_root=%s\n' "$source_root"
    printf 'origin_url=%s\n' "$(git -C "$source_root" remote get-url origin)"
    printf 'head_commit=%s\n' "$(git -C "$source_root" rev-parse HEAD)"
    printf 'head_subject=%s\n' "$(git -C "$source_root" log -1 --format=%s)"
    printf 'license_files=\n'
    find "$source_root" -maxdepth 2 -type f \( -iname 'license*' -o -iname 'copying*' \) -printf '%P\n' | sort
    printf 'root_status=\n'
    git -C "$source_root" status --short
    printf 'submodule_status=\n'
    git -C "$source_root" submodule status --recursive
  } > "$evidence"

  {
    printf '# candidate=%s root=%s\n' "$candidate" "$source_root"
    git -C "$source_root" diff --binary
    while read -r submodule_hash submodule_path remainder; do
      [ -n "$submodule_path" ] || continue
      printf '\n# submodule=%s\n' "$submodule_path"
      if [ -d "$source_root/$submodule_path" ] && { [ -d "$source_root/$submodule_path/.git" ] || [ -f "$source_root/$submodule_path/.git" ]; }; then
        git -C "$source_root/$submodule_path" diff --binary
      else
        printf '# unavailable locally; not required for the headless training path\n'
      fi
    done < <(git -C "$source_root" submodule status --recursive)
  } > "$patch"
  sha256sum "$evidence" "$patch" > "$destination/$candidate.sha256"
done

printf 'Wrote provenance evidence under %s\n' "$destination"
