#!/usr/bin/env bash
# Benchmark AmpereOne-focused commits over the last 24h (plus one commit before).
# Results are written under bench_results/<UTC timestamp>/.

set -euo pipefail

CALLER_DIR="$(pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BRANCH="${1:-ampereoneopt}"
SINCE="${SINCE:-24 hours ago}"
SIZES="${SIZES:-512 2048 8192}"
LOOPS="${LOOPS:-5}"
trap 'cd "${CALLER_DIR}"' EXIT

# Always operate from repo root so relative paths work even when invoked in subdirs.
cd "${TOP_DIR}"

# Detect usable parallelism
detect_jobs() {
  if command -v nproc >/dev/null 2>&1; then
    nproc
  elif [[ "$(uname -s)" == "Darwin" ]]; then
    sysctl -n hw.ncpu
  else
    echo 32
  fi
}
JOBS="${JOBS:-$(detect_jobs)}"
THREADS="${THREADS:-$JOBS}"

RESULT_ROOT="${RESULT_ROOT:-bench_results}"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
RESULT_DIR="${TOP_DIR}/${RESULT_ROOT}/${TS}"
mkdir -p "${RESULT_DIR}"

echo "[bench] collecting commits from '${BRANCH}' since '${SINCE}'"
mapfile -t commits < <(cd "${TOP_DIR}" && git rev-list --reverse --since="${SINCE}" "${BRANCH}")

if [[ "${#commits[@]}" -eq 0 ]]; then
  echo "No commits found in the requested window." >&2
  exit 1
fi

# Include one commit before the window for comparison.
first="${commits[0]}"
if (cd "${TOP_DIR}" && git rev-parse --verify "${first}^" >/dev/null 2>&1); then
  parent_before="$(cd "${TOP_DIR}" && git rev-parse "${first}^")"
  commits=("${parent_before}" "${commits[@]}")
fi

{
  echo "Branch: ${BRANCH}"
  echo "Since : ${SINCE}"
  echo "Threads: ${THREADS}"
  echo "Sizes: ${SIZES}"
  echo "Commits:"
  for c in "${commits[@]}"; do
    (cd "${TOP_DIR}" && git show -s --format='%h %ad %s' --date=short "${c}")
  done
} | tee "${RESULT_DIR}/overview.txt"

for c in "${commits[@]}"; do
  short="${c:0:12}"
  wt="$(mktemp -d "${RESULT_DIR}/wt-${short}-XXXX")"
  echo "[bench] preparing worktree ${wt} for ${short}"
  (cd "${TOP_DIR}" && git worktree add --quiet "${wt}" "${c}")

  pushd "${wt}" >/dev/null

  echo "[bench] building ${short}"
  make -j"${JOBS}"
  make -C benchmark -j"${JOBS}" goto

  printf "Commit: %s\n" "${c}" > "${RESULT_DIR}/${short}.info"
  (cd "${ROOT_DIR}" && git show -s --format='Hash: %H%nAuthor: %an <%ae>%nDate: %ad%nSubject: %s' "${c}") >> "${RESULT_DIR}/${short}.info"
  {
    echo "THREADS=${THREADS}"
    echo "SIZES=${SIZES}"
    echo "LOOPS=${LOOPS}"
  } >> "${RESULT_DIR}/${short}.info"

  for n in ${SIZES}; do
    log="${RESULT_DIR}/${short}-sgemm-${n}.log"
    echo "[bench] ${short} SGEMM N=${n} -> ${log}"
    OPENBLAS_NUM_THREADS=${THREADS} GOTO_NUM_THREADS=${THREADS} \
    OPENBLAS_PARAM_M=${n} OPENBLAS_PARAM_N=${n} OPENBLAS_PARAM_K=${n} \
    OPENBLAS_LOOPS=${LOOPS} \
      ./benchmark/sgemm.goto "${n}" "${n}" "${n}" 2>&1 | tee "${log}"
  done

  popd >/dev/null
  (cd "${TOP_DIR}" && git worktree remove --force "${wt}")
done

echo "[bench] done. Results stored in ${RESULT_DIR}"
