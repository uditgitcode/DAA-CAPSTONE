#!/usr/bin/env bash
# Reproduce every artefact in the submission.
set -e
cd "$(dirname "$0")"
echo "== regenerate medium/large datasets (fixed seeds)"; python3 data/generate_data.py
echo "== unit tests";            python3 -m unittest discover -s tests -t . -v
echo "== demo runs";             python3 main.py data/small_city.json --adjacency --dp-table
python3 main.py data/medium_city.json --all-trips | tail -9
echo "== benchmarks";            python3 benchmark.py | tee results/benchmark_log.txt
echo "== design diagram";        python3 tools/make_design_diagram.py
echo "== screenshots / maps";    python3 tools/make_screenshots.py
echo "done."
