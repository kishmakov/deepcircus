#!/usr/bin/env bash
set -euo pipefail

# Builds the paper PDF with its bibliography: `docs/paper.tex` cites
# `docs/paper.bib` through plain BibTeX

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

PAPER="${1:-$ROOT/docs/paper.tex}"
PAPER_DIR="$(cd "$(dirname "$PAPER")" && pwd)"
PAPER_NAME="$(basename "$PAPER" .tex)"

cd "$PAPER_DIR"

run_pdflatex() {
    pdflatex -interaction=nonstopmode -halt-on-error -file-line-error "$PAPER_NAME.tex"
}

run_pdflatex
bibtex "$PAPER_NAME"
run_pdflatex
run_pdflatex

echo "built $PAPER_DIR/$PAPER_NAME.pdf"
