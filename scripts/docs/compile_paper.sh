#!/usr/bin/env bash
set -euo pipefail

# Compiles the paper PDF with its bibliography: `docs/paper.tex` cites
# `docs/paper.bib` through plain BibTeX. Every intermediate (.aux/.log/.bbl/.blg/...)
# is written to a scratch directory, so the PDF is the only file the run leaves
# behind next to the source.

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

PAPER="${1:-$ROOT/docs/paper.tex}"
PAPER_DIR="$(cd "$(dirname "$PAPER")" && pwd)"
PAPER_NAME="$(basename "$PAPER" .tex)"

BUILD_DIR="$(mktemp -d)"
trap 'rm -rf "$BUILD_DIR"' EXIT

cd "$PAPER_DIR"

run_pdflatex() {
    pdflatex -interaction=nonstopmode -halt-on-error -file-line-error \
        -output-directory="$BUILD_DIR" "$PAPER_NAME.tex"
}

run_pdflatex
# BibTeX refuses to write through an absolute path (openout_any = p), so run it
# from the scratch directory and let it find the .bib next to the source.
(cd "$BUILD_DIR" && BIBINPUTS="$PAPER_DIR:" bibtex "$PAPER_NAME")
run_pdflatex
run_pdflatex

mv "$BUILD_DIR/$PAPER_NAME.pdf" "$PAPER_DIR/$PAPER_NAME.pdf"

# Drop intermediates left next to the source by earlier runs of other tooling.
for ext in aux log bbl blg out toc lof lot synctex.gz; do
    rm -f "$PAPER_DIR/$PAPER_NAME.$ext"
done

echo "compiled $PAPER_DIR/$PAPER_NAME.pdf"
