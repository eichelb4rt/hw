#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$SCRIPT_DIR" || exit 1
file="report"
bibtex "$file"
pdflatex -shell-escape "$file" && pdflatex -shell-escape "$file"