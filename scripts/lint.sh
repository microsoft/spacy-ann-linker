#!/usr/bin/env bash

set -e
set -x

mypy spacy_ann --disallow-untyped-defs
black spacy_ann tests --check
isort --multi-line=3 --trailing-comma --force-grid-wrap=0 --combine-as --line-width 88 --recursive --check-only --thirdparty spacy_ann spacy_ann tests
