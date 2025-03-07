#!/bin/bash

# Runs (only) if the container is (re-)build
# ==========================================

if $IS_POETRY; then
    ## Configure poetry
    poetry config virtualenvs.in-project true
    poetry install --with dev, docs
    poetry run pre-commit install
fi

# Create a directory /.vscode in your workspace if not present
if [ ! -d ./.vscode  ] ; then
    mkdir ./.vscode
fi

# Initializes Auto-Docstring configuration into your Project-VSCode settings.
# But only, if the file "settings.json" doesn't exist.
if [ ! -f ./.vscode/settings.json ] ; then
    touch ./.vscode/settings.json
    echo "{" >> ./.vscode/settings.json
    echo '    "autoDocstring.docstringFormat": "numpy",' >> ./.vscode/settings.json
    echo '    "autoDocstring.generateDocstringOnEnter": true' >> ./.vscode/settings.json
    echo "}" >> ./.vscode/settings.json
fi

