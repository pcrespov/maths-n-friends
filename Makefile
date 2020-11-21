
# defaults
.DEFAULT_GOAL := help

# Use bash not sh
SHELL := /bin/bash


.PHONY: devenv
.venv:
	@python3 -m venv $@
	@$@/bin/pip install --upgrade pip setuptools wheel
	@$@/bin/pip install pip-tools


devenv: .venv ## sets up development enviroment
	@$</bin/pip install -r requirements.txt
	@echo "Type 'source .venv/bin/activate' to activate a python virtual environment"


start: .venv ## starts jupyter
	@$</bin/jupyter notebook


.PHONY: clean clean-all
_GIT_CLEAN_ARGS = -dxf -e .vscode -e .venv

clean: ## cleans all unversioned files in project and temp files create by this makefile
	# Cleaning unversioned
	@git clean -n $(_GIT_CLEAN_ARGS)
	@echo -n "Are you sure? [y/N] " && read ans && [ $${ans:-N} = y ]
	@echo -n "$(shell whoami), are you REALLY sure? [y/N] " && read ans && [ $${ans:-N} = y ]
	@git clean $(_GIT_CLEAN_ARGS)

.PHONY: requirements.txt
requirements.txt: requirements.in
	pip-compile --build-isolation --upgrade $<

clean-all:
	-@rm -rf .venv


.PHONY: help
help: ## this help
	@echo "usage: make [target] ..."
	@echo ""
	@echo "Targets for '$(notdir $(CURDIR))':"
	@echo ""
	@awk --posix 'BEGIN {FS = ":.*?## "} /^[[:alpha:][:space:]_-]+:.*?## / {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""