.DEFAULT_GOAL := help

SHELL := /bin/bash

# version control
VCS_URL       := $(shell git config --get remote.origin.url)
VCS_REF       := $(shell git rev-parse --short HEAD)
NOW_TIMESTAMP := $(shell date -u +"%Y-%m-%dT%H:%M:%SZ")
REPO_BASE_DIR := $(shell git rev-parse --show-toplevel)


.PHONY: clean

_GIT_CLEAN_ARGS := -dx --force --exclude=.vscode --exclude=TODO.md --exclude=.venv --exclude=.python-version --exclude=*keep*

.check-clean:
	@git clean -n $(_GIT_CLEAN_ARGS)
	@echo -n "Are you sure? [y/N] " && read ans && [ $${ans:-N} = y ]
	@echo -n "$(shell whoami), are you REALLY sure? [y/N] " && read ans && [ $${ans:-N} = y ]

clean: .check-clean ## cleans all unversioned files in project and temp files create by this makefile
	# Cleaning unversioned
	@git clean $(_GIT_CLEAN_ARGS)



.PHONY: info
info: ## displays basic info
	# system
	@echo ' CURDIR           : $(CURDIR)'
	@echo ' NOW_TIMESTAMP    : $(NOW_TIMESTAMP)'
	@echo ' VCS_URL          : $(VCS_URL)'
	@echo ' VCS_REF          : $(VCS_REF)'
	# dev tools version
	@echo ' make   : $(shell make --version 2>&1 | head -n 1)'
	@echo ' python : $(shell python3 --version)'


.PHONY: help
# thanks to https://marmelab.com/blog/2016/02/29/auto-documented-makefile.html
help:
	@echo "usage: make [target] ..."
	@echo ""
	@echo "Targets for '$(notdir $(CURDIR))':"
	@echo ""
	@awk --posix 'BEGIN {FS = ":.*?## "} /^[[:alpha:][:space:]_-]+:.*?## / {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
