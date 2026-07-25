##---------------------------------
## Simple rules to automate chores.
##---------------------------------
##

# comments with a single leading # are internal (for developers working on this makefile)
# comments with two leading # are incorporated into the help message

# Ensure a venv is active, and abort otherwise.
ifeq (${VIRTUAL_ENV},)
	$(info No venv is active, but this Makefile can only safely be used inside one.)
	$(info You can usually create and activate a venv with `python3 -m venv venv && source venv/bin/activate`.)
	$(error Please activate a venv before proceeding)
endif

help:  ## Show this help.
	@sed -ne '/@sed/!s/[^:]*[#][#]//p' $(MAKEFILE_LIST)

# Ensure uv is installed for dependency management.
ifeq (,$(shell command -v uv))
	$(shell python3 -m pip install uv)
endif

# Shell check to find supported architecture
HAS_NVIDIA := $(shell nvidia-smi >/dev/null 2>&1 && echo "yes" || echo "no")
HAS_ROCM   := $(shell rocm-smi >/dev/null 2>&1 && echo "yes" || echo "no")

ifeq ($(HAS_NVIDIA),yes)
	MODE := cuda
else ifeq ($(HAS_ROCM),yes)
	MODE := rocm
else
	MODE := cpu
endif

##
##   Installing Dependencies
##

install-min:  ## Synchronise contents of venv and requirements/$(MODE).txt.
	uv pip install --extra $(MODE) -r pyproject.toml

install-dev:  ## Synchronise contents of venv and requirements/$(MODE)-dev.txt.
	uv pip install --extra $(MODE) -r --extra dev pyproject.toml

install: install-dev ## Synonymous with install-dev

##
##
## Chores
##

logs:
	mkdir -p logs

CHECK_DIRS = src/ tst/

.PHONY: format lint test type-check chores
format: logs  ## Do code formatting with isort and autopep8.
	python3 -m isort $(CHECK_DIRS) 2>&1 | tee logs/isort.log
	python3 -m autopep8 -v $(CHECK_DIRS) 2>&1 | tee logs/autopep8.log

lint: logs  ## Lint the project with ruff.
	python3 -m ruff check --fix $(CHECK_DIRS) 2>&1 | tee logs/ruff.log

test: logs  ## Run tests with coverage.
	python3 -m pytest --cov $(CHECK_DIRS) 2>&1 | tee logs/pytest.log

type-check: logs  ## Run static type checking with mypy.
	python3 -m mypy $(CHECK_DIRS) 2>&1 | tee logs/mypy.log

chores: format lint test type-check  ## Format, lint, test and type check the repository.

##
##
## Generating Artifacts
##

DOC_FORMATS = latexpdf epub dirhtml man
.PHONY: docs
docs: logs ## Generate PDF documentation.
	@rm -r docs/build || true
	@rm -r docs/source/_generated || true
	mkdir -p logs/docs
	for fmt in $(DOC_FORMATS); do \
	    $(MAKE) -C docs "$$fmt" 2>&1 | tee "logs/docs/$$fmt.log"; \
	done
	# move compiled pdfs to dedicated dir
	mkdir -p docs/build/pdf
	mv docs/build/latex/*.pdf docs/build/pdf
