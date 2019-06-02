.PHONY: all clean install


all: install

.venv:
	@python3 -m venv .venv
	@.venv/bin/pip install --upgrade pip setuptools wheel
	@.venv/bin/pip install pip-tools
	@echo "Type 'source .venv/bin/activate' to activate a python virtual environment"

install: .venv
	@.venv/bin/pip install -r requirements.txt

clean:
	@rm -rf .venv