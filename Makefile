.PHONY: setup clean

clean:
	@if [ -d .venv ]; then rm -rf .venv; fi
	@if [ -f uv.lock ]; then rm uv.lock; fi
	uv cache clean
	uv cache prune
	echo "Cleaned up successfully, Deactivate the venv using 'deactivate'"

setup:
	uv venv .venv
	uv sync
