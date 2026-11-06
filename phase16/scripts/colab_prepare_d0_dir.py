"""Create the disposable D0 workspace in a Colab CPU session."""

from pathlib import Path

Path("/content/phase16").mkdir(parents=True, exist_ok=True)
print("/content/phase16 ready")
