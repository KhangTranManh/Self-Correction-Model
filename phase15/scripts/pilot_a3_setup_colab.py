from pathlib import Path

root = Path('/content/p15a3')
(root / 'vendor/src/data/verifiers').mkdir(parents=True, exist_ok=True)
(root / 'vendor/src/core').mkdir(parents=True, exist_ok=True)
print(f'Prepared {root}')
