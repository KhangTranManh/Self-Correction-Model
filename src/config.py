"""Load .env (secrets) + configs/phase1.yaml (hyperparams/paths) thanh 1 object duy nhat."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent


@dataclass
class DeepSeekConfig:
    api_key: str
    model: str
    base_url: str
    temperature: float
    max_tokens: int
    max_retries: int


@dataclass
class HuggingFaceConfig:
    token: str
    repo_id: str
    private: bool


@dataclass
class Config:
    raw: dict = field(repr=False)
    deepseek: DeepSeekConfig
    huggingface: HuggingFaceConfig | None = None

    @property
    def small_model(self) -> dict:
        return self.raw["small_model"]

    @property
    def generation(self) -> dict:
        return self.raw["generation"]

    @property
    def verifier(self) -> dict:
        return self.raw["verifier"]

    @property
    def training(self) -> dict:
        return self.raw["training"]

    def path(self, key: str) -> Path:
        p = Path(self.raw["paths"][key])
        return p if p.is_absolute() else ROOT_DIR / p


def load_config(yaml_path: str | Path = ROOT_DIR / "configs" / "phase1.yaml") -> Config:
    load_dotenv(ROOT_DIR / ".env")

    with open(yaml_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    model = os.environ.get("DEEPSEEK_MODEL", "")
    if not api_key or not model:
        raise RuntimeError(
            "DEEPSEEK_API_KEY hoac DEEPSEEK_MODEL chua duoc set trong .env"
        )

    ds_cfg = DeepSeekConfig(
        api_key=api_key,
        model=model,
        base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        temperature=raw["deepseek"]["temperature"],
        max_tokens=raw["deepseek"]["max_tokens"],
        max_retries=raw["deepseek"]["max_retries"],
    )

    # hf_cfg duoc nap bat cu khi nao co HF_TOKEN, khong phu thuoc training.push_to_hub
    # -- vi push_to_hub.py (script doc lap, push thu cong) cung can config nay.
    hf_cfg = None
    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token:
        hf_cfg = HuggingFaceConfig(
            token=hf_token,
            repo_id=raw["huggingface"]["repo_id"],
            private=raw["huggingface"]["private"],
        )
    elif raw.get("training", {}).get("push_to_hub", False):
        raise RuntimeError(
            "training.push_to_hub=true trong phase1.yaml nhung HF_TOKEN chua duoc set trong .env"
        )

    return Config(raw=raw, deepseek=ds_cfg, huggingface=hf_cfg)
