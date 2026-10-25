"""Cache the pinned official pretrained model; never execute remote model code."""

import os
from pathlib import Path

from signaldesk.config import PROJECT_ROOT
from signaldesk_data.source import write_json


def main() -> None:
    import json

    config = json.loads((PROJECT_ROOT / "config/sentiment-transformer.json").read_text())
    os.environ["HF_HOME"] = str(PROJECT_ROOT / ".runtime/huggingface")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        config["model"],
        revision=config["revision"],
        allow_patterns=[
            "config.json",
            "model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.txt",
            "LICENSE",
            "README.md",
        ],
    )
    write_json(
        PROJECT_ROOT / ".runtime/pretrained.json",
        {
            "model": config["model"],
            "revision": config["revision"],
            "path": str(Path(path).resolve()),
        },
    )
    print(f"Cached pinned revision {config['revision']} at {path}")


if __name__ == "__main__":
    main()
