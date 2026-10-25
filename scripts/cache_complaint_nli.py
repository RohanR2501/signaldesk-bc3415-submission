"""Download one pinned, official safe checkpoint; no model execution or data access."""

import json
import os

from signaldesk.config import PROJECT_ROOT
from signaldesk_data.source import write_json


def main() -> None:
    config = json.loads((PROJECT_ROOT / "config/complaint-models.json").read_text())
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
            "vocab.json",
            "merges.txt",
            "special_tokens_map.json",
            "README.md",
            "LICENSE",
        ],
    )
    write_json(
        PROJECT_ROOT / ".runtime/complaint-pretrained.json",
        {
            "model": config["model"],
            "revision": config["revision"],
            "path": path,
        },
    )
    print(f"Cached complaint NLI revision {config['revision']}")


if __name__ == "__main__":
    main()
