import os
from typing import Optional
from huggingface_hub import snapshot_download

base_dir = os.path.dirname(os.path.dirname(__file__))


def download_from_hub(
    repo_id: Optional[str] = None, token: Optional[str] = os.getenv("HF_TOKEN")
) -> None:
    repo_dir = os.path.join(base_dir, "checkpoints", repo_id)
    os.makedirs(repo_dir, exist_ok=True)
    print(repo_dir)

    assert os.path.exists(repo_dir)
    snapshot_download(
        repo_id,
        local_dir=f"checkpoints/{repo_id}",
        local_dir_use_symlinks=False,
        resume_download=True,
        allow_patterns=[
            "*.bin*",
            "tokenizer*",
            "generation_config.json",
            "config.json",
        ],
        token=token,
    )


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(download_from_hub)
