#!/usr/bin/env python3
"""Download BAAI/bge-reranker-base into huggingface_hub cache layout via ModelScope.

huggingface_hub's hf_hub_download fails on this repo because hf-mirror redirects
ONNX file metadata requests back to huggingface.co (blocked by GFW), and the
hf-mirror CDN is slow (~0.75MB/s). ModelScope (Aliyun) serves the same files at
~6MB/s from a China CDN.

This script fetches each file from ModelScope and writes it into the exact
huggingface_hub cache structure fastembed expects:
  models--BAAI--bge-reranker-base/
    refs/main                    (commit hash)
    blobs/<sha256>               (actual file content)
    snapshots/<commit>/<filename>(copy of blob; Windows can't symlink without admin)

Run: python download_bge_reranker.py
"""
import hashlib
import os
import shutil
import sys
import requests

REPO = "BAAI/bge-reranker-base"
COMMIT = "2cfc18c9415c912f9d8155881c133215df768a70"
CACHE = os.environ.get("HF_HOME", "user/huggingface")
BASE = os.path.join(CACHE, "models--" + REPO.replace("/", "--"))
# ModelScope (Aliyun) — fastest China CDN (~6MB/s vs hf-mirror's 0.75MB/s).
MS_BASE = "https://modelscope.cn/api/v1/models/BAAI/bge-reranker-base/repo"

# Files fastembed's TextCrossEncoder needs: the ONNX model + tokenizer/config.
FILES = [
    "onnx/model.onnx",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "sentencepiece.bpe.model",
]


def main() -> int:
    os.makedirs(os.path.join(BASE, "blobs"), exist_ok=True)
    os.makedirs(os.path.join(BASE, "snapshots", COMMIT, "onnx"), exist_ok=True)
    refs_path = os.path.join(BASE, "refs", "main")
    os.makedirs(os.path.dirname(refs_path), exist_ok=True)
    with open(refs_path, "w") as f:
        f.write(COMMIT)

    sess = requests.Session()
    for fname in FILES:
        snap_path = os.path.join(BASE, "snapshots", COMMIT, fname)
        if os.path.exists(snap_path) and os.path.getsize(snap_path) > 0:
            sz = os.path.getsize(snap_path)
            print(f"  SKIP (exists): {fname} ({sz} bytes)", flush=True)
            continue
        url = f"{MS_BASE}?Revision=master&FilePath={fname}"
        print(f"  downloading {fname} from ModelScope ...", flush=True)
        r = sess.get(url, stream=True, timeout=120, allow_redirects=True)
        r.raise_for_status()
        h = hashlib.sha256()
        tmp_blob = os.path.join(BASE, "blobs", ".tmp_download")
        downloaded = 0
        with open(tmp_blob, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):  # 1MB
                f.write(chunk)
                h.update(chunk)
                downloaded += len(chunk)
                if downloaded % (50 << 20) == 0:  # log every 50MB
                    print(f"    {fname}: {downloaded >> 20}MB", flush=True)
        blob_hash = h.hexdigest()
        blob_path = os.path.join(BASE, "blobs", blob_hash)
        os.replace(tmp_blob, blob_path)
        os.makedirs(os.path.dirname(snap_path), exist_ok=True)
        shutil.copy2(blob_path, snap_path)
        print(f"    OK: {os.path.getsize(snap_path)} bytes -> {snap_path}", flush=True)

    print("\nVerifying with huggingface_hub local_files_only...", flush=True)
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    from huggingface_hub import hf_hub_download
    for fname in FILES:
        p = hf_hub_download(repo_id=REPO, filename=fname, cache_dir=CACHE,
                            local_files_only=True)
        assert os.path.exists(p), f"missing: {p}"
    print("Verification OK. bge-reranker-base cache is complete.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
