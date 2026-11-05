"""Download the official NIH Montgomery County Chest X-ray Dataset for External Validation.

Dataset: Montgomery County Department of Health and Human Services, Maryland, USA.
Total: 138 CXR radiographs (80 Normal, 58 Tuberculosis).
Format: MCUCXR_XXXX_0.png (Normal = 0), MCUCXR_XXXX_1.png (TB = 1).
"""

import concurrent.futures
import json
import os
import sys
import urllib.request
from typing import Dict, List, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEST_DIR = os.path.join(PROJECT_ROOT, "data", "external", "montgomery")
MANIFEST_PATH = os.path.join(PROJECT_ROOT, "data", "external", "montgomery_manifest.json")
BASE_URL = "https://huggingface.co/datasets/Famatsu123/montgomery-shenzhen-tuberculosis-cxr/resolve/main/MontgomerySet/CXR_png/"
TREE_URL = "https://huggingface.co/api/datasets/Famatsu123/montgomery-shenzhen-tuberculosis-cxr/tree/main/MontgomerySet/CXR_png"


def get_image_list() -> List[str]:
    """Retrieve list of 138 CXR filenames from HuggingFace dataset tree."""
    req = urllib.request.Request(TREE_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        items = json.loads(resp.read().decode())
    
    filenames = []
    for item in items:
        path = item.get("path", "")
        if path.endswith(".png") and not path.startswith("."):
            filenames.append(os.path.basename(path))
    return sorted(filenames)


def download_single_image(fname: str) -> Tuple[str, bool]:
    """Download one image if not already cached."""
    dest_path = os.path.join(DEST_DIR, fname)
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 10000:
        return fname, True

    url = BASE_URL + fname
    try:
        urllib.request.urlretrieve(url, dest_path)
        return fname, True
    except Exception as e:
        print(f"[!] Error downloading {fname}: {e}")
        return fname, False


def main():
    os.makedirs(DEST_DIR, exist_ok=True)
    print(f"[*] Fetching Montgomery County file list...")
    filenames = get_image_list()
    print(f"[*] Found {len(filenames)} images. Starting multi-threaded download to: {DEST_DIR}")

    success_count = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(download_single_image, f): f for f in filenames}
        for future in concurrent.futures.as_completed(futures):
            fname, ok = future.result()
            if ok:
                success_count += 1
                if success_count % 20 == 0 or success_count == len(filenames):
                    print(f"    - Downloaded {success_count}/{len(filenames)} images...")

    print(f"[✓] Download complete: {success_count}/{len(filenames)} images ready.")

    # Build manifest
    paths = []
    labels = []
    for fname in filenames:
        p = os.path.join(DEST_DIR, fname)
        if not os.path.exists(p) or os.path.getsize(p) < 10000:
            continue
        # Format: MCUCXR_XXXX_0.png -> 0 (Normal), MCUCXR_XXXX_1.png -> 1 (TB)
        label = int(fname.split("_")[-1].replace(".png", ""))
        paths.append(p)
        labels.append(label)

    manifest = {
        "dataset_name": "Montgomery County Chest X-ray Set (NIH / Montgomery County Dept of Health)",
        "source": "Montgomery County, Maryland, USA (Independent External Cohort)",
        "total_images": len(paths),
        "normal_images": labels.count(0),
        "tb_images": labels.count(1),
        "paths": paths,
        "labels": labels,
    }

    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[✓] Manifest saved to: {MANIFEST_PATH}")
    print(f"    - Normal: {labels.count(0)} | Tuberculosis: {labels.count(1)} | Total: {len(paths)}")


if __name__ == "__main__":
    main()
