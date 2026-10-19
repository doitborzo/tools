"""Build step: every model in /models loaded once (the internet still on), so the image runs offline.
Stops the build when there is no herd model: docker/models/ must hold herd_models_*.tgz."""
import glob
import json
import os
import sys

sys.path.insert(0, "/opt/herd/herd")

import common  # noqa: E402,F401  (puts cowbench on the path)


def main():
    import numpy as np
    import torch
    from PIL import Image
    models = sorted(glob.glob("/models/**/model.pt", recursive=True))
    if not models:
        sys.exit("!! no herd model (model.pt) in the image: put herd_models_*.tgz (export_pod.sh) into "
                 "docker/models/ and build again")
    from model import FrameEncoder
    encoders = set()
    for path in models:
        f = torch.load(path, map_location="cpu", weights_only=False).get("features", {})
        key = (f.get("encoder", "facebook/dinov2-small"), f.get("grid", 2))
        if key not in encoders:
            enc = FrameEncoder(*key).eval()
            with torch.no_grad():
                out = enc(torch.zeros(1, f.get("crop", 224), f.get("crop", 224), 3, dtype=torch.uint8))
            print(f"[warmup] frame encoder {key[0]} (grid {key[1]}): {tuple(out.shape)}", flush=True)
            encoders.add(key)
        print(f"[warmup] herd model {path}", flush=True)
    import detector as det_mod
    dets = sorted(os.path.dirname(p) for p in glob.glob("/models/**/best/det_train_meta.json", recursive=True))
    if not dets:
        print("!! no detector (a best/ folder with det_train_meta.json) in the image", flush=True)
    for best in dets:
        live = det_mod.Live(best)
        live(Image.fromarray(np.zeros((360, 640, 3), np.uint8)))
        print(f"[warmup] detector {best} ({live.name}, threshold {live.threshold})", flush=True)
    for path in sorted(glob.glob("/models/**/identity.pt", recursive=True)):
        torch.load(path, map_location="cpu", weights_only=False)
        print(f"[warmup] identity model {path}", flush=True)
    for path in sorted(glob.glob("/models/**/rumination.pt", recursive=True)):
        torch.load(path, map_location="cpu", weights_only=False)
        print(f"[warmup] rumination model {path}", flush=True)
    with open("/models/models.json", "w") as fh:
        json.dump({"herd": models, "detectors": dets}, fh, indent=1)


if __name__ == "__main__":
    main()
