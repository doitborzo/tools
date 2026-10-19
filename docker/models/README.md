Put the model archives here before `docker compose build` - they are unpacked into the image's /models:

- `herd_models_*.tgz` (needed; `herd/export_pod.sh` on a pod): the herd model, the detectors, DINOv2
- `herd_reid_*.tgz` (optional; `reid_pod.sh`): the identity model `reid/id_model_all/`
- `herd_rumination_*.tgz` (optional): the rumination model

Folders already unpacked (`herd/`, `hf-cache/`, `reid/`, `rum/`) work too. After adding a model, build again.
