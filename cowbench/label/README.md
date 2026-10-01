# Labelling a new video

Fully manual labelling, step by step, in Ukrainian: [manual_labeling_uk.docx](manual_labeling_uk.docx)
(keyframes with `prelabel.py --detector none`, every box drawn by hand in CVAT).

    prelabel.py   video -> keyframes (1/s) -> cows found -> tracks -> [Muse labels] -> CVAT import file
    CVAT          a person corrects boxes, joins tracks, sets posture / activity
    cvat2ava.py   CVAT export -> annotations/ava_<split>_v2.1.csv, CBVD-5's format

The result is a dataset root like CBVD-5's (`annotations/`, `labelframes/labelframes/`),
so `cowbench.py plan --root newfarm`, `detector.py` and `lora/train_lora.py` read it as they are.

## 1. Pre-label (PC or pod)

    pip install -r label/requirements.txt
    python label/prelabel.py farm.mp4 --out newfarm

OWLv2 finds "a cow" without having seen this barn; on 51 barn photos of
another dataset it found 67% of the cows at threshold 0.3 with 65% of its
boxes right, so expect to add and delete boxes. Look at a few keyframes in
`newfarm/labelframes/labelframes/` and move `--det-threshold` (lower: more
cows found, more junk). `--base-url <pod>:8000 --model muse-glimmer` also asks
Muse for posture and activity; without it they are "?".

## 2. CVAT on the PC (Docker Desktop)

    git clone https://github.com/cvat-ai/cvat && cd cvat
    docker compose up -d
    docker exec -it cvat_server bash -ic 'python3 ~/manage.py createsuperuser'

http://localhost:8080 -> Tasks -> create: labels in the "Raw" tab = the contents
of `newfarm/cvat_labels.json`; file = the same farm.mp4; no frame step, no
start frame (the boxes are placed on the video's own frame numbers). Then
Actions -> Upload annotations -> "CVAT 1.1" -> `newfarm/cvat_prelabels.xml`.

Each cow is a track with a box on every keyframe (one a second); CVAT
interpolates between. Fix boxes, delete what is not a cow, draw the cows that
were missed, merge two tracks of one cow, and set posture / activity where
they change - an attribute holds until the next keyframe that changes it.
Rumination is jaw movement: decide it by playing the video, not on one frame.

Export: Actions -> Export task dataset -> "CVAT for video 1.1".

## 3. Convert

    python label/cvat2ava.py export.zip --root newfarm                 # test set: ava_val_v2.1.csv
    python label/cvat2ava.py export.zip --root newfarm --split train   # for training

A keyframe where any cow still has "?" is left out, and listed.

## Faster: train the detector on the first minutes

Correct the first 1-2 minutes, export, then:

    python label/cvat2ava.py export.zip --root newfarm --split train
    python detector.py train --root newfarm --out newfarm_det --model /workspace/lora-runs/detector/best
    python label/prelabel.py farm.mp4 --out newfarm --detector newfarm_det/best --keep export.zip --keep-until 120

and upload the new `cvat_prelabels.xml`: up to 120 s it is your corrected
export, after it boxes from a detector trained on this barn. Training wants a GPU.
