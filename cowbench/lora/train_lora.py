#!/usr/bin/env python3
"""LoRA fine-tuning of Muse Glimmer on CBVD-5, in exactly the bench's format.

The model is trained on the same question the bench asks - the same prompt,
the same full frame with one cow outlined in lime, the same rendering - and
evaluated on the same 2532 val cows with the same example ids, so its
results.jsonl drops straight into `cowbench.py score / report / compare`
against the zero-shot runs in runs/.

Stages, each resumable and each its own process:

    data   count the examples, write a few rendered samples to look at
    train  load, sanity-check, train the adapter (resumes from checkpoints)
    eval   answer the val manifest with or without the adapter

Two units of question, --unit:

    cow    one cow outlined, one answer - the bench's own question
    frame  every cow of the keyframe outlined and numbered, one answer listing
           them all (../frame.py): the frame is encoded once instead of once
           per cow. Results are still written per cow.

Weights. The published checkpoint is FP8 block-quantized (compressed-tensors),
and an A100 has no FP8 matmul. The FP8 weights are expanded to BF16 on load
(weight * per-128x128-block scale) - no 60 GB BF16 download, and the adapter
is trained against exactly the weights vLLM serves, rounding included. On a
GPU too small for BF16 (< 70 GiB) the script switches to QLoRA over the
original BF16 repo instead.

Split. Train = ava_train minus the 5 clips it shares with val (341, 344, 345,
393, 394); eval = all of val. Not a single val clip is seen in training.
"""

from __future__ import annotations

import argparse
import collections
import datetime
import glob
import io
import json
import math
import os
import random
import re
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)
sys.path.insert(0, BENCH)

import cbvd  # noqa: E402
import client as client_mod  # noqa: E402
import frame as frame_mod  # noqa: E402
import render as render_mod  # noqa: E402

FP8_REPO = "RedHatAI/Muse-Glimmer-30B-FP8-block"
BF16_REPO = "meta-models/Muse-Glimmer-30B"
VAL_MANIFEST = os.path.join(BENCH, "runs", "2026-09-25_val-full_w1920_f1", "manifest.jsonl")

# Every Linear of the text decoder: attention q/k/v/o, the attention output
# gate, and the MLP. The vision tower and its projection stay frozen - the
# errors this is meant to fix are about reading the scene, not about seeing it,
# and the tower is the part the FP8 checkpoint never quantized anyway.
TARGET_RE = (r".*language_model\.layers\.\d+\."
             r"(self_attn\.(q_proj|k_proj|v_proj|o_proj|gate_proj)|mlp\.(gate_proj|up_proj|down_proj))")
TARGETS_PER_LAYER = 8


def answer_json(posture: str, activity: str) -> str:
    """The answer the model is taught to give, in the key order and spacing it
    already uses zero-shot: {"posture": "standing", "activity": "feeding"}.

    No confidence field: there is no ground truth for one, and a constant would
    teach the model to always say the same number.
    """
    return json.dumps({"posture": posture, "activity": activity})


# ------------------------------------------------------------------ examples

def load_train(root, include_val_clips=False):
    boxes = cbvd.load_boxes(os.path.join(root, "annotations", "ava_train_v2.1.csv"))
    usable, rejected = cbvd.partition(boxes)
    val = cbvd.load_boxes(os.path.join(root, "annotations", "ava_val_v2.1.csv"))
    val_clips = {b.video_id for b in val}
    kept = usable if include_val_clips else [b for b in usable if b.video_id not in val_clips]
    rows = [{"id": b.uid, "video_id": b.video_id, "timestamp": b.timestamp,
             "bbox": list(b.xyxy), "gt_posture": b.posture, "gt_activity": b.activity}
            for b in kept]
    rows.sort(key=lambda r: r["id"])
    info = {"train_boxes": len(boxes), "contradictory_dropped": len(rejected),
            "val_clip_boxes_dropped": len(usable) - len(kept),
            "val_clips_dropped": sorted({b.video_id for b in usable} & val_clips, key=int),
            "examples": len(rows), "clips": len({r["video_id"] for r in rows}),
            "keyframes": len({(r["video_id"], r["timestamp"]) for r in rows})}
    return rows, info


def subsample(rows, limit, seed):
    """Whole keyframes, not single cows: a keyframe is one image, and keeping
    all of its cows keeps the per-image cost of the sample honest."""
    if not limit or limit >= len(rows):
        return rows
    frames = collections.defaultdict(list)
    for r in rows:
        frames[(r["video_id"], r["timestamp"])].append(r)
    keys = sorted(frames)
    random.Random(seed).shuffle(keys)
    out = []
    for k in keys:
        if len(out) >= limit:
            break
        out.extend(frames[k])
    return sorted(out, key=lambda r: r["id"])


def split_dev(rows, frac, seed):
    """Hold out whole clips. Frames of one clip share the camera, the cows
    and often the labels, so a dev set cut by frame would be scored on
    scenes the adapter was trained on - the first LoRA got 8.7% error on
    its own training cows and 44% on val."""
    if frac <= 0:
        return rows, []
    clips = sorted({r["video_id"] for r in rows}, key=lambda v: (len(v), v))
    random.Random(seed).shuffle(clips)
    dev = set(clips[:max(1, round(len(clips) * frac))])
    return ([r for r in rows if r["video_id"] not in dev],
            [r for r in rows if r["video_id"] in dev])


def thin_frames(rows, per_clip):
    """Keep `per_clip` keyframes of each clip, spread over it. The six
    annotated seconds of a clip are near-duplicates - same cows, same
    camera - so all six mostly repeat one scene to memorise."""
    if not per_clip:
        return rows
    stamps = collections.defaultdict(set)
    for r in rows:
        stamps[r["video_id"]].add(r["timestamp"])
    keep = set()
    for vid, ts in stamps.items():
        ts = sorted(ts)
        if len(ts) <= per_clip:
            idx = range(len(ts))
        elif per_clip == 1:
            idx = [len(ts) // 2]
        else:
            idx = {round(i * (len(ts) - 1) / (per_clip - 1)) for i in range(per_clip)}
        keep.update((vid, ts[i]) for i in idx)
    return [r for r in rows if (r["video_id"], r["timestamp"]) in keep]


def training_rows(args):
    """(training rows, dev rows, info): dev clips first, then thinning and
    the training cap, so the dev set does not move with those knobs."""
    rows, info = load_train(args.root)
    rows, dev_all = split_dev(rows, args.holdout, args.seed)
    rows = thin_frames(rows, args.frames_per_clip)
    rows = subsample(rows, args.train_limit, args.seed)
    dev = subsample(dev_all, args.dev_size, args.seed)
    info.update({
        "dev_clips": sorted({r["video_id"] for r in dev_all}, key=lambda v: (len(v), v)),
        "dev_cows_held_out": len(dev_all), "dev_cows_scored": len(dev),
        "frames_per_clip": args.frames_per_clip,
        "train_clips": len({r["video_id"] for r in rows}),
        "train_keyframes": len({(r["video_id"], r["timestamp"]) for r in rows}),
    })
    return rows, dev, info


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def read_frame(root, box, tries=10):
    """The keyframe, decoded. Retried: on a network volume (RunPod's
    /workspace is MooseFS) a read can fail for a minute or two - ENXIO,
    EIO, or the file briefly "missing" - and one failed read in a DataLoader
    worker would otherwise end hours of training."""
    from PIL import Image
    delay = 2.0
    for attempt in range(1, tries + 1):
        try:
            with open(cbvd.frame_path(root, box), "rb") as fh:
                data = fh.read()
            img = Image.open(io.BytesIO(data))
            img.load()
            return img
        except OSError as e:   # FileNotFoundError and PIL's truncated-file errors included
            if attempt == tries:
                raise
            print(f"[io] {box.video_id} t={box.timestamp}: {e!r} - retry {attempt}/{tries - 1} "
                  f"in {delay:.0f}s", file=sys.stderr, flush=True)
            time.sleep(delay)
            delay = min(delay * 2, 60.0)


def make_image(root, row, width, quality):
    """What the bench sends: render, then a JPEG round trip at the bench's
    quality, because vLLM gets a JPEG data URL and the model should be trained
    on the same compression it is tested on."""
    from PIL import Image
    box = cbvd.Box(row["video_id"], row["timestamp"], *row["bbox"], "1", ())
    img = render_mod.render(read_frame(root, box), row["bbox"], mode="marked",
                            max_width=width)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def make_frame_image(root, fr, width, quality):
    """The frame unit's image: every cow of the keyframe outlined and numbered
    (frame.render), then the same JPEG round trip."""
    from PIL import Image
    box = cbvd.Box(fr["video_id"], fr["timestamp"], 0, 0, 0, 0, "1", ())
    img = frame_mod.render(read_frame(root, box), fr["cows"], width)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


# ------------------------------------------------------------------ prompting

def messages(text=None):
    # Image first, then text: the order the bench's client sends them in.
    text = client_mod.build_prompt(1, 0.0) if text is None else text
    return [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]


class Chat:
    """The prompt prefix and the answer wrapper, both cut from the model's own
    chat template rather than typed out here.

    Muse Glimmer's final answer is "<|start|>assistant to=user<|message|>...
    <|eot|>", its reasoning a separate "to=self" message before it. Training on
    the suffix below teaches it to go straight to the answer - which is also
    what vLLM will see it do once the adapter is served.
    """

    def __init__(self, processor):
        self.processor = processor
        self.tok = processor.tokenizer
        self.prefix = processor.apply_chat_template(messages(), tokenize=False,
                                                    add_generation_prompt=True)
        probe = answer_json("standing", "none")
        full = processor.apply_chat_template(
            messages() + [{"role": "assistant", "content": probe}], tokenize=False)
        if not full.startswith(self.prefix):
            raise SystemExit("chat template: the answered conversation does not start with "
                             "the generation prompt; cannot cut the answer off it.\n"
                             f"prefix tail: {self.prefix[-80:]!r}\nfull tail: {full[-160:]!r}")
        tail = full[len(self.prefix):]
        at = tail.index(probe)
        self.answer_open = tail[:at]                     # " to=user<|message|>"
        self.answer_close = tail[at + len(probe):]       # "<|eot|>"
        self._prefixes = {}

    def prefix_for(self, text):
        """The generation prompt for another question text (the frame unit's
        prompt lists the frame's boxes, so it differs per frame)."""
        if text not in self._prefixes:
            if len(self._prefixes) > 4096:
                self._prefixes.clear()
            self._prefixes[text] = self.processor.apply_chat_template(
                messages(text), tokenize=False, add_generation_prompt=True)
        return self._prefixes[text]

    def wrap(self, body):
        return self.answer_open + body + self.answer_close

    def flags(self, body, masked):
        """One flag per token of wrap(body): False where the token overlaps a
        masked (start, end) character span of body."""
        ids = self.tok(self.wrap(body), add_special_tokens=False, return_offsets_mapping=True)
        off = len(self.answer_open)
        spans = [(s + off, e + off) for s, e in masked]
        return [not any(a < e and b > s for s, e in spans) for a, b in ids["offset_mapping"]]

    def frame_flags(self, cows, fields):
        """loss_flags for the frame unit: the same three policies, per cow."""
        spans = frame_mod.activity_spans(cows)
        if fields == "both":
            masked = []
        elif fields == "posture":
            masked = [(s, e) for s, e, _ in spans]
        else:
            masked = [(s, e) for s, e, a in spans if a == "ruminating"]
        return self.flags(frame_mod.answer_json(cows), masked)

    def suffix(self, posture, activity):
        return self.answer_open + answer_json(posture, activity) + self.answer_close

    def suffix_ids(self, text):
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def loss_flags(self, posture, activity, fields):
        """One flag per suffix token: does the loss train it.

        both             every answer token
        posture          not the activity value: the adapter learns the format
                         and the posture, the activity answer stays the base
                         model's. Activity learned from ava_train did not carry
                         over to val, which looks like another recording
                         session (clips 341-394).
        skip-ruminating  not the activity value of cows labelled ruminating;
                         feeding, drinking and none are trained. Rumination
                         cannot be seen on a still, is labelled per clip rather
                         than per cow, and its 1106 standing-at-the-barrier
                         labels in train are what taught the first adapters to
                         call feeding cows ruminating on val.
        """
        text = self.suffix(posture, activity)
        ids = self.tok(text, add_special_tokens=False, return_offsets_mapping=True)
        if fields == "both" or (fields == "skip-ruminating" and activity != "ruminating"):
            return [True] * len(ids["input_ids"])
        body = answer_json(posture, activity)
        key = '"activity": "'
        start = len(self.answer_open) + body.index(key) + len(key)
        end = start + len(activity)
        return [not (a < end and b > start) for a, b in ids["offset_mapping"]]


_NESTED = {}


def encode(processor, texts, images):
    """Batch through the processor. Some processors want one list of images
    per text; try the flat form first and remember which one worked."""
    if "nested" not in _NESTED:
        try:
            out = processor(text=texts, images=images, return_tensors="pt", padding=True)
            _NESTED["nested"] = False
            return out
        except Exception:
            _NESTED["nested"] = True
    if _NESTED["nested"]:
        return processor(text=texts, images=[[i] for i in images], return_tensors="pt", padding=True)
    return processor(text=texts, images=images, return_tensors="pt", padding=True)


class Collator:
    """A batch of cows (unit "cow") or of keyframes (unit "frame")."""

    def __init__(self, processor, chat, root, width, quality, fields="both", unit="cow"):
        self.processor, self.chat = processor, chat
        self.root, self.width, self.quality = root, width, quality
        self.fields, self.unit = fields, unit

    def suffix(self, item):
        if self.unit == "frame":
            return self.chat.wrap(frame_mod.answer_json(item["cows"]))
        return self.chat.suffix(item["gt_posture"], item["gt_activity"])

    def __call__(self, rows):
        import torch
        texts, images, flags = [], [], []
        for r in rows:
            if self.unit == "frame":
                texts.append(self.chat.prefix_for(frame_mod.prompt(r["cows"])) + self.suffix(r))
                images.append(make_frame_image(self.root, r, self.width, self.quality))
                flags.append(self.chat.frame_flags(r["cows"], self.fields))
            else:
                texts.append(self.chat.prefix + self.suffix(r))
                images.append(make_image(self.root, r, self.width, self.quality))
                flags.append(self.chat.loss_flags(r["gt_posture"], r["gt_activity"], self.fields))
        enc = encode(self.processor, texts, images)
        # Left padding puts every answer at the very end of its row, so the
        # answer is always the last n tokens and the loss only needs the logits
        # of the last n+1 positions - not a 202k-wide vocabulary over the
        # whole ~1000-token sequence.
        mask = torch.zeros_like(enc["input_ids"], dtype=torch.bool)
        for i, f in enumerate(flags):
            mask[i, -len(f):] = torch.tensor(f)
        enc["answer_mask"] = mask
        enc["answer_len"] = torch.tensor([len(f) for f in flags])
        return dict(enc)


# ------------------------------------------------------------------ model

def gpu_gib():
    import torch
    return torch.cuda.get_device_properties(0).total_memory / 2**30


def dequantize_fp8_into(model, repo):
    """Overwrite every FP8-cast weight with weight * block scale.

    The model was loaded with the quantization config removed, so transformers
    cast the float8 weights to bf16 as they are: right values, missing their
    per-block scales. The scales sit next to them in the safetensors as
    `<name>.weight_scale`, one float per 128x128 block.
    """
    import torch
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    path = snapshot_download(repo, allow_patterns=["*.safetensors", "*.json"])
    params = dict(model.named_parameters())

    def find(name):
        if name in params:
            return params[name]
        tail = name.split(".", 1)[1]
        hits = [p for n, p in params.items() if n.endswith(tail)]
        return hits[0] if len(hits) == 1 else None

    done, unmatched = 0, []
    for shard in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
        with safe_open(shard, framework="pt", device="cpu") as fh:
            for skey in sorted(k for k in fh.keys() if k.endswith(".weight_scale")):
                wkey = skey[: -len("_scale")]
                target = find(wkey)
                if target is None:
                    unmatched.append(wkey)
                    continue
                w = fh.get_tensor(wkey).to(target.device)
                s = fh.get_tensor(skey).to(target.device, torch.float32)
                out_f, in_f = w.shape
                bo, bi = math.ceil(out_f / s.shape[0]), math.ceil(in_f / s.shape[1])
                scale = s.repeat_interleave(bo, 0)[:out_f].repeat_interleave(bi, 1)[:, :in_f]
                with torch.no_grad():
                    target.copy_((w.to(torch.float32) * scale).to(target.dtype))
                done += 1
    if unmatched:
        raise SystemExit(f"FP8 dequant: {len(unmatched)} checkpoint weights have no matching "
                         f"model parameter, e.g. {unmatched[:3]}")
    return done


def load_model(args):
    import torch
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

    precision = args.precision
    if precision == "auto":
        precision = "bf16" if gpu_gib() >= 70 else "qlora"
    source = args.base or (FP8_REPO if precision == "bf16" else BF16_REPO)
    print(f"[model] {source}  precision={precision}  gpu={torch.cuda.get_device_name(0)} "
          f"{gpu_gib():.0f} GiB", flush=True)

    processor = AutoProcessor.from_pretrained(FP8_REPO)
    config = AutoConfig.from_pretrained(source)
    fp8 = bool(getattr(config, "quantization_config", None))
    kwargs = {"dtype": torch.bfloat16, "device_map": {"": 0},
              "attn_implementation": args.attn}

    if precision == "qlora":
        if fp8:
            raise SystemExit(f"{source} is FP8; QLoRA needs the BF16 weights ({BF16_REPO}). "
                             "Pass --base with a BF16 repo, or run on an 80 GB GPU.")
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_skip_modules=["vision_tower", "vision_adapter", "vision_projection",
                                   "lm_head", "embed_tokens"])
    elif fp8:
        del config.quantization_config
    t0 = time.time()
    model = AutoModelForImageTextToText.from_pretrained(source, config=config, **kwargs)
    if fp8 and precision == "bf16":
        n = dequantize_fp8_into(model, source)
        print(f"[model] FP8 -> BF16: {n} weights rescaled", flush=True)
    fp8_types = {getattr(torch, "float8_e4m3fn", None), getattr(torch, "float8_e5m2", None)} - {None}
    left = [n for n, p in model.named_parameters() if p.dtype in fp8_types]
    if left:
        raise SystemExit(f"{len(left)} parameters are still float8, e.g. {left[:3]}")
    print(f"[model] loaded in {time.time() - t0:.0f}s, "
          f"{torch.cuda.memory_allocated() / 2**30:.1f} GiB on GPU", flush=True)
    model.config.use_cache = False
    return model, processor, {"source": source, "precision": precision,
                              "fp8_dequantized": fp8 and precision == "bf16"}


def add_lora(model, args, precision):
    from peft import LoraConfig, get_peft_model
    for p in model.parameters():
        p.requires_grad_(False)
    if precision == "qlora":
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    cfg = LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=args.dropout,
                     target_modules=TARGET_RE, bias="none", task_type="CAUSAL_LM")
    return get_peft_model(model, cfg)


def count_lora(model):
    return sum(1 for n, m in model.named_modules()
               if hasattr(m, "lora_A") and not n.endswith(("lora_A", "lora_B")))


def text_layers(model):
    cfg = model.config
    return getattr(getattr(cfg, "text_config", cfg), "num_hidden_layers")


def answer_loss(model, batch):
    """Mean cross-entropy over the answer tokens only."""
    import torch.nn.functional as F
    batch = dict(batch)
    mask = batch.pop("answer_mask")
    # The whole answer, not just its trained tokens: with fields="posture"
    # the mask has a gap where the activity value is.
    k = int(batch.pop("answer_len").max())
    try:
        logits = model(**batch, logits_to_keep=k + 1, use_cache=False).logits
    except TypeError:
        logits = model(**batch, use_cache=False).logits
    logits = logits[:, -(k + 1):-1, :].float()
    target = batch["input_ids"][:, -k:]
    m = mask[:, -k:].reshape(-1)
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1), reduction="none")
    return (loss * m).sum() / m.sum()


def answer_rows(model, processor, chat, rows, args, batch):
    """Greedy answers, `batch` cows (or, unit "frame", keyframes) at a time;
    yields the per-cow records of each batch. Generation starts where the
    final answer begins - no reasoning - which is what the adapter is trained
    to do. For frames, `rows` must hold every cow of each keyframe: the
    question lists them all."""
    import torch
    tok = processor.tokenizer
    prefix = chat.prefix + chat.answer_open
    stop = [tok.convert_tokens_to_ids(t) for t in re.findall(r"<\|[a-z_]+\|>", chat.answer_close)]
    if getattr(args, "unit", "cow") == "frame":
        frames = frame_mod.group(rows)
        with torch.no_grad():
            for i in range(0, len(frames), batch):
                chunk = frames[i:i + batch]
                images = [make_frame_image(args.root, f, args.width, args.jpeg_quality) for f in chunk]
                texts = [chat.prefix_for(frame_mod.prompt(f["cows"])) + chat.answer_open for f in chunk]
                enc = encode(processor, texts, images)
                enc = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in enc.items()}
                # ~16 tokens a cow, with room to spare
                most = max(len(f["cows"]) for f in chunk)
                out = model.generate(**enc, max_new_tokens=24 * most + 32, do_sample=False,
                                     use_cache=True, eos_token_id=stop or None,
                                     pad_token_id=tok.pad_token_id)
                answers = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
                yield [rec for f, text in zip(chunk, answers)
                       for rec in frame_mod.records(f["cows"], text)]
        return
    with torch.no_grad():
        for i in range(0, len(rows), batch):
            chunk = rows[i:i + batch]
            images = [make_image(args.root, r, args.width, args.jpeg_quality) for r in chunk]
            enc = encode(processor, [prefix] * len(chunk), images)
            enc = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in enc.items()}
            out = model.generate(**enc, max_new_tokens=48, do_sample=False, use_cache=True,
                                 eos_token_id=stop or None, pad_token_id=tok.pad_token_id)
            texts = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            recs = []
            for r, text in zip(chunk, texts):
                p, a, err = parse_answer(text)
                rec = dict(r, posture=p, activity=a, content=text)
                if err:
                    rec["parse_error"] = err
                recs.append(rec)
            yield recs


def error_rates(recs):
    n = max(len(recs), 1)
    bad_p = [r["posture"] != r["gt_posture"] for r in recs]
    bad_a = [r["activity"] != r["gt_activity"] for r in recs]
    return {"n": len(recs),
            "exact": sum(p or a for p, a in zip(bad_p, bad_a)) / n,
            "posture": sum(bad_p) / n, "activity": sum(bad_a) / n,
            "activity_answers": dict(collections.Counter(r["activity"] for r in recs))}


# ------------------------------------------------------------------ stages

def cmd_data(args):
    rows, dev, info = training_rows(args)
    print(json.dumps(info, indent=1))
    print("training on:", len(rows), "examples")
    print("  posture :", dict(collections.Counter(r["gt_posture"] for r in rows)))
    print("  activity:", dict(collections.Counter(r["gt_activity"] for r in rows)))
    print(f"dev: {len(dev)} cows scored, from {info['dev_cows_held_out']} in "
          f"{len(info['dev_clips'])} held-out clips")
    print("  activity:", dict(collections.Counter(r["gt_activity"] for r in dev)))
    val = read_jsonl(args.manifest)
    print("eval manifest:", args.manifest, len(val), "examples")
    overlap = {r["video_id"] for r in rows} & {r["video_id"] for r in val}
    if overlap:
        raise SystemExit(f"train and eval share clips {sorted(overlap)}")
    missing = set()
    for r in rows + dev + val:
        try:
            cbvd.frame_path(args.root, cbvd.Box(r["video_id"], r["timestamp"], *r["bbox"], "1", ()))
        except FileNotFoundError as exc:
            missing.add(str(exc))
    if missing:
        raise SystemExit("missing keyframes:\n  " + "\n  ".join(sorted(missing)[:10]))
    sample_dir = os.path.join(args.out, "samples")
    os.makedirs(sample_dir, exist_ok=True)
    if args.unit == "frame":
        frames = frame_mod.group(rows)
        print(f"unit frame: {len(frames)} keyframes, "
              f"{sum(len(f['cows']) for f in frames) / max(len(frames), 1):.1f} cows each, "
              f"up to {max(len(f['cows']) for f in frames)}")
        for f in random.Random(args.seed).sample(frames, min(3, len(frames))):
            stem = os.path.join(sample_dir, f"frame_{f['video_id']}_{f['timestamp']:05d}")
            make_frame_image(args.root, f, args.width, args.jpeg_quality).save(stem + ".jpg")
            with open(stem + ".txt", "w", encoding="utf-8") as fh:
                fh.write(frame_mod.prompt(f["cows"]) + "\n\n" + frame_mod.answer_json(f["cows"]) + "\n")
    for r in random.Random(args.seed).sample(rows, 4):
        make_image(args.root, r, args.width, args.jpeg_quality).save(
            os.path.join(sample_dir, f"{r['id']}_{r['gt_posture']}_{r['gt_activity']}.jpg"))
    print("samples ->", sample_dir)


def cmd_manifest(args):
    """A manifest of training cows, in the val manifest's format, so `eval`
    can score an adapter on the data it was trained on. Near train labels
    there means the training worked and val differs; far from them means the
    training itself went wrong."""
    rows, _, _ = training_rows(args)
    rows = subsample(rows, args.sample, args.seed + 1)
    out = args.manifest_out or os.path.join(args.out, "train-sample", "manifest.jsonl")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} training cows on {len({(r['video_id'], r['timestamp']) for r in rows})} "
          f"keyframes from {len({r['video_id'] for r in rows})} clips -> {out}")
    print("  posture :", dict(collections.Counter(r["gt_posture"] for r in rows)))
    print("  activity:", dict(collections.Counter(r["gt_activity"] for r in rows)))


def cmd_train(args):
    import torch
    from transformers import Trainer, TrainingArguments

    rows, dev, info = training_rows(args)
    os.makedirs(args.out, exist_ok=True)

    model, processor, minfo = load_model(args)
    processor.tokenizer.padding_side = "left"
    chat = Chat(processor)
    print(f"[chat] answer wrapped as {chat.answer_open!r} ... {chat.answer_close!r}", flush=True)
    collate = Collator(processor, chat, args.root, args.width, args.jpeg_quality,
                       args.train_fields, args.unit)
    # What the Trainer iterates over: cows, or whole keyframes.
    items = frame_mod.group(rows) if args.unit == "frame" else rows

    # --- sanity, before any hour is spent -----------------------------
    probe = collate(items[:2])
    ids, mask = probe["input_ids"], probe["answer_mask"]
    for i, r in enumerate(items[:2]):
        want = chat.suffix_ids(collate.suffix(r))
        got = ids[i][-len(want):].tolist()
        if got != want:
            raise SystemExit("answer tokens do not line up with the sequence end - the loss "
                             f"would train on the wrong tokens.\nwant {want}\ngot  {got}")
    seq_len = int(ids.shape[1])
    trained = chat.tok.decode(ids[0][mask[0]].tolist())
    print(f"[data] {len(items)} {args.unit} examples ({len(rows)} cows), {seq_len} tokens, answer "
          f"{int(probe['answer_len'].max())} tokens; fields={args.train_fields}, "
          f"loss on: {trained!r}", flush=True)

    val = read_jsonl(args.manifest)
    if args.unit == "frame":
        pick, step_ = random.Random(0).sample(frame_mod.group(val), 4), 2
    else:
        pick, step_ = random.Random(0).sample(val, min(16, len(val))), 4
    model.eval()
    zs = []
    with torch.no_grad():
        for i in range(0, len(pick), step_):
            b = {k: (v.to(model.device) if hasattr(v, "to") else v)
                 for k, v in collate(pick[i:i + step_]).items()}
            zs.append(float(answer_loss(model, b)))
    zero_shot = sum(zs) / len(zs)
    # A healthy base model knows JSON and knows these words. A loss far above
    # this bound means the weights did not load right - a missed scale, a
    # wrong layer mapping - and training would only paper over it.
    print(f"[sanity] zero-shot answer loss on {len(pick)} val {args.unit}s: {zero_shot:.3f} nats/token",
          flush=True)
    if zero_shot > args.max_zero_shot_loss:
        raise SystemExit(f"zero-shot loss {zero_shot:.2f} > {args.max_zero_shot_loss}: "
                         "the weights look broken. Not training.")

    model = add_lora(model, args, minfo["precision"])
    n_targets = count_lora(model)
    expected = text_layers(model) * TARGETS_PER_LAYER
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[lora] {n_targets} adapted modules (expected {expected}), "
          f"{trainable / 1e6:.0f}M trainable parameters", flush=True)
    if n_targets != expected:
        names = [n for n, _ in model.named_modules() if "layers.0." in n][:30]
        raise SystemExit(f"LoRA matched {n_targets} modules, expected {expected}. "
                         f"Layer 0 modules: {names}")

    # --- dev: held-out training clips, scored while training ------------
    # The adapter is snapshotted at every dev score and the best one - not
    # the last - becomes adapter/. Val stays untouched until the end, so it
    # remains an honest estimate.
    hist_path = os.path.join(args.out, "dev_history.jsonl")
    done_steps = {h["step"] for h in read_jsonl(hist_path)}

    def dev_eval(model, step):
        t = time.time()
        was_training = model.training
        model.eval()
        recs = [r for chunk in answer_rows(model, processor, chat, dev, args, args.dev_batch)
                for r in chunk]
        if was_training:
            model.train()
        torch.cuda.empty_cache()
        m = dict(step=step, **error_rates(recs), seconds=round(time.time() - t))
        if step > 0:
            model.save_pretrained(os.path.join(args.out, "adapters", f"step-{step:05d}"))
        with open(hist_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(m, ensure_ascii=False) + "\n")
        done_steps.add(step)
        print(f"\n[dev] step {step}: exact {m['exact']:.1%}  posture {m['posture']:.1%}  "
              f"activity {m['activity']:.1%}  answers {m['activity_answers']}  "
              f"({m['seconds']} s)", flush=True)

    from transformers import TrainerCallback

    class DevCallback(TrainerCallback):
        def on_step_end(self, _args, state, control, model=None, **kwargs):
            step = state.global_step
            if dev and args.dev_every and step % args.dev_every == 0 and step not in done_steps:
                dev_eval(model, step)

    if dev and 0 not in done_steps:
        dev_eval(model, 0)   # the adapter starts at zero: this is the base model

    class AnswerOnlyTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            loss = answer_loss(model, inputs)
            return (loss, None) if return_outputs else loss

    targs = TrainingArguments(
        output_dir=os.path.join(args.out, "checkpoints"),
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        # A float below 1 is a fraction of all steps. transformers 5 dropped
        # the separate warmup_ratio argument.
        warmup_steps=0.03,
        weight_decay=0.0,
        max_grad_norm=1.0,
        bf16=True,
        logging_steps=10,
        logging_first_step=True,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        dataloader_num_workers=args.workers,
        dataloader_persistent_workers=args.workers > 0,
        remove_unused_columns=False,
        report_to="none",
        seed=args.seed,
        optim="adamw_torch_fused",
    )
    trainer = AnswerOnlyTrainer(model=model, args=targs, train_dataset=items, data_collator=collate,
                                callbacks=[DevCallback()])
    # The loss is already a per-token mean; let Trainer do the plain
    # 1/accumulation scaling rather than look for token counts in `labels`.
    trainer.model_accepts_loss_kwargs = False

    last = None
    ckpts = sorted(glob.glob(os.path.join(targs.output_dir, "checkpoint-*")),
                   key=lambda p: int(p.rsplit("-", 1)[1]))
    if ckpts:
        last = ckpts[-1]
        print(f"[train] resuming from {last}", flush=True)
    t0 = time.time()
    result = trainer.train(resume_from_checkpoint=last)
    hours = (time.time() - t0) / 3600

    adapter = os.path.join(args.out, "adapter")
    best = None
    # Judged on what was trained: a posture-only adapter by posture, where
    # small activity drift between steps is noise, not something it learnt.
    metric = "posture" if args.train_fields == "posture" else "exact"
    if dev:
        if trainer.state.global_step not in done_steps:
            dev_eval(model, trainer.state.global_step)
        scored = {h["step"]: h for h in read_jsonl(hist_path) if h["step"] > 0}
        best = min(scored.values(), key=lambda h: (h[metric], h["exact"], h["step"]))
        shutil.rmtree(adapter, ignore_errors=True)
        shutil.copytree(os.path.join(args.out, "adapters", f"step-{best['step']:05d}"), adapter)
        print(f"[dev] best by {metric}: step {best['step']} of {trainer.state.global_step}, "
              f"posture {best['posture']:.1%}, exact {best['exact']:.1%} (base {read_jsonl(hist_path)[0]['exact']:.1%}) "
              f"-> {adapter}", flush=True)
    else:
        model.save_pretrained(adapter)
    losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
    meta = {
        "date": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        **minfo, **info,
        "unit": args.unit, "trained_on": len(rows), "train_examples": len(items),
        "train_limit": args.train_limit,
        "width": args.width, "jpeg_quality": args.jpeg_quality, "tokens_per_example": seq_len,
        "prompt_sha": frame_mod.PROMPT_SHA if args.unit == "frame" else client_mod.prompt_sha(1, 0.0),
        "lora": {"rank": args.rank, "alpha": args.alpha, "dropout": args.dropout,
                 "modules": n_targets, "trainable_params": trainable},
        "optim": {"lr": args.lr, "epochs": args.epochs, "batch": args.batch, "accum": args.accum,
                  "effective_batch": args.batch * args.accum, "steps": trainer.state.global_step},
        "train_fields": args.train_fields,
        "zero_shot_answer_loss": zero_shot,
        "dev": {"best": best, "chosen_by": metric, "every": args.dev_every, "history": hist_path},
        "first_loss": losses[0] if losses else None,
        "last_loss": sum(losses[-10:]) / len(losses[-10:]) if losses else None,
        "hours_this_session": round(hours, 2),
        "peak_gpu_gib": round(torch.cuda.max_memory_allocated() / 2**30, 1),
        "gpu": torch.cuda.get_device_name(0),
        "train_runtime_s": result.metrics.get("train_runtime"),
    }
    with open(os.path.join(args.out, "train_meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)
    print(f"[train] done: {hours:.1f} h, loss {meta['first_loss']} -> {meta['last_loss']}, "
          f"adapter -> {adapter}", flush=True)


def parse_answer(text):
    """JSON first; the enums by name as a fallback, so a stray token around a
    correct answer is not scored as the model being wrong."""
    m = re.search(r"\{.*?\}", text, re.S)
    if m:
        try:
            d = json.loads(m.group(0))
            p, a = d.get("posture"), d.get("activity")
            if p in cbvd.POSTURES and a in cbvd.ACTIVITIES:
                return p, a, None
        except json.JSONDecodeError:
            pass
    p = set(re.findall(r"\b(standing|lying)\b", text))
    a = set(re.findall(r"\b(feeding|drinking|ruminating|none)\b", text))
    if len(p) == 1 and len(a) == 1:
        return p.pop(), a.pop(), None
    return None, None, f"unparseable: {text[:120]!r}"


def cmd_eval(args):
    manifest = read_jsonl(args.manifest)
    if args.eval_limit:
        manifest = manifest[:args.eval_limit]
    os.makedirs(args.eval_out, exist_ok=True)
    results_path = os.path.join(args.eval_out, "results.jsonl")
    done = {r["id"] for r in read_jsonl(results_path)}
    todo = [r for r in manifest if r["id"] not in done]
    print(f"[eval] {len(manifest)} examples, {len(done)} done, {len(todo)} to go", flush=True)
    if not todo:
        return
    if args.unit == "frame":
        # The question lists every cow of the keyframe: ask with all of them,
        # write only the ones not answered yet.
        open_frames = {(r["video_id"], r["timestamp"]) for r in todo}
        ask = [r for r in manifest if (r["video_id"], r["timestamp"]) in open_frames]
    else:
        ask = todo

    model, processor, minfo = load_model(args)
    adapter = None if args.adapter in (None, "", "none") else args.adapter
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    model.config.use_cache = True
    processor.tokenizer.padding_side = "left"
    chat = Chat(processor)
    # Reasoning is skipped in both arms: the prompt ends where the final answer
    # begins. For the adapter that is what it was trained to do; for the base
    # model it makes the base arm a like-for-like control - same width, same
    # no-thinking setup - so the difference between the two is the training.
    meta_src = os.path.join(os.path.dirname(os.path.abspath(args.manifest)), "run_meta.json")
    excluded = []
    if os.path.exists(meta_src):
        with open(meta_src, encoding="utf-8") as fh:
            excluded = json.load(fh).get("excluded", [])
    tag = ("LoRA " + os.path.basename(os.path.dirname(os.path.abspath(adapter)))
           if adapter else "base, no reasoning")
    meta = {
        "run_date": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "model": f"muse-glimmer ({tag})",
        "model_repo": minfo["source"],
        "quantization": ("FP8 block (compressed-tensors), dequantized to BF16 for this run"
                         if minfo["fp8_dequantized"] else minfo["precision"]),
        "adapter": adapter,
        "engine": "transformers generate, greedy",
        "reasoning": "off: generation starts at the final answer",
        "temperature": 0.0, "seed": 0,
        "render_mode": "numbered (all cows of the frame)" if args.unit == "frame" else "marked",
        "unit": args.unit, "frames": 1, "span": 0.0, "min_width": 0,
        "max_width": args.width, "jpeg_quality": args.jpeg_quality,
        "prompt_sha": frame_mod.PROMPT_SHA if args.unit == "frame" else client_mod.prompt_sha(1, 0.0),
        "prompt": (frame_mod._PROMPT if args.unit == "frame" else client_mod.build_prompt(1, 0.0)),
        "annotations": os.path.join("annotations", "ava_val_v2.1.csv"),
        "videos": ", ".join(sorted({r["video_id"] for r in manifest}, key=int)),
        "excluded": excluded,
    }
    with open(os.path.join(args.eval_out, "run_meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)

    t0 = time.time()
    with open(results_path, "a", encoding="utf-8") as fh:
        n = 0
        for recs in answer_rows(model, processor, chat, ask, args, args.eval_batch):
            recs = [r for r in recs if r["id"] not in done]
            for rec in recs:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            n += len(recs)
            rate = n / (time.time() - t0)
            print(f"\r  {n}/{len(todo)}  {rate:.1f}/s  eta {(len(todo) - n) / rate / 60:.0f} min",
                  end="", flush=True)
    print(f"\n[eval] -> {results_path}", flush=True)


# ------------------------------------------------------------------ cli

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=("data", "train", "eval", "manifest"))
    p.add_argument("--root", required=True, help="unpacked CBVD-5 (with annotations/, labelframes/)")
    p.add_argument("--out", default="lora-out", help="training run directory")
    p.add_argument("--manifest", default=VAL_MANIFEST,
                   help="eval examples; default is the bench's full-val manifest, so ids match runs/")
    p.add_argument("--base", default=None,
                   help=f"model repo; default {FP8_REPO} (bf16) or {BF16_REPO} (qlora)")
    p.add_argument("--precision", choices=("auto", "bf16", "qlora"), default="auto",
                   help="auto: bf16 on >= 70 GiB GPUs, qlora below")
    p.add_argument("--attn", default="sdpa")
    p.add_argument("--width", type=int, default=896, help="frame width fed to the model, train and eval")
    p.add_argument("--jpeg-quality", type=int, default=90)
    p.add_argument("--train-limit", type=int, default=0,
                   help="cap on training cows (whole keyframes); 0 = all")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--alpha", type=int, default=16)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--accum", type=int, default=4)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--save-steps", type=int, default=50)
    p.add_argument("--max-zero-shot-loss", type=float, default=4.0)
    p.add_argument("--adapter", default=None, help="eval: adapter dir, or 'none' for the base model")
    p.add_argument("--eval-out", default=None, help="eval: output dir (default <out>/eval-lora|eval-base)")
    p.add_argument("--eval-batch", type=int, default=16)
    p.add_argument("--eval-limit", type=int, default=0)
    p.add_argument("--holdout", type=float, default=0.1,
                   help="share of training clips held out as dev (0: none, last adapter wins)")
    p.add_argument("--dev-size", type=int, default=300, help="dev cows scored (whole keyframes)")
    p.add_argument("--dev-every", type=int, default=100, help="score dev every N optimizer steps")
    p.add_argument("--dev-batch", type=int, default=8)
    p.add_argument("--frames-per-clip", type=int, default=3,
                   help="training keyframes kept per clip, spread over it; 0 = all six")
    p.add_argument("--unit", choices=("cow", "frame"), default="cow",
                   help="cow: one outlined cow per question; frame: every cow of the keyframe, "
                        "numbered, in one question (see ../frame.py)")
    p.add_argument("--train-fields", choices=("both", "posture", "skip-ruminating"), default="both",
                   help="posture: no loss on the activity value; skip-ruminating: none on the "
                        "activity value of ruminating cows (see Chat.loss_flags)")
    p.add_argument("--sample", type=int, default=800,
                   help="manifest: training cows to pick (whole keyframes)")
    p.add_argument("--manifest-out", default=None,
                   help="manifest: output file (default <out>/train-sample/manifest.jsonl)")
    args = p.parse_args(argv)
    if args.stage == "eval" and not args.eval_out:
        args.eval_out = os.path.join(
            args.out, "eval-base" if args.adapter in (None, "", "none") else "eval-lora")
    {"data": cmd_data, "train": cmd_train, "eval": cmd_eval,
     "manifest": cmd_manifest}[args.stage](args)


if __name__ == "__main__":
    main()
