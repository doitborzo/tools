"""Markdown report.

The header block is not decoration: model name, test date, quantization and
dataset link were explicitly asked for, so they are generated from the run
metadata rather than typed in afterwards and going stale.
"""

from __future__ import annotations

from cbvd import ACTIVITIES, POSTURES

DATASET_NAME = "CBVD-5 (Cow Behavior Video Dataset)"
DATASET_URL = "https://www.kaggle.com/datasets/fandaoerji/cbvd-5cow-behavior-video-dataset"
DATASET_PAPER = "https://www.nature.com/articles/s41598-024-65953-x"

# Not discoverable over the OpenAI API: vLLM reports the quantization method in
# its startup log ("quantization=compressed-tensors"), not in /v1/models.
QUANTIZATION = {
    "RedHatAI/Muse-Glimmer-30B-FP8-block":
        "FP8 block-wise (W8A8, compressed-tensors); KV cache: auto (bf16)",
}


def _pct(x):
    return f"{100 * x:.1f}%"


def _rows_section(add, results):
    """Every cow, one line each.

    Ordered by keyframe and then left to right across the frame, so a row can
    actually be found in the picture: the annotation carries no identity, and
    "cow 47" means nothing without a position to look at.
    """
    add("## Every example")
    add("")
    add(f"All {len(results)} annotated cows, {len(set((r['video_id'], r['timestamp']) for r in results))} "
        f"keyframes. Ordered by keyframe, then left to right. `x` marks the horizontal "
        f"centre of the box as a fraction of frame width, `y` the vertical centre.")
    add("")
    add("| # | clip | t, s | x | y | ground truth | prediction | |")
    add("|---|---|---|---|---|---|---|---|")

    ordered = sorted(results, key=lambda r: (r["video_id"], r["timestamp"],
                                             (r["bbox"][0] + r["bbox"][2]) / 2))
    for i, r in enumerate(ordered, 1):
        x1, y1, x2, y2 = r["bbox"]
        gt = f"{r['gt_posture']} / {r['gt_activity']}"
        failed = bool(r.get("error") or r.get("parse_error")
                      or r.get("posture") is None or r.get("activity") is None)
        if failed:
            pred, mark = "—", "no answer"
        else:
            pred = f"{r['posture']} / {r['activity']}"
            wrong = []
            if r["posture"] != r["gt_posture"]:
                wrong.append("posture")
            if r["activity"] != r["gt_activity"]:
                wrong.append("activity")
            mark = "ok" if not wrong else " + ".join(wrong)
        add(f"| {i} | {r['video_id']} | {r['timestamp']} | {(x1 + x2) / 2:.2f} | "
            f"{(y1 + y2) / 2:.2f} | {gt} | {pred} | {mark} |")
    add("")


def render(meta: dict, metrics: dict, results=None) -> str:
    model_repo = meta.get("model_repo") or meta.get("model") or "unknown"
    quant = meta.get("quantization") or QUANTIZATION.get(model_repo, "unknown")

    lines = []
    add = lines.append

    add("# Muse Glimmer on CBVD-5: cow behaviour recognition")
    add("")
    add("| | |")
    add("|---|---|")
    add(f"| Model | `{model_repo}` |")
    add(f"| Served as | `{meta.get('model')}` |")
    add(f"| Quantization | {quant} |")
    add(f"| Test date | {meta.get('run_date')} |")
    add(f"| Dataset | [{DATASET_NAME}]({DATASET_URL}) — [paper]({DATASET_PAPER}) |")
    add(f"| Split / source | `{meta.get('annotations')}` |")
    clips = meta.get("videos") or ""
    if metrics.get("n_clips", 0) > 8:
        clips = f"{metrics['n_clips']} clips"
    add(f"| Clips | {clips} |")
    add(f"| Keyframes | {metrics.get('n_keyframes', '?')} |")
    add(f"| Examples (annotated boxes) | **{metrics['n_examples']}** |")
    add(f"| **Exact-match error rate** | **{_pct(metrics['exact_match']['error_rate'])}** |")
    if meta.get("engine"):
        # Runs made off the server, e.g. the LoRA eval in lora/train_lora.py:
        # no vLLM, no structured output, and reasoning may be switched off.
        add(f"| Serving | {meta['engine']} |")
        add(f"| Sampling | temperature={meta.get('temperature')}, seed={meta.get('seed')}, "
            f"reasoning {meta.get('reasoning', '?')} |")
        if meta.get("adapter"):
            add(f"| Adapter | `{meta['adapter']}` |")
    else:
        add(f"| Serving | vLLM {meta.get('vllm_version') or '?'} |")
        output = ("structured output (json_schema)"
                  if not str(meta.get("reasoning", "")).startswith("off")
                  else f"reasoning {meta['reasoning']}, no json_schema")
        add(f"| Sampling | temperature={meta.get('temperature')}, seed={meta.get('seed')}, "
            f"max_tokens={meta.get('max_tokens')}, {output} |")
    frames = meta.get("frames") or 1
    temporal = ("annotated keyframe only (single still)" if frames <= 1
                else f"{frames} frames over {meta.get('span')}s from the clip")
    add(f"| Image mode | {meta.get('render_mode')}, max width {meta.get('max_width')}px |")
    add(f"| Temporal context | {temporal} |")
    add(f"| Prompt revision | `{meta.get('prompt_sha')}` |")
    add("")

    add("## What is being measured")
    add("")
    add("One example = one annotated cow on one keyframe. The model sees the full "
        "keyframe with that cow outlined and answers two independent fields, matching "
        "how CBVD-5 annotates: a **posture** (standing / lying) and an **activity** "
        "(feeding / drinking / ruminating / none). Exact match means both are right.")
    add("")
    if frames <= 1:
        add("**Read `ruminating` with that in mind.** Rumination is chewing cud — a "
            "movement. On one still frame a ruminating cow and an idle one are the "
            "same picture, so errors on that class measure the frame, not the model. "
            "Re-run with `--frames 5` to give it the motion.")
    else:
        add(f"The bounding box is annotated on the middle frame only; over "
            f"{meta.get('span')}s a moving cow drifts off its own outline. That is a "
            f"cost of the temporal context, paid to make `ruminating` observable at all.")
    add("")

    add("## Headline numbers")
    add("")
    add("| Axis | Error rate | 95% CI | Majority baseline | Correct |")
    add("|---|---|---|---|---|")
    for axis, label in (("exact_match", "Exact match (both fields)"),
                        ("posture", "Posture only"),
                        ("activity", "Activity only")):
        m = metrics[axis]
        base = metrics["baseline"].get(axis.replace("exact_match", "posture"))
        base_txt = "—" if axis == "exact_match" else \
            f"{_pct(base['error_rate'])} (always `{base['label']}`)"
        if axis == "activity":
            b = metrics["baseline"]["activity"]
            base_txt = f"{_pct(b['error_rate'])} (always `{b['label']}`)"
        add(f"| {label} | **{_pct(m['error_rate'])}** | "
            f"{_pct(m['ci95'][0])} – {_pct(m['ci95'][1])} | {base_txt} | "
            f"{m['correct']}/{m['n']} |")
    add("")
    if metrics["n_failed"]:
        add(f"{metrics['n_failed']} of {metrics['n_examples']} requests produced no usable "
            f"answer (server error, refusal or truncated output). They are counted as "
            f"errors above and listed at the end.")
        add("")

    # Written from the numbers, not from an impression of them: an error rate
    # on a set this skewed can look respectable while the model is answering
    # the majority class to everything.
    notes = []
    for axis, values in (("posture", POSTURES), ("activity", ACTIVITIES)):
        predicted = (metrics.get("predicted") or {}).get(axis, {})
        support = metrics["support"][axis]
        for value in values:
            if support.get(value, 0) and not predicted.get(value, 0):
                notes.append(
                    f"The model never answered `{value}` once, on any of the "
                    f"{metrics['n_examples']} examples — although {support[value]} "
                    f"are annotated as such. Recall for it is 0 by construction.")
        base = metrics["baseline"][axis]["error_rate"]
        got = metrics[axis]["error_rate"]
        if got >= base:
            notes.append(
                f"On {axis} the model does not beat the majority baseline "
                f"({_pct(got)} vs {_pct(base)} for answering `{base and metrics['baseline'][axis]['label']}` "
                f"to everything).")
        elif base - got < 0.05:
            notes.append(
                f"On {axis} the model is within {_pct(base - got)} of the majority "
                f"baseline ({_pct(got)} vs {_pct(base)}) — close enough that the "
                f"headline number is carried by the class skew, not by the model.")
    if notes:
        add("## Reading the numbers")
        add("")
        for note in notes:
            add(f"- {note}")
        add("")

    add("## Predictions given vs annotated")
    add("")
    add("| Axis | Class | Annotated | Predicted |")
    add("|---|---|---|---|")
    for axis, values in (("posture", POSTURES), ("activity", ACTIVITIES)):
        predicted = (metrics.get("predicted") or {}).get(axis, {})
        for value in values:
            add(f"| {axis} | `{value}` | {metrics['support'][axis].get(value, 0)} | "
                f"{predicted.get(value, 0)} |")
    add("")

    by_clip = metrics.get("by_clip") or []
    if by_clip:
        rates = [c["error_rate"] for c in by_clip]
        add("## Error rate by clip")
        add("")
        add(f"{len(by_clip)} clips, worst first. Spread {_pct(min(rates))} – "
            f"{_pct(max(rates))}. A single-clip number sits somewhere in this range "
            f"and says as much about that camera angle as about the model.")
        add("")
        add("| Clip | Keyframes | Cows | Exact-match error |")
        add("|---|---|---|---|")
        for c in by_clip:
            add(f"| {c['clip']} | {c['keyframes']} | {c['n']} | "
                f"{_pct(c['error_rate'])} |")
        add("")

    by_size = metrics.get("by_size") or []
    if by_size:
        add("## Error rate by how big the cow is in frame")
        add("")
        add("Quartiles of bounding-box area. `y centre` is where those boxes sit "
            "vertically, which in this barn is also how far away they are: the near "
            "feed barrier is the lower half of the frame, the far cubicle row the upper.")
        add("")
        add("| Quartile | Box area, % of frame | n | y centre | Annotated `lying` | "
            "Exact-match error |")
        add("|---|---|---|---|---|---|")
        for band in by_size:
            add(f"| Q{band['quartile']} | {band['area_pct_min']:.2f} – "
                f"{band['area_pct_max']:.2f} | {band['n']} | "
                f"{band['y_centre_median']:.2f} | {_pct(band['lying_share'])} | "
                f"**{_pct(band['error_rate'])}** |")
        add("")

    add("## Per class")
    add("")
    add("| Class | Support | Precision | Recall | F1 |")
    add("|---|---|---|---|---|")
    for key, m in metrics["per_class"].items():
        add(f"| `{key}` | {m['support']} | {m['precision']:.2f} | "
            f"{m['recall']:.2f} | {m['f1']:.2f} |")
    add("")

    for axis, values in (("posture", POSTURES), ("activity", ACTIVITIES)):
        add(f"## Confusion matrix — {axis}")
        add("")
        cols = list(values) + ["<failed>"]
        add("| ground truth / predicted | " + " | ".join(f"`{c}`" for c in cols) + " |")
        add("|---" * (len(cols) + 1) + "|")
        for gt in values:
            row = metrics["confusion"][axis][gt]
            add(f"| `{gt}` | " + " | ".join(str(row[c]) for c in cols) + " |")
        add("")

    if results:
        _rows_section(add, results)

    if metrics["failures"]:
        add("## Failed requests")
        add("")
        for f in metrics["failures"]:
            add(f"- `{f['id']}` — {f['reason']}")
        add("")

    excluded = meta.get("excluded") or []
    if excluded:
        add("## Excluded from scoring")
        add("")
        add("Boxes whose annotation has no single correct answer (contradictory or "
            "missing posture label). They are dropped before the model is queried, "
            "so they affect neither the count nor the error rate.")
        add("")
        for item in excluded:
            add(f"- `{item['id']}` — {item['reason']}")
        add("")

    return "\n".join(lines) + "\n"
