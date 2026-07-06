"""
benchmark_templates.py — compare template matching performance for:

  Method A: avg_stored    — pre-averaged waveforms from SpikeData pickle
  Method B: raw_extracted — waveforms re-extracted from raw .h5

Evaluation against ground-truth spike times from the SpikeData pickle.

Metrics per unit and overall:
  - Precision : matched detections that are true spikes
  - Recall    : true spikes that were detected and matched
  - F1        : harmonic mean of precision and recall

Usage:
    python benchmark_templates.py \
        --pickle  /path/to/spikedata.pkl \
        --h5      /path/to/recording.raw.h5 \
        --spikelab-src /path/to/SpikeLab/src \
        --duration 180 \
        --output  benchmark_results
"""

import argparse, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "synapsortrt"))
from synapsortrt.templates import TemplateLibrary
from synapsortrt.pipeline import SynapsortPipeline
import spikeinterface.extractors as se

MATCH_WINDOW_MS = 1.0   # ±ms to consider a detection a true positive

parser = argparse.ArgumentParser()
parser.add_argument("--pickle",       required=True)
parser.add_argument("--h5",           required=True)
parser.add_argument("--spikelab-src", required=True)
parser.add_argument("--duration",     type=float, default=180.0)
parser.add_argument("--corr-thresh",  type=float, default=0.75)
parser.add_argument("--output",       default="benchmark_results")
args = parser.parse_args()

SPIKELAB_SRC = args.spikelab_src
if SPIKELAB_SRC not in sys.path:
    sys.path.insert(0, SPIKELAB_SRC)
from spikelab.data_loaders import load_spikedata_from_pickle


def evaluate(spike_trains_pred, spike_trains_gt, fs, window_ms=MATCH_WINDOW_MS):
    """
    Compute precision, recall, F1 per unit and overall.

    Args:
        spike_trains_pred : dict[unit_id -> np.ndarray of spike times in ms]
        spike_trains_gt   : dict[unit_id -> np.ndarray of spike times in ms]
        fs                : sampling frequency (for reference)
        window_ms         : matching window in ms

    Returns:
        dict with per-unit and overall metrics
    """
    results = {}
    tp_total, fp_total, fn_total = 0, 0, 0

    for uid in spike_trains_gt:
        gt = np.sort(spike_trains_gt[uid])
        pred = np.sort(spike_trains_pred.get(uid, np.array([])))

        tp, fp, fn = 0, 0, 0
        matched_gt = set()

        for t in pred:
            diffs = np.abs(gt - t)
            if len(diffs) == 0:
                fp += 1
                continue
            best = int(np.argmin(diffs))
            if diffs[best] <= window_ms and best not in matched_gt:
                tp += 1
                matched_gt.add(best)
            else:
                fp += 1

        fn = len(gt) - len(matched_gt)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = (2 * precision * recall / (precision + recall)
                     if (precision + recall) > 0 else 0.0)

        results[uid] = dict(tp=tp, fp=fp, fn=fn,
                            precision=precision, recall=recall, f1=f1,
                            n_gt=len(gt), n_pred=len(pred))
        tp_total += tp; fp_total += fp; fn_total += fn

    prec_overall = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0
    rec_overall  = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0
    f1_overall   = (2 * prec_overall * rec_overall / (prec_overall + rec_overall)
                    if (prec_overall + rec_overall) > 0 else 0)
    results["_overall"] = dict(precision=prec_overall, recall=rec_overall,
                               f1=f1_overall, tp=tp_total, fp=fp_total, fn=fn_total)
    return results


# ── load ground truth ────────────────────────────────────────────────────────
print("Loading ground truth spike trains ...")
sd = load_spikedata_from_pickle(args.pickle)
gt_trains = {}
for i, times_ms in enumerate(sd.train):
    times_ms = times_ms[times_ms <= args.duration * 1000.0]
    if len(times_ms) > 0:
        gt_trains[i] = times_ms

# ── load raw recording ───────────────────────────────────────────────────────
print("Loading raw recording ...")
rec = se.MaxwellRecordingExtractor(args.h5)
fs = rec.get_sampling_frequency()
n_frames = min(int(args.duration * fs), rec.get_num_frames())
gains = rec.get_property("gain_to_uV")
ch_ids = rec.get_channel_ids()
raw = rec.get_traces(start_frame=0, end_frame=n_frames,
                     channel_ids=ch_ids, return_in_uV=False).astype(np.float32)
traces = (raw * gains[None, :]).T

# ── run both methods ─────────────────────────────────────────────────────────
results = {}

for method, lib in [
    ("avg_stored", TemplateLibrary.from_spikedata_pickle_avg(
        args.pickle, args.spikelab_src,
        duration_s=args.duration)),
    ("raw_extracted", TemplateLibrary.from_spikedata_pickle(
        args.pickle, args.spikelab_src, args.h5,
        duration_s=args.duration)),
]:
    print(f"\n── {method} ──")
    print(lib)
    pipe = SynapsortPipeline(
        traces=traces,
        channel_ids=np.arange(len(ch_ids)),
        fs=fs,
        library=lib,
        corr_thresh=args.corr_thresh,
    )
    spike_trains = pipe.run(verbose=True)
    metrics = evaluate(spike_trains, gt_trains, fs)
    results[method] = metrics
    ov = metrics["_overall"]
    print(f"  Overall — P={ov['precision']:.3f}  R={ov['recall']:.3f}  F1={ov['f1']:.3f}")

# ── save results ─────────────────────────────────────────────────────────────
out = Path(args.output).with_suffix(".npz")
np.savez(out, results=results)
print(f"\nSaved benchmark results to {out}")
print("\n── Summary ──")
for method, metrics in results.items():
    ov = metrics["_overall"]
    print(f"  {method:20s}  P={ov['precision']:.3f}  R={ov['recall']:.3f}  F1={ov['f1']:.3f}")
