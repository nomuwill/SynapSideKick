"""
run_sorting.py — Phase 2 CLI

Run SynapSortRT detection + matching on a Maxwell .h5 recording.

Usage:
    python run_sorting.py --recording file.raw.h5 --templates templates.npz --output spike_trains.npz
"""

import argparse, sys, os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "synapsortrt"))

from synapsortrt.templates import TemplateLibrary
from synapsortrt.pipeline import SynapsortPipeline
import spikeinterface.extractors as se

parser = argparse.ArgumentParser()
parser.add_argument("--recording",   required=True)
parser.add_argument("--templates",   required=True)
parser.add_argument("--output",      default="spike_trains")
parser.add_argument("--duration",    type=float, default=30.0, help="Seconds to process")
parser.add_argument("--thresh-mult", type=float, default=4.0)
parser.add_argument("--corr-thresh", type=float, default=0.75)
parser.add_argument("--chunk-s",     type=float, default=1.0)
args = parser.parse_args()

print(f"Loading templates from {args.templates} ...")
lib = TemplateLibrary.from_npz(args.templates)
print(lib)

print(f"Loading recording: {args.recording} ...")
rec = se.MaxwellRecordingExtractor(args.recording)
fs = rec.get_sampling_frequency()
n_frames = min(int(args.duration * fs), rec.get_num_frames())
gains = rec.get_property("gain_to_uV")
channel_ids = rec.get_channel_ids()

print(f"  {len(channel_ids)} channels, {n_frames/fs:.1f}s @ {fs:.0f} Hz")
print("Loading traces ...")
raw = rec.get_traces(start_frame=0, end_frame=n_frames,
                     channel_ids=channel_ids, return_in_uV=False).astype(np.float32)
traces = (raw * gains[None, :]).T  # (n_ch, n_frames)

pipe = SynapsortPipeline(
    traces=traces,
    channel_ids=np.arange(len(channel_ids)),
    fs=fs,
    library=lib,
    thresh_mult=args.thresh_mult,
    corr_thresh=args.corr_thresh,
)

spike_trains = pipe.run(chunk_s=args.chunk_s)

out = Path(args.output)
if not out.suffix:
    out = out.with_suffix(".npz")
np.savez(out, spike_trains=spike_trains)
print(f"Saved spike trains to {out}")
