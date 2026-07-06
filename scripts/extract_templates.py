"""
extract_templates.py — Phase 1 CLI

Extract raw waveform templates from a SpikeLab SpikeData pickle + matching
Maxwell .h5 recording, and save a SynapSortRT TemplateLibrary.

Usage:
    python extract_templates.py \
        --pickle /path/to/spikedata.pkl \
        --h5     /path/to/recording.raw.h5 \
        --spikelab-src /path/to/SpikeLab/src \
        --output templates \
        [--duration 180]
"""

import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "synapsortrt"))
from synapsortrt.templates import TemplateLibrary

parser = argparse.ArgumentParser()
parser.add_argument("--pickle",       required=True, help="SpikeData .pkl path")
parser.add_argument("--h5",           required=True, help="Maxwell .h5 recording path")
parser.add_argument("--spikelab-src", required=True, help="Path to SpikeLab src/")
parser.add_argument("--output",       default="templates", help="Output path (no extension)")
parser.add_argument("--duration",     type=float, default=None, help="Seconds to use (default: all)")
parser.add_argument("--amp-thresh",   type=float, default=0.2)
parser.add_argument("--ms-before",    type=float, default=1.0)
parser.add_argument("--ms-after",     type=float, default=2.0)
args = parser.parse_args()

print(f"Extracting raw templates from SpikeData pickle ...")
lib = TemplateLibrary.from_spikedata_pickle(
    pickle_path=args.pickle,
    spikelab_src=args.spikelab_src,
    raw_h5_path=args.h5,
    amp_thresh=args.amp_thresh,
    duration_s=args.duration,
    ms_before=args.ms_before,
    ms_after=args.ms_after,
)
print(lib)
out = lib.save(args.output)
print(f"Saved to {out}")
