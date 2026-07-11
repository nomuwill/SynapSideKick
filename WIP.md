# WIP — where we left off

Last updated: 2026-07-11

## Current goal

Validate a **per-channel template-matching detection** method on the htho thalamus
recording, using the existing curated KS2 sort as ground truth. This is the
"ignore real-time, prove the concept" version of `synapsortrt`.

### The method we want to test

For each **routed channel (electrode)**, build a hash-table entry holding:
- an array of every unit's **average waveform that occurs on that channel**,
- which **unit** each waveform is tied to,
- the **detection threshold** for that channel.

Then, on the held-out part of the recording: when a channel crosses its threshold,
template-match the snippet against every stored waveform for that channel. Evaluate:

1. **Is it a real spike?** — does a threshold crossing that matches a template
   (corr ≥ thresh) correspond to a true spike? (precision of matched detections)
2. **Right unit?** — does the matched unit equal the actual firing unit from the
   curated sort, within a ±window? (per-unit P/R/F1 + confusion among units that
   share a channel)
3. **Main unit firing?** — does a crossing+match on a channel line up with that
   unit's spike on its **main** channel? (cross-channel consistency)

Train/test split: build templates from spikes in **[0, 300 s)**, evaluate detection
on **[300 s, 790 s]**. (Full recording is 790 s / ~13.2 min, 1019 ch @ 20 kHz.)

## Dataset

`Data/htho_agg/24448a_htho_d63_061626/` (chip 24448a, human thalamic organoid, day 63):
- `raw/24448a_htho_d63_061626.raw.h5` — 5 GB, 1019 routed ch, 15,800,600 samples @ 20 kHz
- `sorted_spikedata_curated.pkl` — 289 units (curated on spike-time criteria)
- `sorted_spikedata.pkl` — 318 units (uncurated, spike times only)

Fetched/curated via `scripts/fetch_htho_24448a_d63.py` + `scripts/curate_htho_24448a_d63.py`.
Source: public S3 endpoint `https://s3.braingeneers.gi.ucsc.edu`, bucket `braingeneers`
(no creds needed via this endpoint), UUID `2026-06-22-e-htho-agg_041326`.

## ⚠️ BLOCKER / open decision — resolve this first

The **multi-channel neighbor templates are NOT saved** in either pickle — verified:

| | curated (289) | uncurated (318) |
|---|---|---|
| `avg_waveform` | `(1, 60)` — 1 channel | `None` |
| `waveforms` | `(1, 60, n_spikes)` — 1 channel | `None` |
| `neighbor_channels` / `neighbor_templates` / `template` | `None` | `None` |
| `traces_meta.channels` | `(1,)` | — |

Cause: `scripts/curate_htho_24448a_d63.py:89` extracts waveforms on the unit's
**assigned KS2 channel only**. All 289 best-channels are distinct, so with
single-channel templates every channel's "array of waveforms" has length 1 — which
defeats the whole per-channel-collision idea.

Noah's position (last message): the entire multi-channel array *should* be there.
It isn't on disk, but the footprints exist in the source data. Two ways to recover:

- **Option A (recommended)** — re-extract multi-channel templates from the raw `.h5`:
  for each unit, at its spike times, pull waveforms across a neighborhood of channels
  and average → true-µV multi-channel templates. Matches the existing
  `TemplateLibrary.from_spikedata_pickle` path and the "SpikeInterface for
  extraction" rule. Raw file is already local.
  - Sub-decision: neighborhood = all 1019 channels thresholded by amplitude, **or**
    spatial neighbors within a radius of the best channel? (electrode x,y are in
    `neuron_attributes[i]['location']`.)
- **Option B** — re-fetch KS2 phy `templates.npy` from S3 (all-channel, but whitened
  low-rank; needs un-whitening for µV). The fetch script downloads the phy zip but
  currently discards `templates.npy`.

**NEXT ACTION: get Noah's answer on A vs B (and the neighborhood definition), then
this becomes Phase A of the plan below.**

## Planned experiment (once template source is resolved)

Reuse the existing `synapsortrt` components — do not reimplement:
- `synapsortrt/templates.py` — `TemplateLibrary` (per-unit templates, active channels,
  per-channel `waveforms`). Add/verify a multi-channel loader (Phase A).
- `synapsortrt/detector.py` — `Detector` (MAD threshold crossing + snippet extraction).
- `synapsortrt/matcher.py` — `Matcher` already builds the per-channel hash table
  internally (`_ch_templates[ch]`, `_ch_unit_ids[ch]`). Extend to (a) store a
  per-channel threshold in the same structure and (b) serialize it.
- `synapsortrt/pipeline.py` — `SynapsortPipeline.run()` (chunked detect+match).
- `scripts/benchmark_templates.py` — has `evaluate()` for P/R/F1 vs ground truth;
  extend with confusion matrix, cross-channel consistency, and cosine-sim distribution.

Rough phases (each can be a sub-agent task, own working dir under `experiments/`):
- **A. Templates** — build multi-channel `TemplateLibrary` from spikes in [0,300 s);
  emit the per-channel hash table (channel → templates, unit_ids, threshold). Save npz.
- **B. Detect + match** — run pipeline on [300,790 s]; save detected+matched events.
- **C. Evaluate** — vs curated ground truth in [300,790 s]: per-unit P/R/F1, confusion
  matrix (channel-sharing units), cross-channel consistency, cosine-sim distribution
  (matched-true vs mismatched-false → informs `corr_thresh`). Figures + summary.

Also on the wishlist (from earlier in the conversation, lower priority):
- Manual curated-unit selection: add a `unit_ids` filter param to the
  `TemplateLibrary.from_spikedata_pickle*` loaders so a subset of trusted units can be
  chosen independent of the pickle contents.
- Per-unit metadata binning (tissue type, age, …) for the training set — tissue+age
  are already encoded in the `Data/` path convention (`<tissue>_<line>/<chip>_<tissue>_d<age>_<date>`).
- `synapremoval` artifact rejection: use logged stim timestamps to blank post-stim
  windows before detection. Deferred — current datasets are non-stim. Needs a
  recording with a stim log first.

## Environment gotchas (for the other computer)

- Conda env: `spikelab` (or `brain`) — has spikeinterface 0.104, spikelab, torch+CUDA.
  GPU is an RTX 5090 (CUDA 13.1 driver), `torch.cuda.is_available()` == True.
- **Maxwell `.h5` loading**: use `synapconnect.software.maxwell_bridge.load_maxwell_recording`
  (passes `install_maxwell_plugin=False` and handles a duplicate-channel-id fallback).
  Otherwise spikeinterface's auto-installer crashes trying to `mkdir` a stale
  `/home/mxwbio/...` entry in `HDF5_PLUGIN_PATH`. Correct plugin dir on this machine:
  `/home/sharf-lab/MaxLab/so/` (contains `libcompression.so`). Root cause is a bad
  line in `~/.profile:32` (not fixed — it's outside this repo).
- **Library rule** (standing instruction): SpikeInterface for all sorting/extraction;
  SpikeLab only for its data objects (`SpikeData` etc.); never `spikelab.spike_sorting`.
- KS2-from-MATLAB is available (`/home/sharf-lab/Kilosort2`, MATLAB R2025b) but we
  abandoned running a fresh KS2 — the curated sort already provides the units.

## Repo orientation

See `REPO_MAP.md` (added this session) for the full layout and `synapsortrt` class
relationships. `Data/` is gitignored except its README.
