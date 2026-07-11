# SynapSideKick Repo Map

Quick orientation for this repo. For narrative/pipeline usage see [README.md](README.md); for the subagent system see [agents/README.md](agents/README.md).

## Directory structure

```
SynapSideKick/
├── README.md               Pipeline overview, usage examples
├── REPO_MAP.md              This file
├── agents/README.md         Index of Claude Code subagents (definitions live in .claude/agents/)
├── .claude/agents/           Subagent definitions (maxwell-hardware.md, ...)
│
├── synapsortrt/              IMPLEMENTED — real-time spike sorting via template matching
│   ├── templates.py           TemplateLibrary — load/save unit templates from any sorter
│   ├── detector.py             Detector — threshold crossing + waveform extraction
│   ├── matcher.py               Matcher — GPU cosine-similarity template matching
│   └── pipeline.py               SynapsortPipeline — ties Detector + Matcher together
│
├── synapremoval/              STUB — stimulation artifact detection and removal (not yet built)
├── synapanalysis/              STUB — spike train analysis pipelines (not yet built)
├── synapkit/                    Top-level package; re-exports synapsortrt
│
├── synapconnect/                Hardware/software connectors
│   ├── hardware/maxwell/          mxwserver + maxlab API wrappers
│   │   ├── server.py                is_running / start / stop / status (mxwserver process)
│   │   ├── session.py                initialize / activate (chip-level maxlab API)
│   │   └── REALTIME.md               Closed-loop options on the rig (C++ API vs ZMQ streaming)
│   ├── hardware/3brain.py          STUB — placeholder if 3Brain rig support is ever added
│   └── software/maxwell_bridge.py   load_maxwell_recording / to_spikedata — .raw.h5 → SpikeInterface/SpikeLab
│
├── scripts/                     CLIs for the synapsortrt pipeline + one-off data-fetch scripts
│   ├── extract_templates.py       Phase 1 CLI — build a TemplateLibrary from a SpikeData pickle
│   ├── run_sorting.py               Phase 2 CLI — run detect+match on a recording
│   ├── benchmark_templates.py        avg_stored vs raw_extracted template accuracy (P/R/F1)
│   ├── fetch_htho_24448a_d63.py       Downloads one training-set recording pair from S3
│   └── curate_htho_24448a_d63.py       Curates + attaches waveforms for that recording
│
└── Data/                        Gitignored (except README.md) — local training-set cache
    └── <tissue>_<line>/<chip>_<tissue>_d<age>_<date>/
        ├── raw/<stem>.raw.h5
        ├── sorted_spikedata.pkl
        └── sorted_spikedata_curated.pkl
```

## Status at a glance

| Subpackage | Status | Purpose |
|---|---|---|
| `synapsortrt` | Built, being validated | Detect + template-match spikes on a recording (chunked, not yet live-streamed) |
| `synapconnect` | Built | Talk to the physical/software Maxwell rig; load `.raw.h5` files |
| `synapremoval` | Stub only | Planned: reject/blank stim-artifact windows before detection, using logged stim timestamps as ground truth |
| `synapanalysis` | Stub only | Planned: spike train analysis on `synapsortrt` output |
| `synapkit` | Built | Top-level namespace package, re-exports `synapsortrt` |

## `synapsortrt` — core classes

```
TemplateLibrary  (templates.py)
      │  .from_kilosort() / .from_spikedata_pickle() / .from_spikedata_pickle_avg() / .from_npz()
      │  .save() / .load()
      ▼
   Matcher  (matcher.py)  ◄── built once from a TemplateLibrary, holds per-channel
      │                        normalized template tensors on GPU (mps/cuda/cpu)
      │  .match(events: list[CrossingEvent]) -> list[MatchResult]
      ▲
CrossingEvent  ◄──  Detector  (detector.py)
                       │  .set_thresholds(traces)   MAD-based per-channel noise threshold
                       │  .detect(traces) -> list[CrossingEvent]   trough-centered snippets

SynapsortPipeline  (pipeline.py)
      Wraps Detector + Matcher; .run(chunk_s=1.0) -> dict[unit_id -> spike_times_ms]
      processes a full recording in chunks (offline/batch, not a live stream yet)
```

| Class | Key fields/methods | Notes |
|---|---|---|
| `TemplateLibrary` | `templates`, `active_channels`, `waveforms`, `unit_ids`, `fs`, `source` | Sorter-agnostic. `source` tracks provenance: `"avg_stored"`, `"raw_extracted"`, `"kilosort"` |
| `Detector` | `set_thresholds()`, `detect()` | `thresh_mult=4.0` default (MAD-based), `n_samples=82`, `refractory_ms=1.0` |
| `Matcher` | `match()`, `corr_thresh=0.75` | One `torch.mv()` per channel per event batch; device auto-detected (mps/cuda/cpu) |
| `SynapsortPipeline` | `run(chunk_s=1.0)` | Chunked processing keeps memory bounded; not the real-time/streaming path yet |

## `synapconnect` — hardware/software connectors

| Function | File | Purpose |
|---|---|---|
| `is_running()`, `start()`, `stop()`, `status()` | `hardware/maxwell/server.py` | Control the `mxwserver` process |
| `initialize(wells)`, `activate(wells)` | `hardware/maxwell/session.py` | Thin wrapper around the `maxlab` chip-level API |
| `load_maxwell_recording(h5_path)` | `software/maxwell_bridge.py` | `.raw.h5` → SpikeInterface `BaseRecording`, with a dedup fallback for firmware quirks |
| `to_spikedata(recording)` | `software/maxwell_bridge.py` | SpikeInterface recording → SpikeLab `SpikeData`, via SpikeLab's own converter |

Real-time options are documented in [REALTIME.md](synapconnect/hardware/maxwell/REALTIME.md) — two paths exist (MaxLab's C++ closed-loop API, and direct ZMQ frame streaming from Python). `synapsortrt`'s real-time detector/matcher is built around the ZMQ path.

## `Data/` — training-set convention

Gitignored locally; fetched from the `braingeneers` S3 bucket (public read, `https://s3.braingeneers.gi.ucsc.edu` — no special credentials needed via this endpoint). Each recording gets its own per-recording fetch + curate script pair (see `scripts/fetch_htho_24448a_d63.py` / `curate_htho_24448a_d63.py` for the pattern), and lands at:

```
Data/<tissue>_<line>/<chip>_<tissue>_d<age>_<date>/
├── raw/<stem>.raw.h5
├── sorted_spikedata.pkl            # uncurated
└── sorted_spikedata_curated.pkl    # firing-rate + ISI + min-spike-count curated, with per-unit waveforms attached
```

Tissue type and age are encoded directly in the directory/file naming (e.g. `htho` = human thalamic organoid, `d63` = day 63) — this is the basis for the planned per-unit metadata binning (tissue type, age, etc.) for the `synapsortrt` training set. Currently: one recording (`24448a_htho_d63_061626`, thalamus-only, no stim).

## Where things are documented elsewhere

- Subagents: [agents/README.md](agents/README.md) — currently just `maxwell-hardware`
- Data provenance for the one recording so far: [Data/README.md](Data/README.md)
- Pipeline usage examples: [README.md](README.md)
