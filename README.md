# S<sub>y</sub>napKit

**Modular real-time sorter-agnostic spike sorting and stimulation**

TLDR: You can use whichever spike sorter to get a baseline, then real-time
  detect spikes on channels! This enables 1000x faster iteration though
  stimulation params, and enables closed loop experiments. 


## Subpackages

| Package | Description |
|---|---|
| `synapsortrt` | Real-time spike sorting via template matching |
| `synapremoval` | Stimulation artifact detection and removal |
| `synapanalysis` | Spike train analysis pipelines |
| `synapconnect` | Connectors for hardware systems and pre-sorting methods |

## Pipeline

**Phase 1** — Extract unit templates from any sorter output (SpikeData pickle)
```python
from synapsortrt.templates import TemplateLibrary
lib = TemplateLibrary.from_spikedata_pickle_avg(pickle_path, spikelab_src)
lib.save("templates")
```

**Phase 2** — Real-time detection + similarity matching
```python
from synapsortrt.pipeline import SynapsortPipeline
pipe = SynapsortPipeline(traces, channel_ids, fs, library=lib)
spike_trains = pipe.run()
```

## Scripts

```bash
# Phase 1: extract templates from SpikeData pickle
python scripts/extract_templates.py --pickle data.pkl --h5 rec.raw.h5 --spikelab-src /path/to/SpikeLab/src

# Phase 2: run sorting
python scripts/run_sorting.py --recording rec.raw.h5 --templates templates.npz

# Benchmark avg_stored vs raw_extracted templates
python scripts/benchmark_templates.py --pickle data.pkl --h5 rec.raw.h5 --spikelab-src /path/to/SpikeLab/src
```
