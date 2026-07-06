"""
templates.py — Phase 1: load and store unit templates from any sorter output.

A TemplateLibrary holds:
  - templates : (n_units, n_samples)        mean waveform on each unit's best channel
  - channels  : (n_units, n_active_ch)      which channels each unit appears on
  - waveforms : dict[unit_id -> (n_active_ch, n_samples)]  per-channel waveforms
  - fs        : sampling frequency (Hz)

Supported loaders:
  - from_kilosort(output_dir)         — Kilosort2/3/4 phy output folder
  - from_spikedata_pickle(pkl, h5)    — SpikeLab SpikeData pickle + raw .h5
  - from_npz(path)                    — SynapSortRT native format

Save/load:
  - save(path)  →  {path}.npz
  - load(path)  →  TemplateLibrary
"""

from __future__ import annotations
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class TemplateLibrary:
    """Sorter-agnostic store of unit templates."""

    fs: float                                      # sampling frequency (Hz)
    unit_ids: np.ndarray                           # (n_units,) int
    # per-unit best-channel waveform: (n_units, n_samples)
    templates: np.ndarray
    # per-unit active channel indices (list of arrays, variable length)
    active_channels: List[np.ndarray]
    # per-unit per-channel waveforms: unit_id -> (n_active_ch, n_samples)
    waveforms: Dict[int, np.ndarray] = field(default_factory=dict)
    # channel map: index -> electrode id (optional)
    channel_map: Optional[np.ndarray] = None
    # how templates were derived — used for benchmarking
    # "avg_stored"  : from pre-averaged waveforms stored in SpikeData pickle
    # "raw_extracted": re-extracted from raw .h5 and averaged
    # "kilosort"    : from Kilosort phy output
    source: str = "unknown"

    @property
    def n_units(self) -> int:
        return len(self.unit_ids)

    @property
    def n_samples(self) -> int:
        return self.templates.shape[1]

    # ── loaders ───────────────────────────────────────────────────────────────

    @classmethod
    def from_kilosort(
        cls,
        output_dir: str | Path,
        fs: float,
        amp_thresh: float = 0.2,
        n_samples: int = 82,
    ) -> "TemplateLibrary":
        """
        Load templates from a Kilosort phy output folder.

        Args:
            output_dir   : path containing templates.npy, channel_map.npy, etc.
            fs           : sampling frequency in Hz
            amp_thresh   : fraction of peak amplitude — channels above this are
                           considered "active" for a unit (default 0.2 = 20%)
            n_samples    : waveform snippet length in samples (default 82 = ~4ms at 20kHz)
        """
        d = Path(output_dir)
        raw = np.load(d / "templates.npy")          # (n_units, n_t, n_ch)
        channel_map = np.load(d / "channel_map.npy").flatten()

        # Good units only (if cluster_group.tsv exists)
        good_units = None
        cg_path = d / "cluster_group.tsv"
        if cg_path.exists():
            import csv
            good_units = set()
            with open(cg_path) as f:
                reader = csv.DictReader(f, delimiter="\t")
                for row in reader:
                    if row.get("group", "").lower() == "good":
                        good_units.add(int(row["cluster_id"]))

        # cluster_ids
        try:
            cluster_ids = np.load(d / "spike_clusters.npy")
            unit_ids = np.unique(cluster_ids)
        except FileNotFoundError:
            unit_ids = np.arange(raw.shape[0])

        if good_units is not None:
            unit_ids = np.array([u for u in unit_ids if u in good_units])

        templates_list = []
        active_channels_list = []
        waveforms_dict = {}

        for uid in unit_ids:
            tmpl = raw[uid]                         # (n_t, n_ch)
            peak_amp = np.max(np.abs(tmpl), axis=0) # (n_ch,)
            active = np.where(peak_amp >= amp_thresh * peak_amp.max())[0]

            best_ch = int(np.argmax(peak_amp))
            best_wf = tmpl[:, best_ch]

            # Trim/pad to n_samples centred on peak
            peak_t = int(np.argmax(np.abs(best_wf)))
            half = n_samples // 2
            s = peak_t - half
            e = s + n_samples
            pad_l = max(0, -s)
            pad_r = max(0, e - tmpl.shape[0])
            s, e = max(0, s), min(tmpl.shape[0], e)

            def _trim(wf):
                out = np.zeros(n_samples, dtype=np.float32)
                out[pad_l:pad_l + (e - s)] = wf[s:e]
                return out

            templates_list.append(_trim(best_wf))
            active_channels_list.append(channel_map[active])

            wf_per_ch = np.stack([_trim(tmpl[:, ch]) for ch in active])
            waveforms_dict[int(uid)] = wf_per_ch.astype(np.float32)

        return cls(
            fs=float(fs),
            unit_ids=unit_ids,
            templates=np.stack(templates_list).astype(np.float32),
            active_channels=active_channels_list,
            waveforms=waveforms_dict,
            channel_map=channel_map,
        )

    @classmethod
    def from_spikedata_pickle_avg(
        cls,
        pickle_path: str | Path,
        spikelab_src: str | Path,
        amp_thresh: float = 0.2,
        duration_s: Optional[float] = None,
    ) -> "TemplateLibrary":
        """
        Fast loader — uses pre-averaged waveforms already stored in the
        SpikeData pickle's neuron_attributes (template, neighbor_templates).

        No raw .h5 needed. Templates are clean (averaged, low-noise) which
        may reduce matching accuracy on raw data — use for benchmarking
        against from_spikedata_pickle() which re-extracts from raw traces.

        Args:
            pickle_path : path to the SpikeData .pkl file
            spikelab_src: path to SpikeLab src/ directory
            amp_thresh  : fraction of peak amplitude for active channel selection
            duration_s  : if set, only include units with spikes in first N seconds
        """
        import sys
        spikelab_src = str(spikelab_src)
        if spikelab_src not in sys.path:
            sys.path.insert(0, spikelab_src)
        from spikelab.data_loaders import load_spikedata_from_pickle

        sd = load_spikedata_from_pickle(str(pickle_path))
        fs = (float(sd.raw_time) * 1000.0 if np.isscalar(sd.raw_time)
              else float(1.0 / np.mean(np.diff(sd.raw_time)) * 1000.0))

        attrs = sd.neuron_attributes or [{}] * sd.N
        unit_ids, templates_list, active_channels_list, waveforms_dict = [], [], [], {}

        for i, (times_ms, info) in enumerate(zip(sd.train, attrs)):
            if duration_s is not None and len(times_ms) > 0:
                if times_ms.min() > duration_s * 1000.0:
                    continue

            tmpl = info.get("template")
            n_chs = info.get("neighbor_channels")
            n_tmpls = info.get("neighbor_templates")

            if tmpl is None:
                continue

            tmpl = np.asarray(tmpl, dtype=np.float32)
            best_ch = int(info.get("channel", 0))

            if n_chs is not None and n_tmpls is not None:
                n_chs = np.asarray(n_chs)
                n_tmpls = np.asarray(n_tmpls, dtype=np.float32)
                peak_amps = np.max(np.abs(n_tmpls), axis=1)
                if peak_amps.max() > 0:
                    active_mask = peak_amps >= amp_thresh * peak_amps.max()
                else:
                    active_mask = np.ones(len(n_chs), dtype=bool)
                active_chs = n_chs[active_mask]
                active_wfs = n_tmpls[active_mask]
            else:
                active_chs = np.array([best_ch])
                active_wfs = tmpl[None, :]

            unit_ids.append(i)
            templates_list.append(tmpl)
            active_channels_list.append(active_chs)
            waveforms_dict[i] = active_wfs

        return cls(
            fs=fs,
            unit_ids=np.array(unit_ids),
            templates=np.stack(templates_list),
            active_channels=active_channels_list,
            waveforms=waveforms_dict,
            source="avg_stored",
        )

    @classmethod
    def from_spikedata_pickle(
        cls,
        pickle_path: str | Path,
        spikelab_src: str | Path,
        raw_h5_path: str | Path,
        amp_thresh: float = 0.2,
        duration_s: Optional[float] = None,
        ms_before: float = 1.0,
        ms_after: float = 2.0,
    ) -> "TemplateLibrary":
        """
        Load templates from a SpikeLab SpikeData pickle, extracting raw
        waveforms from the matching .h5 recording so templates have realistic
        noise characteristics.

        Uses SpikeLab's waveform extraction and recentering pipeline.

        Args:
            pickle_path  : path to the SpikeData .pkl file
            spikelab_src : path to SpikeLab src/ directory
            raw_h5_path  : path to the matching Maxwell .h5 recording
            amp_thresh   : fraction of peak amplitude for active channel selection
            duration_s   : if set, only use spikes in the first N seconds
            ms_before    : waveform window before spike (ms)
            ms_after     : waveform window after spike (ms)
        """
        import sys
        spikelab_src = str(spikelab_src)
        if spikelab_src not in sys.path:
            sys.path.insert(0, spikelab_src)

        from spikelab.data_loaders import load_spikedata_from_pickle
        from spikelab.spikedata.utils import extract_waveforms
        from spikelab.spike_sorting.waveform_utils import (
            center_spike_times, classify_polarity, get_max_channels,
            compute_half_window_sizes,
        )
        import spikeinterface.extractors as se

        # ── load SpikeData ────────────────────────────────────────────────
        sd = load_spikedata_from_pickle(str(pickle_path))
        fs = float(sd.raw_time) * 1000.0 if np.isscalar(sd.raw_time) else float(
            1.0 / np.mean(np.diff(sd.raw_time)) * 1000.0
        )

        # ── load raw recording ────────────────────────────────────────────
        rec = se.MaxwellRecordingExtractor(str(raw_h5_path))
        fs_rec = rec.get_sampling_frequency()
        n_frames = min(
            int(duration_s * fs_rec) if duration_s else rec.get_num_frames(),
            rec.get_num_frames(),
        )
        gains = rec.get_property("gain_to_uV")
        ch_ids = rec.get_channel_ids()

        print(f"  Loading raw traces ({len(ch_ids)} ch, {n_frames/fs_rec:.1f}s) ...")
        raw = rec.get_traces(
            start_frame=0, end_frame=n_frames,
            channel_ids=ch_ids, return_in_uV=False,
        ).astype(np.float32)
        raw_uv = (raw * gains[None, :]).T   # (n_channels, n_frames)

        fs_kHz = fs_rec / 1000.0

        # ── build spike_times_by_unit dict (ms → samples filter) ─────────
        n_units = sd.N
        attrs = sd.neuron_attributes or [{}] * n_units

        # Gather templates for recentering
        raw_templates = []
        for info in attrs:
            tmpl = info.get("template")
            raw_templates.append(np.asarray(tmpl, dtype=np.float32) if tmpl is not None
                                 else np.zeros(100, dtype=np.float32))
        raw_templates_arr = np.stack(raw_templates)   # (n_units, n_samples)

        use_pos_peak = classify_polarity(raw_templates_arr)
        chans_max = get_max_channels(raw_templates_arr, use_pos_peak)
        half_windows = compute_half_window_sizes(raw_templates_arr, chans_max)

        # spike times in ms → filter to duration
        spike_times_by_unit = {}
        for i, times_ms in enumerate(sd.train):
            if duration_s is not None:
                times_ms = times_ms[times_ms <= duration_s * 1000.0]
            if len(times_ms) == 0:
                continue
            spike_times_by_unit[i] = times_ms

        print(f"  Recentering spike times for {len(spike_times_by_unit)} units ...")
        centered = center_spike_times(
            recording=rec,
            spike_times_by_unit=spike_times_by_unit,
            chans_max=chans_max,
            use_pos_peak=use_pos_peak,
            half_window_sizes=half_windows,
        )

        # ── extract raw waveforms per unit ────────────────────────────────
        unit_ids = []
        templates_list = []
        active_channels_list = []
        waveforms_dict = {}

        print(f"  Extracting waveforms ...")
        for i, times_ms in centered.items():
            if len(times_ms) == 0:
                continue

            info = attrs[i]
            n_chs = info.get("neighbor_channels")
            ch_indices = np.asarray(n_chs) if n_chs is not None else np.array([int(info.get("channel", 0))])

            # extract_waveforms returns (n_channels, n_samples, n_spikes)
            wfs_raw = extract_waveforms(
                raw_data=raw_uv,
                spike_times_ms=times_ms,
                fs_kHz=fs_kHz,
                ms_before=ms_before,
                ms_after=ms_after,
                channel_indices=ch_indices,
            )   # (n_active_ch, n_samples, n_spikes)

            mean_wfs = wfs_raw.mean(axis=2).astype(np.float32)   # (n_active_ch, n_samples)

            # Active channel selection by amplitude
            peak_amps = np.max(np.abs(mean_wfs), axis=1)
            if peak_amps.max() > 0:
                active_mask = peak_amps >= amp_thresh * peak_amps.max()
            else:
                active_mask = np.ones(len(ch_indices), dtype=bool)

            active_chs = ch_indices[active_mask]
            active_wfs = mean_wfs[active_mask]

            # Best channel template
            best_local = int(np.argmax(peak_amps[active_mask]))
            best_tmpl = active_wfs[best_local]

            unit_ids.append(i)
            templates_list.append(best_tmpl)
            active_channels_list.append(active_chs)
            waveforms_dict[i] = active_wfs

        return cls(
            fs=fs_rec,
            unit_ids=np.array(unit_ids),
            templates=np.stack(templates_list),
            active_channels=active_channels_list,
            waveforms=waveforms_dict,
            source="raw_extracted",
        )

    @classmethod
    def from_npz(cls, path: str | Path) -> "TemplateLibrary":
        """Load a previously saved TemplateLibrary."""
        p = Path(path)
        if not p.suffix:
            p = p.with_suffix(".npz")
        d = np.load(p, allow_pickle=True)
        active_channels = list(d["active_channels"])
        waveforms = d["waveforms"].item() if "waveforms" in d else {}
        channel_map = d["channel_map"] if "channel_map" in d else None
        source = str(d["source"]) if "source" in d else "unknown"
        return cls(
            fs=float(d["fs"]),
            unit_ids=d["unit_ids"],
            templates=d["templates"],
            active_channels=active_channels,
            waveforms=waveforms,
            channel_map=channel_map,
            source=source,
        )

    # ── save ──────────────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> Path:
        """Save to {path}.npz."""
        p = Path(path)
        if not p.suffix:
            p = p.with_suffix(".npz")
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            p,
            fs=self.fs,
            unit_ids=self.unit_ids,
            templates=self.templates,
            active_channels=np.array(self.active_channels, dtype=object),
            waveforms=self.waveforms,
            channel_map=self.channel_map if self.channel_map is not None else np.array([]),
            source=np.array(self.source),
        )
        return p

    def __repr__(self) -> str:
        return (
            f"TemplateLibrary(n_units={self.n_units}, "
            f"n_samples={self.n_samples}, fs={self.fs:.0f} Hz, "
            f"source='{self.source}')"
        )
