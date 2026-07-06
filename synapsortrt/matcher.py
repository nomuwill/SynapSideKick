"""
matcher.py — Phase 2b: GPU-accelerated cosine similarity template matching.

Per-channel template matrices are pre-loaded onto MPS (or CPU) as normalized
torch tensors at startup. Each threshold crossing triggers a single
torch.mv() call — one matrix-vector multiply regardless of unit count.

Usage:
    m = Matcher(library, corr_thresh=0.75)
    results = m.match(events)
    # results: list of MatchResult (unit_id=None if no match / noise)
"""

from __future__ import annotations
import numpy as np
import torch
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
from .templates import TemplateLibrary
from .detector import CrossingEvent


def _get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@dataclass
class MatchResult:
    channel: int
    frame: int
    unit_id: Optional[int]   # None = noise / no match
    correlation: float        # best cosine similarity score (0–1)
    waveform: np.ndarray      # (n_samples,) µV


class Matcher:
    """
    GPU-accelerated cosine similarity template matcher.

    At init, builds a per-channel dict of pre-normalized torch tensors:
        _ch_templates[ch] : (n_units_on_ch, n_samples)  — on device
        _ch_unit_ids[ch]  : list[int]                   — unit IDs in row order

    Each match() call sends the waveform snippet to the device, normalizes it,
    and does a single matrix-vector multiply per active channel.

    Args:
        library     : TemplateLibrary from Phase 1
        corr_thresh : minimum cosine similarity to accept a match (0–1)
        device      : "mps", "cuda", or "cpu" (auto-detected if None)
    """

    def __init__(
        self,
        library: TemplateLibrary,
        corr_thresh: float = 0.75,
        device: Optional[str] = None,
    ):
        self.library = library
        self.corr_thresh = corr_thresh
        self.device = device or _get_device()

        # Build per-channel template matrices on device
        # _ch_templates[ch] : (n_units_on_ch, n_samples) float32 on device
        # _ch_unit_ids[ch]  : list of unit_ids matching row order
        self._ch_templates: Dict[int, torch.Tensor] = {}
        self._ch_unit_ids:  Dict[int, List[int]]    = {}

        # Accumulate per channel
        ch_rows: Dict[int, List[Tuple[int, np.ndarray]]] = {}
        for u_idx, (uid, active_chs) in enumerate(
            zip(library.unit_ids, library.active_channels)
        ):
            wfs = library.waveforms.get(int(uid))
            for k, ch in enumerate(active_chs):
                ch = int(ch)
                tmpl = wfs[k] if wfs is not None else library.templates[u_idx]
                norm = np.linalg.norm(tmpl)
                if norm == 0:
                    continue
                ch_rows.setdefault(ch, []).append((int(uid), tmpl / norm))

        # Stack into tensors and move to device
        for ch, rows in ch_rows.items():
            uid_list, tmpl_list = zip(*rows)
            mat = torch.tensor(
                np.stack(tmpl_list).astype(np.float32),
                dtype=torch.float32,
                device=self.device,
            )  # (n_units_on_ch, n_samples)
            self._ch_templates[ch] = mat
            self._ch_unit_ids[ch] = list(uid_list)

        n_ch = len(self._ch_templates)
        avg_units = (sum(t.shape[0] for t in self._ch_templates.values()) / n_ch
                     if n_ch > 0 else 0)
        print(f"Matcher: {n_ch} active channels, "
              f"{avg_units:.1f} avg units/channel, device={self.device}")

    def match(self, events: List[CrossingEvent]) -> List[MatchResult]:
        """
        Match a list of CrossingEvents against channel templates on the GPU.

        For efficiency, events on the same channel are batched into a single
        matrix multiply: (n_events_on_ch, n_samples) @ (n_samples, n_units_on_ch)

        Returns a MatchResult for every event (unit_id=None if noise/no match).
        """
        if not events:
            return []

        # Group events by channel
        by_channel: Dict[int, List[CrossingEvent]] = {}
        for ev in events:
            by_channel.setdefault(ev.channel, []).append(ev)

        results_map: Dict[int, MatchResult] = {}  # ev index → result

        # Global event index for stable ordering
        ev_index = {id(ev): i for i, ev in enumerate(events)}
        placeholder = [None] * len(events)

        for ch, ch_events in by_channel.items():
            tmpl_mat = self._ch_templates.get(ch)   # (n_units, n_samples) or None

            for ev in ch_events:
                idx = ev_index[id(ev)]

                if tmpl_mat is None:
                    # No templates on this channel — noise by default
                    placeholder[idx] = MatchResult(
                        channel=ev.channel, frame=ev.frame,
                        unit_id=None, correlation=-1.0, waveform=ev.waveform,
                    )
                    continue

                wf = ev.waveform.astype(np.float32)
                norm = float(np.linalg.norm(wf))
                if norm == 0:
                    placeholder[idx] = MatchResult(
                        channel=ev.channel, frame=ev.frame,
                        unit_id=None, correlation=-1.0, waveform=ev.waveform,
                    )
                    continue

                # Normalize and send to device
                wf_t = torch.tensor(wf / norm, dtype=torch.float32,
                                    device=self.device)

                # Cosine sim: (n_units,) — templates already normalized
                n = min(wf_t.shape[0], tmpl_mat.shape[1])
                scores = tmpl_mat[:, :n] @ wf_t[:n]   # (n_units,)

                best_idx = int(torch.argmax(scores).item())
                best_corr = float(scores[best_idx].item())
                best_uid = self._ch_unit_ids[ch][best_idx] if best_corr >= self.corr_thresh else None

                placeholder[idx] = MatchResult(
                    channel=ev.channel,
                    frame=ev.frame,
                    unit_id=best_uid,
                    correlation=best_corr,
                    waveform=ev.waveform,
                )

        return placeholder
