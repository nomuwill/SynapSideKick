"""
detector.py — Phase 2a: threshold crossing + waveform extraction.

For each channel, detects threshold crossings and extracts a waveform
snippet centred on the trough. Designed to process chunks of raw data
(e.g. 30s blocks or live stream buffers).

Usage:
    det = Detector(fs=20000, thresh_mult=4.0, n_samples=82, refractory_ms=1.0)
    det.set_thresholds(baseline_traces)   # estimate noise from baseline
    crossings = det.detect(traces)        # (n_events,) list of CrossingEvent
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class CrossingEvent:
    channel: int
    frame: int          # sample index of trough
    waveform: np.ndarray  # (n_samples,) float32 in µV


class Detector:
    """
    Multi-channel threshold crossing detector.

    Args:
        fs            : sampling frequency (Hz)
        thresh_mult   : threshold = thresh_mult × median-based noise estimate
        n_samples     : waveform snippet length (samples)
        refractory_ms : dead time after a crossing (ms)
        neg_only      : if True, only detect negative crossings (default True)
    """

    def __init__(
        self,
        fs: float,
        thresh_mult: float = 4.0,
        n_samples: int = 82,
        refractory_ms: float = 1.0,
        neg_only: bool = True,
    ):
        self.fs = fs
        self.thresh_mult = thresh_mult
        self.n_samples = n_samples
        self.half = n_samples // 2
        self.refractory = int(refractory_ms * fs / 1000)
        self.neg_only = neg_only
        self.thresholds: Optional[np.ndarray] = None   # (n_channels,)

    def set_thresholds(self, traces: np.ndarray) -> np.ndarray:
        """
        Estimate per-channel noise from traces and set thresholds.

        Args:
            traces: (n_channels, n_frames) float32 µV
        Returns:
            thresholds: (n_channels,) — stored internally and returned
        """
        # Robust noise estimate: median absolute deviation / 0.6745
        noise = np.median(np.abs(traces), axis=1) / 0.6745
        self.thresholds = self.thresh_mult * noise
        return self.thresholds

    def detect(
        self,
        traces: np.ndarray,
        channel_ids: Optional[np.ndarray] = None,
    ) -> List[CrossingEvent]:
        """
        Detect threshold crossings in traces.

        Args:
            traces      : (n_channels, n_frames) float32 µV
            channel_ids : optional mapping from row index → electrode id

        Returns:
            List of CrossingEvent, sorted by frame.
        """
        if self.thresholds is None:
            raise RuntimeError("Call set_thresholds() before detect().")

        n_ch, n_frames = traces.shape
        events: List[CrossingEvent] = []

        for ch in range(n_ch):
            thr = self.thresholds[ch]
            sig = traces[ch]
            ch_id = int(channel_ids[ch]) if channel_ids is not None else ch

            # Threshold crossing indices
            if self.neg_only:
                crossed = np.where(sig < -thr)[0]
            else:
                crossed = np.where(np.abs(sig) > thr)[0]

            if len(crossed) == 0:
                continue

            # Apply refractory period — keep only first crossing in each burst
            last = -self.refractory - 1
            for idx in crossed:
                if idx - last < self.refractory:
                    continue
                last = idx

                # Refine to trough within ±half window
                s = max(0, idx - self.half)
                e = min(n_frames, idx + self.half)
                if e - s < 3:
                    continue
                trough = s + int(np.argmin(sig[s:e]))

                # Extract snippet centred on trough
                ws = trough - self.half
                we = ws + self.n_samples
                wf = np.zeros(self.n_samples, dtype=np.float32)
                src_s = max(0, ws)
                src_e = min(n_frames, we)
                dst_s = src_s - ws
                wf[dst_s:dst_s + (src_e - src_s)] = sig[src_s:src_e]
                wf -= np.median(wf)

                events.append(CrossingEvent(channel=ch_id, frame=trough, waveform=wf))

        events.sort(key=lambda e: e.frame)
        return events
