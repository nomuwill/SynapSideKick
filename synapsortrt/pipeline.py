"""
pipeline.py — end-to-end SynapSortRT pipeline.

Ties together Detector + Matcher and runs on a recording
(from a file or numpy array).

Usage:
    pipe = SynapsortPipeline.from_recording(rec, library, corr_thresh=0.75)
    spike_trains = pipe.run(duration_s=30.0)
    # spike_trains: dict[unit_id -> np.ndarray of spike times in ms]
"""

from __future__ import annotations
import numpy as np
import time
from typing import Dict, Optional
from .templates import TemplateLibrary
from .detector import Detector
from .matcher import Matcher, MatchResult


class SynapsortPipeline:
    """
    Full detection + matching pipeline.

    Args:
        traces      : (n_channels, n_frames) float32 µV — full recording
        channel_ids : (n_channels,) electrode ids matching template library
        fs          : sampling frequency (Hz)
        library     : TemplateLibrary from Phase 1
        thresh_mult : noise multiplier for threshold detection
        corr_thresh : minimum correlation for a template match
        n_samples   : waveform snippet length (samples)
        refractory_ms : dead time after crossing (ms)
    """

    def __init__(
        self,
        traces: np.ndarray,
        channel_ids: np.ndarray,
        fs: float,
        library: TemplateLibrary,
        thresh_mult: float = 4.0,
        corr_thresh: float = 0.75,
        n_samples: int = 82,
        refractory_ms: float = 1.0,
    ):
        self.traces = traces
        self.channel_ids = channel_ids
        self.fs = fs
        self.library = library

        self.detector = Detector(
            fs=fs,
            thresh_mult=thresh_mult,
            n_samples=n_samples,
            refractory_ms=refractory_ms,
        )
        self.matcher = Matcher(library, corr_thresh=corr_thresh)

        # Estimate thresholds from full trace
        self.detector.set_thresholds(traces)

    def run(
        self,
        chunk_s: float = 1.0,
        verbose: bool = True,
    ) -> Dict[int, np.ndarray]:
        """
        Run detection + matching across the full recording in chunks.

        Args:
            chunk_s : chunk size in seconds (controls memory use, not accuracy)
            verbose : print progress

        Returns:
            spike_trains: dict[unit_id -> spike times in ms]
        """
        from tqdm import tqdm

        chunk_frames = int(chunk_s * self.fs)
        n_frames = self.traces.shape[1]
        n_chunks = int(np.ceil(n_frames / chunk_frames))

        all_results: list[MatchResult] = []
        t0 = time.time()

        for c in tqdm(range(n_chunks), disable=not verbose):
            s = c * chunk_frames
            e = min(s + chunk_frames, n_frames)
            chunk = self.traces[:, s:e]

            events = self.detector.detect(chunk, channel_ids=self.channel_ids)
            # Adjust frames to global time
            for ev in events:
                ev.frame += s

            results = self.matcher.match(events)
            all_results.extend(results)

        elapsed = time.time() - t0
        if verbose:
            matched = sum(1 for r in all_results if r.unit_id is not None)
            print(f"\nDone in {elapsed:.1f}s — {len(all_results)} crossings, "
                  f"{matched} matched to units")

        # Assemble spike trains per unit
        spike_trains: Dict[int, list] = {int(uid): [] for uid in self.library.unit_ids}
        for r in all_results:
            if r.unit_id is not None:
                spike_trains[r.unit_id].append(r.frame / self.fs * 1000.0)

        return {uid: np.array(times) for uid, times in spike_trains.items()}
