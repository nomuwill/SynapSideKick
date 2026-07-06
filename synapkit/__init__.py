"""
SynapKit — Modular real-time sorter-agnostic spike sorting and stimulation suite.

Subpackages:
  synapsortrt  : real-time spike sorting via template matching
  synapremoval : stimulation artifact detection and removal
  synapanalysis: spike train analysis pipelines
  synapconnect : hardware and pre-sorting method connectors
"""
try:
    from synapsortrt import *
except ImportError:
    pass
