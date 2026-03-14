"""Signal generation for statistical arbitrage and alpha research."""

from .vrp import compute_vrp, VRPSignal
from .pairs import find_cointegrated_pairs, analyze_pair, PairStats
