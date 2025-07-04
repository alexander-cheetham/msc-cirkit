- Clarify the correct computation of negative log-likelihood (NLL) and
  KL divergence in `src/benchmarks.py`. The current implementation assumes
  that circuit outputs are log-likelihoods but this needs to be verified.
- Decide whether benchmarking should track the absolute NLL values or the
  difference between original and Nyström circuits. The implementation now
  logs the mean absolute difference in NLL, but this may still need
  refinement.
- Avoid materialising the dense weight matrix when constructing Nyström
  factors in `nystromlayer.py`. The current helper `_build_factors_from`
  still loads the full matrix into memory.
