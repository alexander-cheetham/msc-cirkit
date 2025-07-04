- Clarify the correct computation of negative log-likelihood (NLL) and
  KL divergence in `src/benchmarks.py`. The current implementation assumes
  that circuit outputs are log-likelihoods but this needs to be verified.
- Decide whether benchmarking should track the absolute NLL values or the
  difference between original and Nyström circuits. The implementation now
  logs the mean absolute difference in NLL (`NLL_diff`), but this may still need
  refinement.
