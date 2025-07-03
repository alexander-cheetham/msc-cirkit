- Clarify the correct computation of negative log-likelihood (NLL) and
  KL divergence in `src/benchmarks.py`. The current implementation assumes
  that circuit outputs are log-likelihoods but this needs to be verified.
- Nyström factor construction now avoids materialising the dense weight matrix.
