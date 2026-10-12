"""GPU re-scoring diagnostics for trained Student/Teacher adapters.

- ``posthoc``: KL, clipped positive-KL fraction, residual-target fit and
  privileged-context JS retention on the training rollouts before a checkpoint.
- ``recurrence``: top-k contribution recurrence R_k between two checkpoints on a
  fixed canonical response set, with a deranged cross-problem baseline.
"""
