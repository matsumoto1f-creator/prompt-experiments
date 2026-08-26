"""Statistics for prompt experiments.

The whole module exists to answer one question honestly: is variant B actually
better, or did we look at the numbers enough times that one of them eventually
drifted past a threshold?

Split by what is being measured, because the right test depends on it:

  proportions.py  binary outcomes (did the response clear a quality bar)
  continuous.py   real-valued outcomes (latency, cost, judge score)
  sequential.py   WHEN it is legitimate to look — the part most platforms get wrong
  power.py        how many samples the question needs before it can be answered
"""

from prompt_experiments.stats.continuous import mann_whitney, welch_t
from prompt_experiments.stats.power import required_n_proportions, mde_at_n
from prompt_experiments.stats.proportions import two_proportion_test, wilson_interval
from prompt_experiments.stats.sequential import (
    SequentialPlan,
    obf_critical_value,
    naive_peeking_error_rate,
)

__all__ = [
    "wilson_interval", "two_proportion_test",
    "welch_t", "mann_whitney",
    "SequentialPlan", "obf_critical_value", "naive_peeking_error_rate",
    "required_n_proportions", "mde_at_n",
]
