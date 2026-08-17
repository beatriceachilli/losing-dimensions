"""
run_options.py
=============================================================================
Extra training knobs not covered by parse_utils.py's original schema
(DataOptions / TrainOptions / ModelOptions).

This lives in its own module — rather than being defined inside train.py —
so that a checkpoint's pickled `args.run` (a RunOptions instance) can be
unpickled correctly from *any* script, not just from train.py itself.

Background: when you run `python train.py`, train.py becomes Python's
`__main__` module. If a dataclass is defined directly in train.py, torch.save
pickles instances of it as `__main__.RunOptions`. Loading that checkpoint
from a different script (e.g. run_svd.py, whose own __main__ is run_svd.py)
then fails with `AttributeError: Can't get attribute 'RunOptions' on
<module '__main__' ...>`, because pickle looks for the class inside
whichever script happens to be __main__ at load time, not the one that
saved it. Defining it in a real, importable module (this file) sidesteps
the problem entirely, the same way DataOptions/TrainOptions/ModelOptions
already avoid it by living in parse_utils.py.
=============================================================================
"""

from dataclasses import dataclass, field as dc_field


@dataclass
class RunOptions:
    beta_start: float = dc_field(default=1e-4)  # beta_min in Table 1
    beta_end: float = dc_field(default=2e-2)  # beta_max in Table 1
    resume: str = dc_field(default="")  # path to a checkpoint to resume from
    ema_update_every: int = dc_field(default=1)
    seed: int = dc_field(default=0)
    keep_milestone_every: int = dc_field(default=0)
