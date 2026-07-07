import os

from .diffusion import Trainer as DiffusionTrainer
from .gan import Trainer as GANTrainer
from .ode import Trainer as ODETrainer

# DISTILL_THREE_GROUP=1 selects the flat three-group DMD trainer (SD's proven placement:
# teacher/student/critic each on their own tp-rank group, cross-group via broadcast) —
# the fix for the co-residency OOM. Unset -> the original single-DMD-module trainer.
if os.environ.get("DISTILL_THREE_GROUP", "").strip() in ("1", "true", "True"):
    from .distillation_3group import Trainer as ScoreDistillationTrainer
else:
    from .distillation import Trainer as ScoreDistillationTrainer

__all__ = [
    "DiffusionTrainer",
    "GANTrainer",
    "ODETrainer",
    "ScoreDistillationTrainer"
]
