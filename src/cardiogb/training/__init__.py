"""Training, callbacks, and ensemble orchestration."""

from cardiogb.training.trainer import (
    CrossSectionalTrainer,
    CrossSectionalTransition,
    TrainerConfig,
)

__all__ = ["CrossSectionalTrainer", "CrossSectionalTransition", "TrainerConfig"]
