import argparse

from absl import app

from diploma_code.config import (
    default_attn_ctc_model_config, default_diploma_config, DEFAULT_RUN_NAME
)
from diploma_code.trainer import (
    LTRTrainer
)

from ml_collections import config_flags

_CONFIG = config_flags.DEFINE_config_file('config')

def main(_):
    trainer = LTRTrainer(_CONFIG.value)
    if _CONFIG.value.wandb.run_name == DEFAULT_RUN_NAME:
        raise ValueError("--config.wandb.run_name is a required arg")
    trainer.train(_CONFIG.value.wandb.run_name)

if __name__ == "__main__":
    app.run(main)
    