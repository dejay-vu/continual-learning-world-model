#!/usr/bin/env python
"""Minimal launcher for Continual-Learning World Models.

The script now delegates **all** heavy-lifting to reusable components:

• Command-line parsing & YAML handling - handled by :class:`clwm.Config`.
• Network instantiation, task scheduling, training - handled by
  :class:`clwm.Trainer`.

Keeping the entry-point slim makes experimentation easier while retaining the
familiar ``python train.py --categories action`` interface.
"""


import torch

# Fast fail when the host lacks CUDA – behaviour unchanged from original
if not torch.cuda.is_available():
    raise SystemExit(
        "CUDA device not available - training requires an NVIDIA GPU."
    )

from clwm import Config, Trainer  # noqa: E402 – after torch check


def main() -> None:
    # The helper returns (cfg, args) but we only need the config object.
    cfg, _ = Config.from_cli()

    # *Unpack* the top-level dictionary directly into the Trainer so that all
    # required sub-sections (model, training, cli…) are available as keyword
    # arguments.
    trainer = Trainer(**cfg.to_dict())

    # Let the Trainer derive the task list from CLI flags and kick-off the
    # full continual-learning loop.
    trainer.train()


if __name__ == "__main__":
    main()
