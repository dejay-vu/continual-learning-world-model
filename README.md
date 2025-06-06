# continual-learning-world-model

This project is licensed under the [MIT License](LICENSE).

## Offline dataset

Use `collect_offline_dataset.py` or the helper in `clwm.data` to gather observations, actions and rewards into `.npz` shards. You can optionally provide a pre-trained PPO policy to obtain higher quality behaviour:

```bash
python collect_offline_dataset.py --game Breakout --steps 10000 --out offline_data
```

Configuration for model size, hyper‑parameters and the list of training tasks is stored in `config.yaml`. Running `python train.py` will automatically collect a dataset for each game in the task list (saving shards under `offline_data/<game>/`) when missing and then start training end to end:

```bash
python train.py
```
