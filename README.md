# continual-learning-world-model

This project is licensed under the [MIT License](LICENSE).

Configuration for model size, hyper‑parameters and the list of training tasks is stored in `config.yaml`. Running `python train.py` will automatically collect a dataset for each game in the task list (saving shards under `offline_data/<game>/`) when missing and then start training end to end:

```bash
python train.py
```
