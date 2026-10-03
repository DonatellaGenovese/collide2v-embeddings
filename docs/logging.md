# Reading a run: MLflow, wandb, and sweeps

A training writes numbers. This is how to look at them, on lxplus and on your own
machine, and what to look at for each kind of model. Section 7 of the README is the
summary; this is the walk-through.

## Choosing a logger

`configs/logger/` holds one file per backend, and `logger=<name>` picks it:

```bash
python src/train.py experiment=<name>                  # mlflow, the default
python src/train.py experiment=<name> logger=wandb
python src/train.py experiment=<name> logger=csv       # a file, nothing to serve
python src/train.py experiment=<name> logger=many_loggers   # mlflow and wandb at once
```

Which to use is a practical choice, not a matter of taste:

- **MLflow** writes to the filesystem, under `logs/mlflow/mlruns`. Nothing leaves the
  machine, and nothing needs an account, which is why it is the default at CERN. The
  cost is that looking at it from your laptop means a tunnel.
- **wandb** writes to a server, so a run started on a batch node is visible from
  anywhere, including from a phone while a 20-hour training is going. The cost is an
  account, a key, and a run's metadata leaving CERN. Check that this is acceptable for
  what you are doing before using it on real work.
- **CSV** is for when you want the numbers in a file and will plot them yourself.

## MLflow on your own machine

```bash
cd logs/mlflow
mlflow ui                      # then open http://127.0.0.1:5000
```

`tracking_uri` in `configs/logger/mlflow.yaml` is `${paths.log_dir}/mlflow/mlruns`, so
the runs sit next to the repository unless you moved `log_dir`.

## MLflow on lxplus, from your laptop

The UI is a web server, and lxplus does not expose ports to the outside, so forward one
over SSH. In a terminal on your laptop:

```bash
ssh -L 5000:localhost:5000 <username>@lxplus.cern.ch
```

Then, in that same session, on lxplus:

```bash
cd ~/collide2v-embeddings/logs/mlflow
mlflow ui --port 5000
```

Now http://localhost:5000 in your laptop's browser is the UI running at CERN. Two things
that go wrong:

- **"Address already in use"**, on either side: someone else on that login node took the
  port, or you left a tunnel open. Pick another, `--port 5001` and `-L 5001:localhost:5001`.
- **An empty UI**: you are in the wrong directory. `mlflow ui` serves `./mlruns` from
  wherever it is started, so it has to be `logs/mlflow`.

If the dataset and the logs live on EOS rather than in the repository, point the server
at them instead: `mlflow ui --backend-store-uri file:///eos/user/<l>/<user>/.../mlruns`.

## wandb, including on batch nodes

Once per machine:

```bash
pip install wandb      # not in requirements.txt; mlflow is
wandb login            # paste the key from https://wandb.ai/authorize
```

Set the project before the first run, in `configs/logger/wandb.yaml` or on the command
line, or everything lands in a project named after the template this repository came
from:

```bash
python src/train.py experiment=<name> logger=wandb logger.wandb.project=collide2v
```

A batch node may have no outbound network. Log offline, then push the run afterwards
from the login node:

```bash
python src/train.py experiment=<name> logger=wandb logger.wandb.offline=true
wandb sync logs/train/runs/<date>/wandb/offline-run-*
```

## What to look at

For a **classifier**, the metrics are the obvious ones, and the names matter because the
callbacks refer to them:

| Metric | Where it comes from |
| --- | --- |
| `train/loss`, `val/loss` | the model |
| `train/acc`, `val/acc` | the model; `val/acc` is what the checkpoint callback keeps and what early stopping watches |
| `val/acc_best` | the best epoch so far, and what two of the sweeps optimise |
| `val/auroc` | the model, macro over classes |
| `val/class_<name>_auc`, `val/mean_auc` | the `multiclassROC` callback, per process. It reads `probs` from what `validation_step` returns, so a model that returns nothing gets a warning and no AUROC — see section 6.4 of the README |

For a **contrastive encoder** there is no accuracy to speak of, and this is the part
that trips people up. Watch:

- `train/contrastive_loss` and `val/contrastive_loss`. Falling is necessary and not
  sufficient: a collapsed embedding, where every event maps to nearly the same point,
  also has a low loss.
- `val/acc`, from the small diagnostic classification head, if it is on. It is a cheap
  proxy for whether anything class-like is in the embedding.
- The real answer comes afterwards, from `src/eval_probes.py`: freeze the encoder, fit a
  linear probe, compare against a classifier trained end to end on the same data. A
  training curve cannot tell you that.

With `use_classification_head: false` there is no `val/acc` at all, and the default
callbacks monitor it — so checkpointing and early stopping have nothing to go on. Point
them at the contrastive loss instead:

```bash
python src/train.py experiment=<name> model=simclr \
    model.use_classification_head=false \
    callbacks.model_checkpoint.monitor=val/contrastive_loss \
    callbacks.model_checkpoint.mode=min \
    callbacks.early_stopping.monitor=val/contrastive_loss \
    callbacks.early_stopping.mode=min
```

## Hyperparameter sweeps

Optuna is wired in through Hydra's sweeper, and `configs/hparams_search/` holds the
searches. `-m` is what makes it a sweep rather than a single run:

```bash
python src/train.py -m hparams_search=collide2v_optuna_multiclass experiment=<name>
```

Each config says what it optimises (`optimized_metric`, which must be a metric the model
or a callback actually logs), in which direction, how many trials, and which parameters
to vary. Results go to a SQLite file, `logs/optuna_logs/optuna.db` by default, so a sweep
survives being interrupted: rerun the same `study_name` against the same storage and it
carries on.

To read one, `notebooks/optuna_sweep_results.ipynb`, which plots the trials and the
importance of each parameter.

A sweep on the full dataset is long. Two things to do first: run a single trial of the
same config to check it trains at all, and make sure `optimized_metric` appears in the
logs of that run. A sweep optimising a metric nobody logs is hours of work for nothing.

## Where a run's files are

```
logs/
├── train/runs/<date>/          one directory per run
│   ├── .hydra/config.yaml      the fully composed config: what actually ran
│   ├── checkpoints/            weights, including last.ckpt
│   ├── tensorboard/, csv/      if those loggers were on
│   └── probe_results.json      if eval_probes.py was run with this output directory
├── mlflow/mlruns/              the MLflow store
├── optuna_logs/optuna.db       sweep history
└── condor_logs/                stdout and stderr of batch jobs
```

The composed config in `.hydra/config.yaml` is the most useful file there: it is the
whole description of the run, including the dataset label, the seed and the feature set,
and it is what to read when a result needs to be reproduced six months later.
