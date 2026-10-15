<div align="center">

# COLLIDE-2V embeddings

**Vectorise the COLLIDE-2V dataset, train classifiers and construct neural embedding on it.**

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>

</div>

## 1. What this repository is

COLLIDE-2V is a large simulated dataset of proton-proton collisions. Each event is one collision, described by the objects reconstructed from it: jets, electrons, muons, photons, missing transverse energy, and more. The dataset is stored as Parquet files, one folder per physics process.

It exists in two versions: a private one on EOS, at CERN, and a public one on Hugging Face. They are not interchangeable (different events, different columns). Section 3 gives the details.

Neural networks do not read Parquet. They read fixed-length vectors. Most of the work in this repository is therefore not the training itself but everything around it: turning variable-length lists of particles into vectors of a fixed size, normalising them, and doing so reproducibly over hundreds of gigabytes.

The pipeline has three stages, and you can run them separately:

```
Parquet files          .npy shards            .npy shards           model
(one per process)  →   (raw vectors)      →   (normalised)      →   weights
                       vectorisation          preprocessing         training
```

1. **Vectorisation** reads the Parquet columns you asked for and writes one flat vector per event. Variable-length lists become a fixed number of slots: the *k* highest-p<sub>T</sub> jets, the *k* highest-p<sub>T</sub> muons, and so on, padded with zeros when an event has fewer objects.
2. **Preprocessing** transforms and normalises those vectors, for example `log1p` on momenta and (sin, cos) on angles, and writes the statistics it used to a file so the same scaling can be reapplied later.
3. **Training** streams the normalised shards into a model.

### What you can do with it

- **Supervised classification**, inherited from the original repository: a small MLP and a small transformer that classify events by physics process.
- **Contrastive learning** (in progress): two worked examples, SimCLR, which uses no labels at all, and SupCon, which uses them only to decide which events should end up close together.

### Where the code comes from

This repository starts as a copy of [pploner/foundation_model_testing](https://github.com/pploner/foundation_model_testing) at commit `13448e6`, written by Philip Ploner. The entire commit history up to that point is his work, and the vectorisation, preprocessing and classification code described below is his design.

It follows the [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template): PyTorch Lightning for the training loop, Hydra for the configuration. If a config file or a directory looks unfamiliar, that template's README explains the convention.

### Map of the repository

| Path | What is inside |
| --- | --- |
| `configs/` | Every parameter of every stage, as Hydra `.yaml` files. Start here. |
| `src/data/` | Vectorisation, the split manifest, the streaming dataset, the Lightning datamodule. |
| `src/preprocessing/` | Transforms, normalisers, and the pipeline that fits and applies them. |
| `src/models/` | One file per model, each a Lightning module. |
| `src/train.py`, `src/eval.py` | Entry points for training and for evaluating a checkpoint. |
| `scripts/` | Batch submission to HTCondor, plotting, data inspection. |
| `src/utils/nEvents_scan/` | The event-count scan of the dataset and the script that regenerates it. |
| `tests/` | Pytest suite. |
| `notebooks/` | Exploration and result plots. |

### Status

| Part | State |
| --- | --- |
| Vectorisation and preprocessing | Working, locally and on HTCondor. What stops a run, and why, is in section 4.4. |
| tinyMLP, tinyTransformer classifiers | Working. See section 6.2. |
| Reading the dataset from EOS | Working. |
| Reading a Hugging Face download | Supported: `scripts/download_collide_subset.py` fetches a subset and scans its event counts, and the default feature set is the one both copies have. See section 3. |
| SimCLR and SupCon, augmentations, linear probes | Working. One module, the config picks the objective; the loss is checked against the published implementations. See section 6. |
| Reproducible file and event selection | Working, and tested without EOS. See section 5. |

## 2. Installation

You will run this code in one of two places, and the setup differs.

- **On lxplus**, inside an Apptainer container. This is where the full dataset lives, and where you submit batch jobs.
- **Anywhere else**: your laptop, a group workstation, a university cluster or a rented GPU machine, in a virtual environment, on a subset of the data downloaded from Hugging Face.

The code is the same in both cases. What changes is where the data sits, where the outputs go, and whether there is a GPU.

### On lxplus

```bash
# 1. Clone, on AFS. Do not clone onto /eos: batch jobs cannot execute files there.
cd /afs/cern.ch/user/<l>/<username>
git clone https://github.com/DonatellaGenovese/collide2v-embeddings.git
cd collide2v-embeddings

# 2. Say where the container image will be, for every terminal from now on. Batch
#    jobs read it from the terminal you submit from, so it goes in ~/.bashrc, not
#    only in this shell.
echo 'export FM_TESTING_IMAGE=/eos/user/<l>/<username>/fm_testing.sif' >> ~/.bashrc
source ~/.bashrc
echo $FM_TESTING_IMAGE        # the path, not an empty line: check it before building

# 3. Build the container. It installs everything in requirements.txt.
#    Expect about half an hour and a .sif file of about 7 GB, which does not fit in
#    your AFS home (10 GB quota). Put it on your EOS area, and the build cache on
#    the node's local /tmp. Apptainer does not create the cache directory itself.
export APPTAINER_CACHEDIR=/tmp/<username>/.apptainer
mkdir -p $APPTAINER_CACHEDIR
apptainer build $FM_TESTING_IMAGE fm_testing.def

# 4. Check that it works: the packages import, and EOS can be read from inside.
apptainer exec --bind /eos:/eos --bind /afs:/afs --bind /run/user/$(id -u) $FM_TESTING_IMAGE \
    bash -c 'python -c "import torch, lightning, hydra" && ls /eos/project/f/foundational-model-dataset/samples/production_final > /dev/null && echo ok'
```

The three binds are all needed. The container sees neither `/eos` nor `/afs` unless they are bound, and EOS lets you in only with your Kerberos ticket, which lxplus keeps in `/run/user/<your uid>`: without that bind every read from EOS fails with `Permission denied`. The check above reads EOS for that reason; an import alone would pass.

`<username>` is your CERN login and `<l>` its first letter. Step 2 matters beyond the build: every command that uses the container reads `FM_TESTING_IMAGE`, and a batch job submitted from a terminal without it waits in the queue and then stops at once, with `no container image at …` in its `.err`. A mistyped path fails the build only at the very end, after the whole installation, which is why step 2 prints it.

From then on, on lxplus, every command in this README that starts with `python` runs inside the container, from the repository root:

```bash
apptainer exec --bind /eos:/eos --bind /afs:/afs --bind /run/user/$(id -u) $FM_TESTING_IMAGE python src/prepare_data.py experiment=fm_testing_binary
```

Batch jobs start the container themselves (sections 4.2 and 6.1); `condor_submit` and `scripts/submit_env.sh` run outside it.

### Anywhere else

```bash
git clone https://github.com/DonatellaGenovese/collide2v-embeddings.git
cd collide2v-embeddings

python -m venv .venv            # Python 3.10 is what the container uses
source .venv/bin/activate       # on Windows: .venv\Scripts\activate
pip install -r requirements.txt

pytest -k "not slow"            # a first check that the environment is sane
pre-commit run --all-files      # the formatting checks CI runs on every push
```

Conda works as well as `venv`, and if the machine already has Apptainer or Docker you can build the same container as on lxplus from `fm_testing.def` and skip the environment entirely. Whatever you choose, the point is that `import torch, lightning, hydra` works and `pytest` runs.

**With a GPU.** `requirements.txt` asks for `torch>=2.0.0` and pip will give you a CUDA build on Linux, but the CUDA version it targets is not always the one your driver supports. Check first, and if the check fails, install the matching build from [pytorch.org](https://pytorch.org/get-started/locally/):

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Then ask for the GPU at run time. The trainer is chosen by config, so you never edit code to switch device:

```bash
python src/train.py trainer=gpu                 # one GPU, bf16 mixed precision
python src/train.py trainer=gpu trainer.devices=2
```


### Setting your paths

The pipeline reads and writes data in four places, defined in `configs/paths/fm_testing.yaml`:

| Path | What the pipeline does there |
| --- | --- |
| `dataset_dir` | **reads** the COLLIDE-2V Parquet files, the input. |
| `eos_data_dir` | **writes** the dataset it builds from them, the `.npy` shards, which training then reads. Section 4.3 shows what lands there. |
| `tmp_data_dir` | **writes temporarily** while building: shards go here first and are moved to `eos_data_dir` when complete. |
| `log_dir` | **writes** the output of each run: its configuration, checkpoints, and the mlflow runs. |

Do not edit `configs/paths/fm_testing.yaml`: it is tracked by Git, so your paths would end up in commits and in other people's defaults. Create `configs/local/default.yaml` yourself (the folder holds only a `.gitkeep`, which keeps it in Git) and set your values in it: Hydra reads it after the shared file, so what you write wins, and Git ignores it. Replace `<username>` with your CERN login and `<l>` with its first letter.

**On lxplus**, the Parquet files are read from the copy on EOS, which `dataset_dir` already points at, so leave it out. Everything the pipeline writes goes to your EOS area, because your AFS home is too small for it:

```yaml
# @package _global_
# configs/local/default.yaml
paths:
  eos_data_dir: /eos/user/<l>/<username>/collide2v_data
  log_dir: /eos/user/<l>/<username>/collide2v_data/logs
  tmp_data_dir: /tmp/<username>/collide2v_tmp
```

**Off lxplus**, there is no EOS: download a few files from Hugging Face first (section 3), and point `dataset_dir` and the event counts at them:

```yaml
# @package _global_
# configs/local/default.yaml
paths:
  dataset_dir: /home/you/data/collide
  eos_data_dir: /home/you/data/collide2v
  log_dir: /home/you/data/collide2v/logs
  tmp_data_dir: /home/you/data/collide2v/tmp
data:
  event_counts_json: /home/you/data/collide/file_event_counts.json
```

Without this file the dataset is written to `data/` inside the repository, which is enough for a test with a few files and nothing more. If the dataset will not fit where it is going, vectorisation stops before writing and says how much space it needs.

These paths, and every other, are resolved from the top of the repository, which the entry points find by looking for the `.project-root` file: clone the repository whole and do not delete it. A script of your own that imports from `src/` needs the same two lines at the top:

```python
import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
```


## 3. The dataset

> **Work in progress.** COLLIDE-2V is still being produced and revised. The numbers, file lists and column names below were measured on the version available at the time of writing (Oct 26), and later releases may change them. Treat this section as a description of what you will find today, not as a specification. Re-run the checks described here whenever a new version is published.

### How the data is organised

One folder per physics process, and inside it Parquet files whose names carry their own metadata:

```
HH_4b/HH_4b-NEVENT10000-RS20000001.parquet
└─ process   └─ process  └─ events requested  └─ random seed of the generation
```

`NEVENT10000` is what was requested from the generator, not necessarily what the file contains: the file above holds 9,993 events. Always take the count from the file, never from its name.

Each row is one collision event. Each column is one property of one kind of object, stored as a list when the object can appear several times per event.

Column names have three parts, `<level>_<object>_<variable>`, and the level matters:

| Prefix | Meaning |
| --- | --- |
| `Gen_` | Generator truth, before the detector. Not usable as an input to a model that should work on real data. |
| `FullReco_` | Objects as the full offline reconstruction would see them. **This is what the pipeline uses.** |
| `L1T_` | Objects as the level-1 trigger would see them, with a coarser resolution. |
| `Event_` | Event-level bookkeeping: cross section, weights, process id. |


### The copy on EOS

```
/eos/project/f/foundational-model-dataset/samples/production_final
```

53 process folders, from single Higgs to top quarks to QCD multijets, several hundred million events in total. It is readable from lxplus and from batch nodes, and it is far too large to copy.

The mapping from the process names used in the configs to these folder names is in `configs/data/collide2v_basic.yaml`, under `process_to_folder`.

### The copy on Hugging Face

[`fastmachinelearning/collide-1m`](https://huggingface.co/datasets/fastmachinelearning/collide-1m), published under MIT, DOI [10.57967/hf/8447](https://doi.org/10.57967/hf/8447): 740 Parquet files over 50 processes, about 953 GB in total, 1.2 to 1.6 GB per file. It is laid out as one directory per process, with the same names as the folders on EOS.

This is the copy to use off lxplus, without a CERN account. Do not clone it: download the few files you need with `scripts/download_collide_subset.py`, in three steps.

**1. See what is there.** This asks Hugging Face for the list of files and downloads nothing:

```bash
python scripts/download_collide_subset.py --list
```

**2. Download a few files per process.** For the two classes of `fm_testing_binary`, the experiment section 4 starts from, three files each, about 8 GB:

```bash
python scripts/download_collide_subset.py --out /home/you/data/collide \
    --processes QCD_HT50toInf ggHbb --files-per-process 3
```

| Option | What it chooses | Default |
| --- | --- | --- |
| `--processes` | which processes, by the names `--list` prints | `QCD_HT50toInf HH_4b` |
| `--files-per-process` | how many files of each, the first ones alphabetically, so the same every time | `2` |
| `--out` | where they go | `data/collide` |

Before it starts, the script prints how many files it takes and roughly how many GB that is. Take at least three per process: the splits are made of whole files, so with two a class cannot fill train, val and test.

After downloading, the script scans the files and writes `file_event_counts.json` next to them, the number of events in each. The pipeline uses it to decide which file goes to which split without opening them all, and the copy shipped with the repository describes the files on EOS, not these.

**3. Tell the pipeline where they are**, with `dataset_dir` and `event_counts_json` in `configs/local/default.yaml`, as in the off-lxplus example of section 2. Three files per class hold about 25k events, fewer than `fm_testing_binary` asks for, so when you build the dataset in section 4, ask for less:

```bash
python src/prepare_data.py experiment=fm_testing_binary \
    data.label=hf_binary 'data.train_val_test_split_per_class=[8000, 8000, 8000]'
```

### EOS and Hugging Face are not the same data

They share file names and process names, which makes it tempting to treat them as interchangeable. They are not, in two separate ways.

**Different events.** The same file name may hold different collisions.

**Different columns.** 173 columns on Hugging Face against 271 on EOS.

| Only on Hugging Face | Only on EOS |
| --- | --- |
| `GenJetAK4`, `GenJetAK8`, `GenPart`, `PFCand`, `PrimaryVertex` | `Event_*`, `Gen_*`, `PFPart`, `Vertex`, `Rho`, `ScalarHT` |

In this repository **the default feature set is the 19 variables both copies have**, `configs/data/collide2v_common.yaml`, so the same experiment definition runs on either. `collide2v_extended_eos.yaml` adds the other ten and only works on EOS. Vectorisation reads the schema of one file per class before it starts, so asking for a column the data does not have fails immediately, naming it, instead of inside a batch job.

**What is the same.** Both copies store these columns as Arrow `large_list<halffloat>`, so the reading code is identical; only the name of the inner list field differs, `item` against `element`, which awkward absorbs. If you build Parquet files of your own, note that a plain `list` and nullable columns also work, but neither is what either copy of COLLIDE-2V uses.

### Inspecting the files yourself

Before trusting any of the above for a process you care about, look:

```python
import pyarrow.parquet as pq

pf = pq.ParquetFile("HH_4b-NEVENT10000-RS20000001.parquet")
print(pf.metadata.num_rows)                    # events, without reading the data
print(len(pf.schema_arrow.names))              # columns
print([c for c in pf.schema_arrow.names if "MuonTight" in c])
```

`scripts/parquet_plotter.py` plots distributions straight from the Parquet files, and `scripts/plot_features.py` does the same for the `.npy` files produced later. Use them to check that what came out of the pipeline still looks like what went in.


## 4. Building a dataset

A dataset is built in two stages, from the Parquet files in `dataset_dir`:

1. **Vectorisation** turns every event into a fixed-length row of numbers: the variables you choose, for a fixed number of objects of each type, padded with zeros.
2. **Preprocessing** rescales those rows so that every feature sits on a scale of order 1, which is what training reads.


### 4.1 Configuring a dataset

Each study is one file in `configs/experiment/`, chosen with `experiment=<name>`. Copy `fm_testing_binary.yaml`, the smallest one, under a new name and edit it: it is then the whole record of what you built. Leave the other configs alone; each explains its own keys in comments.

| File | What it decides |
| --- | --- |
| `configs/experiment/<name>.yaml` | your study: which classes, how many events, the `label` of the output directory, and which feature set |
| `configs/data/collide2v_common.yaml` | the default feature set: 19 variables that both copies of the dataset have |
| `configs/data/collide2v_extended_eos.yaml` | the published study's 29 variables, on lxplus only |
| `configs/data/collide2v_basic.yaml` | what both feature sets share: the seed, the names of the classes (`process_to_folder`), the event-count file |
| `configs/preprocess/collide2v.yaml` | how each kind of variable is rescaled |

Two things to get right in your experiment file, both commented there: give a new `label` whenever you change the event counts or the feature set, and keep `preprocess.enabled: true`.

### 4.2 Running it

Training never builds a dataset: when the one it needs is missing it stops and points here. Rerunning any of the commands below is safe: what is already built is skipped.

**On lxplus, a dataset of a few files per class**, on the login node, inside the container (section 2):

```bash
apptainer exec --bind /eos:/eos --bind /afs:/afs --bind /run/user/$(id -u) $FM_TESTING_IMAGE \
    python src/prepare_data.py experiment=fm_testing_binary
```

`fm_testing_binary`, QCD against ggH→bb with 50k training events each, takes minutes and is the one to start from.

**On lxplus, a large dataset**, on HTCondor. The submit scripts run on lxplus itself, **outside the container**, because `condor_submit` does not exist inside it. `scripts/submit_env.sh` gives them the Python libraries they need, from CVMFS, and creates a 15 MB environment in `~/.venvs/` the first time. The jobs themselves run in the container.

```bash
source scripts/submit_env.sh                                     # once per terminal
python scripts/submit_vectorization_jobs.py experiment=<name>    # ≤50 files per job
condor_q                                                         # wait until empty
python scripts/submit_preprocessing_jobs.py experiment=<name>    # fits the statistics, then submits
```

Preprocessing needs every vectorisation job to be finished. Put every setting in the experiment file: the jobs receive `experiment=` and nothing else, so an override such as `data.label=...` on the command line would not reach them. `FM_TESTING_IMAGE` reaches them from your terminal, so it has to be set there (section 2); the submit scripts refuse to submit when it points nowhere. Logs, submit files and per-job manifests land in `logs/condor_logs/`.

**Off lxplus, with files from Hugging Face** (section 3), the same command as the first, without the container, and with fewer events, since three files per class hold about 25k:

```bash
python src/prepare_data.py experiment=fm_testing_binary \
    data.label=hf_binary 'data.train_val_test_split_per_class=[8000, 8000, 8000]'
```

### 4.3 What lands on disk

Everything goes under `<eos_data_dir>/<label>/`. Shards are written to scratch space first (`tmp_data_dir`, or the node's own disk in a batch job) and moved there once complete:

```
<eos_data_dir>/<label>/
├── vectorized/
│   ├── feature_map.json        # which column sits at which position
│   ├── split_manifest.json     # which Parquet file went to which split (section 5)
│   └── train|val|test/<process>/<file>_x.npy, _y.npy
└── preprocessed/               # what training reads
    ├── feature_map.json        # wider than the raw one: trig and onehot add columns
    ├── norm_stats.json         # the fitted statistics
    └── train|val|test/<process>/<file>_x.npy, _y.npy
```

Each Parquet file gives one pair of shards: `_x.npy` holds the events, `(events, features)` in float32, and `_y.npy` their class as an integer. Use `feature_map.json` to find a column:

```python
import json, numpy as np
fm = json.load(open("feature_map.json"))
X = np.load("<file>_x.npy", mmap_mode="r")
jets = X[:, fm["jets"]["start"]:fm["jets"]["end"]]     # all jet features
```

To check the result, plot the shards next to the Parquet they came from:

```bash
python scripts/plot_features.py --base_dir <eos_data_dir>/<label>/preprocessed --output_dir plots/ --split train --max_files 5
python scripts/parquet_plotter.py --input_dir <dataset_dir> --output_dir plots/parquet
```

After preprocessing a feature should sit around 0 with a spread of order 1. A column of exact zeros is a `topk` slot that never filled; a lone huge outlier usually means statistics fitted on too few files.

### 4.4 What stops a run, and why

The pipeline checks before writing anything, because a large dataset runs for hours:

| Check | Stops when |
| --- | --- |
| Columns | a variable is not in the data, usually an EOS-only one on a Hugging Face download. |
| Feature map | the `label` already holds a dataset built with other variables. |
| Manifest | the seed or the split sizes differ from those the `label` was built with (section 5). |
| Space | the dataset will not fit where it is going. On AFS it reads the quota, since the filesystem reports the whole partition. |

A Parquet file that cannot be read stops the run once every other file is done, and names it; rerunning redoes only that one. `data.skip_unreadable_files=true` goes on without it.



## 5. Which files a dataset is made of

A dataset is a particular set of Parquet files, split into train, val and test, and a particular set of events read from them. Two runs are comparable only if that set is the same, so it is written down when the dataset is built and checked every time it is reused.

### 5.1 The manifest

`split_manifest.json`, in `vectorized/`, records which file went to which split, and under `_meta` the settings that chose them:

```json
{
  "_meta": {"seed": 42, "manifest_strategy": "per_class", "split_counts": [50000, 10000, 10000]},
  "QCD_HT50toInf": {"train": ["QCD_HT50toInf-NEVENT10000-RS26002589.parquet", "..."],
                    "val": ["..."], "test": ["..."]}
}
```

Files are assigned whole, so a split usually holds a little more than it asked for. Once written, the manifest decides what a `label` contains, and the pipeline holds you to it:

| If you | The run |
| --- | --- |
| rerun with the same settings | reuses the manifest and builds only what is missing |
| add a class to `to_classify` | extends the manifest with that class and leaves the others alone |
| change the seed, the strategy or the event counts | stops, names the value that differs, and asks for a new `label` |
| change the variables (`datasets_config`) | stops: `feature_map.json` no longer matches, so the shards would have different columns |
| leave shards on disk that the manifest does not list | stops, and names them |

To plan the splits without opening every Parquet file, the manifest reads how many events each file holds from `src/utils/nEvents_scan/file_event_counts.json`, a snapshot of the copy on EOS. For files it does not list, such as a Hugging Face download, regenerate it with `src/utils/nEvents_scan/scan_parquet_nevent.py` and point `data.event_counts_json` at the new file; the download script of section 3 does this for you.

### 5.2 Seeds

`data.seed` decides which files are drawn. It is inherited from the global `seed`, so one number describes a whole run. With the default `data.manifest_strategy: per_class`, each class draws with its own generator, so adding or removing a class does not change the files of the others. `legacy` exists only to rebuild datasets made before this.

### 5.3 Which events training reads

The events of an epoch are chosen once, before the data loader starts, so:

- asking for N events per class gives exactly N, or all of them with a warning when a class has fewer;
- `num_workers` changes how fast an epoch is read, and nothing else;
- when only part of a shard is needed, its rows are drawn with the seed, not taken from the top of the file.

### 5.4 Empty events

Some events have nothing reconstructed in them: no jet, electron, muon or photon above the storage thresholds. Vectorised, they are rows of zeros, about 5% of `QCD_HT50toInf` and under 0.5% of everything else. `data.drop_empty_events` removes them during vectorisation; it is off by default.

It is a physics choice, not a technical one. Turn it on for a contrastive model, where every empty event is the same point, and to rebuild the datasets of the published study (the `nosparse` in their labels). Leave it off for a classifier, or when the dataset should be what the generator produced. QCD is the class an anomaly-detection model learns as normal, so this moves the edge of what it considers normal: say which way you set it when you report a result. Changing it means vectorising again, into a new `label`.

### 5.5 Reproducing a dataset

Keep the `label`, the `seed`, the strategy, the event counts, the variables and `drop_empty_events`, and you get the same files and the same events. Change any of them, and give the dataset a new `label`.


---

## 6. Training, and the models

### 6.1 Running a training

The dataset has to exist first (section 4): training reads `<eos_data_dir>/<label>/preprocessed/` and stops if it is not there.

Training runs on a GPU. The ordinary lxplus login nodes have none, so on lxplus a run goes either to `lxplus-gpu.cern.ch`, for short interactive tests, or to HTCondor, for everything else.

**On lxplus, interactively**, on a GPU login node, inside the container with `--nv`, from the repository root:

```bash
ssh lxplus-gpu.cern.ch
cd /afs/cern.ch/user/<l>/<username>/collide2v-embeddings
apptainer exec --nv --bind /eos:/eos --bind /afs:/afs --bind /run/user/$(id -u) $FM_TESTING_IMAGE \
    python src/train.py experiment=fm_testing_binary
```

These nodes are shared, so keep it short. Before a long run, check the wiring in a few minutes:

```bash
apptainer exec --nv --bind /eos:/eos --bind /afs:/afs --bind /run/user/$(id -u) $FM_TESTING_IMAGE \
    python src/train.py experiment=<name> \
    trainer.max_epochs=1 +trainer.limit_train_batches=5 +trainer.limit_val_batches=2
```

**On lxplus, on HTCondor**, from the repository root, **outside the container**, with `FM_TESTING_IMAGE` set in that terminal (section 2: it belongs in your `~/.bashrc`). Without it the job waits in the queue and then stops at once, saying so in its `.err`. The job starts the container itself.

```bash
mkdir -p logs/condor_logs                                    # once: HTCondor does not create it
condor_submit src/train_full_pipeline.sub EXPERIMENT=fm_testing_binary
condor_submit src/train_full_pipeline.sub EXPERIMENT=<name> ARGS="model=supcon trainer.max_epochs=50"
condor_submit src/train_full_pipeline.sub EXPERIMENT=<name> ARGS="-m hparams_search=collide2v_optuna_multiclass"
```

`EXPERIMENT` is required. `ARGS` takes any further overrides, separated by spaces. Each job asks for one GPU with at least 40 GB, so it can wait in the queue for a while. The last line is a hyperparameter sweep (section 7). The job's own output goes to `logs/condor_logs/train_<cluster>.out` and `.err`; the run itself lands under `log_dir` like any other.

**Off lxplus, with files from Hugging Face**, with the same `label` and event counts the dataset was built with in section 4.2:

```bash
python src/train.py experiment=fm_testing_binary \
    data.label=hf_binary 'data.train_val_test_split_per_class=[8000, 8000, 8000]'
```

The machine needs a GPU that PyTorch can see; section 2 shows how to check.

**What a run leaves**, in `<log_dir>/train/runs/<date>_<time>/`:

```
.hydra/config.yaml      # the full configuration the run used
checkpoints/            # epoch_NNN.ckpt for the best epoch, and last.ckpt
train.log               # everything printed
roc_plots/, test_predictions/
```

and its metrics in MLflow, under `<log_dir>/mlflow/mlruns` (section 7).

**Evaluating a checkpoint** takes the configuration of the run it came from, so only the checkpoint is needed:

```bash
# on lxplus-gpu, interactively
apptainer exec --nv --bind /eos:/eos --bind /afs:/afs --bind /run/user/$(id -u) $FM_TESTING_IMAGE \
    python src/eval.py ckpt_path=<log_dir>/train/runs/<date>_<time>/checkpoints/last.ckpt

# on HTCondor, outside the container
condor_submit src/eval.sub CKPT=<log_dir>/train/runs/<date>_<time>/checkpoints/last.ckpt
```



### 6.2 The models

Four, and they answer two different questions.

**Classifiers**, which ask *which process is this event*.

| Model | What it does |
| --- | --- |
| `tinyMLP` | Two hidden layers on the flat event vector. The thing to beat: if a transformer does not beat this, the extra structure is not paying for itself. |
| `tinyTransformer` | One token per object plus one for the event-level counts, a projection per group, then a standard encoder. No positional embeddings, because the objects are sorted by pT and their order carries no meaning beyond that. |

**Contrastive encoders**, which ask *what does this event look like*. They produce an
embedding, and a classifier is not the point: `src/models/collide2v_contrastive.py` is
one module, and the config picks which pairs of a batch count as positive.

| Model | Positives | Uses labels |
| --- | --- | --- |
| `simclr` | The two augmented views of one event | No |
| `supcon` | Every event of the same process, views included | Yes |

SimCLR trains on data nobody has labelled, which is the case that matters for a
foundation model. SupCon uses the labels to shape the embedding, which makes it better
at separating the processes it was given and a commitment to that choice.

Both wrap `tinyTransformer` as the encoder, add a projection head that the loss sees and
that is discarded afterwards, and keep a small classification head whose only job is to
give you an accuracy to watch. What comes out of a run is the encoder.

The loss was checked against the two published implementations it replaces: same inputs,
same value to the last digit, same gradients. Those numbers are pinned in
`tests/test_contrastive.py`.

**One thing to know about the transformer.** The type embeddings — the vectors that would
tell a jet token from a muon token — are built and then not added: the lines are
commented out in `src/models/components/transformer.py`. So the model distinguishes
object kinds only through the per-group projections. That is how the published runs were
made; it is not obviously the right choice, and turning them on is a reasonable thing to
try, on a new `label` and against the numbers you already have.

### 6.3 Measuring an embedding

A contrastive run has no accuracy of its own, so a falling loss says nothing about whether the embedding is useful. Freeze the encoder, fit a linear classifier on its output, and see how far that gets. Linear on purpose: anything stronger measures the probe instead of the representation.

The probe rebuilds the encoder from `experiment` and `model` before loading the weights, so both have to be the ones the checkpoint was trained with; `model` can be left out when it is the experiment's default. Like training, it runs on a GPU: it passes every event through the encoder, which on a CPU takes several times longer (for `fm_testing_binary`, about 11 minutes against 2), and it warns in its log when it finds no GPU.

**On lxplus, interactively**, on `lxplus-gpu.cern.ch`, inside the container with `--nv`:

```bash
apptainer exec --nv --bind /eos:/eos --bind /afs:/afs --bind /run/user/$(id -u) $FM_TESTING_IMAGE \
    python src/eval_probes.py experiment=<name> model=simclr \
    ckpt_path=<log_dir>/train/runs/<date>_<time>/checkpoints/last.ckpt
```

**On lxplus, on HTCondor**, from the repository root, outside the container. The job asks for one GPU:

```bash
condor_submit src/probe.sub EXPERIMENT=<name> ARGS="model=simclr" \
    CKPT=<log_dir>/train/runs/<date>_<time>/checkpoints/last.ckpt
```

The job's output goes to `logs/condor_logs/probe_<cluster>.out`.

**Off lxplus, with files from Hugging Face**, on a machine with a GPU that PyTorch can see (section 2), with the `label` and event counts the dataset was built with:

```bash
python src/eval_probes.py experiment=fm_testing_binary model=simclr \
    data.label=hf_binary 'data.train_val_test_split_per_class=[8000, 8000, 8000]' \
    ckpt_path=<log_dir>/train/runs/<date>_<time>/checkpoints/last.ckpt
```

Each run prints accuracy and AUROC per split and writes them to `probe_results.json` in a new directory, `<log_dir>/eval/runs/<date>_<time>/`, not in the training run's. Add `+probe.save_embeddings=true` to keep the embeddings too, as `embeddings.pt` next to it: that is what a UMAP plot or an anomaly-detection study starts from.

It works on every model in the repository, classifiers included, so a probe on `tinyTransformer`'s embedding can be set next to one on SimCLR's. The number to compare against is a `tinyTransformer` trained end to end on the same data: a probe that matches it means the embedding kept what the task needs.



### 6.4 Adding a model

The pipeline builds a model, so a model has to be buildable by it. Five requirements,
each marked in `src/models/template_model.py`, which is a working two-layer classifier
whose purpose is to be copied:

1. **Take hyperparameters only.** Whatever `configs/model/<name>.yaml` holds arrives in
   `__init__`; call `save_hyperparameters()`.
2. **Build the layers in `setup()`,** from `self.trainer.datamodule`: `dm.vlen` is the
   width of an event, `dm.num_classes` the number of processes. Neither is known before
   the data is prepared, which is why `__init__` cannot do it. A model that works on
   objects reads `feature_map.json` from `dm.paths["eos_preproc_dir"]` instead. `setup()`
   runs more than once, so guard against rebuilding.
3. **Offer `get_embeddings(x)`,** one vector per event. Optional for a classifier,
   required for anything whose point is the representation, since that is what
   `src/eval_probes.py` calls.
4. **Log `train/loss`, `train/acc`, `val/loss`, `val/acc`, `val/acc_best`.** The
   checkpoint callback, early stopping and the Optuna sweeps refer to these by name, so a
   model that logs other names trains and then cannot be checkpointed on its best epoch.
5. **Build the optimiser in `configure_optimizers()`** from the partial in the config, so
   a sweep can vary the learning rate without touching code.

Then:

```bash
cp src/models/template_model.py src/models/my_model.py
cp configs/model/template.yaml configs/model/my_model.yaml   # change _target_
python src/train.py experiment=fm_testing_binary model=my_model \
    +trainer.limit_train_batches=5 +trainer.limit_val_batches=2
```

`tests/test_contrastive.py` checks that contract on both the template and the contrastive module, so if you break it the suite says so.

### 6.5 Adding an augmentation

An augmentation for a contrastive objective has to change the numbers without changing
what the event is. Two kinds ship, in `src/models/components/contrastive.py` and
`src/data/augmentations.py`:

| `model.augmentation` | What it does |
| --- | --- |
| `masking` | Zeroes slots, or whole objects with `mask_full_particle`. Stands in for a detector that does not see everything. Needs nothing but the feature map. |
| `physics` | Rotates the event in φ, boosts it in η, smears each pT. Transformations physics says leave the event equivalent. |

Yours goes next to these, as a callable taking `[batch, vlen]` and returning the same
shape, and a branch in `_build_augmentation`. Three things to respect, all of them tested
for the two above:

- **Act on normalised vectors.** The loader returns preprocessed events, so a shift of
  Δη in physical units is a shift of Δη/IQR here; the IQRs are in `norm_stats.json`,
  which is why `PhysicsAugmentation` reads it.
- **Leave padding alone.** An empty slot is zero, and an augmentation that writes into
  one invents an object the detector never saw.
- **Keep the event an event.** The φ rotation is a single angle for every object in the
  event; one angle per object would scramble the geometry the model is meant to learn.

Also worth knowing what cannot be expressed: an event is a fixed number of slots, so an
augmentation that reorders or adds objects has nowhere to go.

## 7. Logging and sweeps

Every run writes its numbers through a logger, chosen with `logger=<name>` from
`configs/logger/`:

| Logger | Where the numbers go |
| --- | --- |
| `mlflow` | the filesystem, under `logs/mlflow/mlruns`. The default: nothing leaves the machine and nothing needs an account. |
| `wandb` | a server, so a batch run can be watched from anywhere. Needs an account and a key, and a run's metadata leaves CERN. |
| `csv` | a file, for plotting it yourself. |

A run's own directory, `logs/train/runs/<date>/`, holds the checkpoints and
`.hydra/config.yaml` — the fully composed config, which is the whole description of what
ran, down to the dataset label and the seed. That file is what to read when a result has
to be reproduced later.

Sweeps use Optuna through Hydra, from `configs/hparams_search/`, and `-m` is what turns a
run into a sweep:

```bash
python src/train.py -m hparams_search=collide2v_optuna_multiclass experiment=<name>
```

**[docs/logging.md](docs/logging.md) has the rest**: serving the MLflow UI from lxplus
over an SSH tunnel, using wandb offline on a batch node and syncing afterwards, which
metrics to watch for a classifier against a contrastive encoder — where there is no
accuracy to watch and a falling loss is not enough — and how to read a sweep.
