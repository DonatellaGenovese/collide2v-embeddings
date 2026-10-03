<div align="center">

# COLLIDE-2V embeddings

**Vectorise the COLLIDE-2V dataset, train classifiers and contrastive encoders on it, and evaluate the embeddings they produce.**

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
| Vectorisation and preprocessing | Working. What still bites is listed in section 4.6. |
| tinyMLP, tinyTransformer classifiers | Working. |
| Reading the dataset from EOS | Working. |
| Reading a Hugging Face download | Supported: `scripts/download_collide_subset.py` fetches a subset and scans its event counts, and the default feature set is the one both copies have. See section 3. |
| Contrastive models, augmentations, probes | To be added. |
| Reproducible file and event selection | Working, and tested without EOS. See section 5. |

## 2. Installation

You will run this code in one of two places, and the setup differs.

- **On lxplus**, inside an Apptainer container. This is where the full dataset lives, and where you submit batch jobs.
- **Anywhere else**: your laptop, a group workstation, a university cluster or a rented GPU machine, in a virtual environment, on a subset of the data downloaded from Hugging Face.

The code is the same in both cases. What changes is where the data sits, where the outputs go, and whether there is a GPU.

### On lxplus

```bash
# 1. Clone, on AFS. Do not clone onto /eos: batch jobs cannot execute files there.
cd /afs/cern.ch/user/<first letter>/<username>
git clone https://github.com/DonatellaGenovese/collide2v-embeddings.git
cd collide2v-embeddings

# 2. Build the container. It installs everything in requirements.txt.
#    Expect a few minutes and a .sif file of about 9 GB: your AFS home directory
#    has a 10 GB quota, so build it under /afs/cern.ch/work/... or on EOS, and tell
#    the wrapper scripts where it is with FM_TESTING_IMAGE. The build cache lives in
#    ~/.apptainer and fills that quota too; APPTAINER_CACHEDIR moves it.
export APPTAINER_CACHEDIR=/afs/cern.ch/work/<letter>/<user>/.apptainer
export FM_TESTING_IMAGE=/afs/cern.ch/work/<letter>/<user>/fm_testing.sif
apptainer build $FM_TESTING_IMAGE fm_testing.def

# 3. Check that it works. /eos must be bound explicitly, it is not visible by default.
apptainer exec --bind /eos:/eos --bind /afs:/afs $FM_TESTING_IMAGE python -c "import torch, lightning, hydra; print('ok')"
```

From then on, every command in this README that starts with `python` is meant to run inside the container:

```bash
apptainer exec --bind /eos:/eos --bind /afs:/afs fm_testing.sif python src/train.py
```

Add `--nv` to that command when you want the GPU.

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
python src/train.py trainer=cpu                 # force CPU even on a GPU machine
```


**Your own data.** Off lxplus there is no dataset until you fetch one: `python scripts/download_collide_subset.py --list` shows what the public copy holds, and section 3 has the two commands that turn a few of its files into a dataset.

**Your own paths.** By default the shards are written under `data/` inside the repository, which is enough for a small dataset and is already in `.gitignore`; the Parquet files are read from the copy on EOS, which off lxplus does not exist. Rather than editing `configs/paths/fm_testing.yaml`, which everyone shares, put your own values in `configs/local/default.yaml`. Both `configs/train.yaml` and `configs/vectorize_preprocess.yaml` end their defaults list with `- optional local: default`, so Hydra loads that file when it exists and carries on when it does not, and `.gitignore` already lists it, so your settings are never committed and never collide with anyone else's:

Three paths matter, and the names of two of them are historical: `eos_data_dir` is simply where the shards are written and read, `tmp_data_dir` the scratch space used while writing them, and neither has to be on EOS.

**On lxplus**, put the shards in your EOS user area, which has room for them, and the scratch space on AFS work. Your AFS home does not: it has a 10 GB quota, and a real dataset is hundreds of gigabytes.

```yaml
# @package _global_
# configs/local/default.yaml — replace <l> with the first letter of your username
paths:
  # dataset_dir already points at the shared copy on EOS, so it is not repeated here
  eos_data_dir: /eos/user/<l>/<username>/collide2v_data
  tmp_data_dir: /afs/cern.ch/work/<l>/<username>/collide2v/tmp
```

Batch jobs ignore `tmp_data_dir` and write their scratch to the node's own `$TMPDIR`, which is local disk and much faster; the value above is what a run on the login node uses.

**On your own machine**, both are ordinary directories, and `dataset_dir` is where you put the files downloaded from Hugging Face:

```yaml
# @package _global_
paths:
  dataset_dir: /home/you/data/collide
  eos_data_dir: /home/you/data/collide2v
  tmp_data_dir: /home/you/data/collide2v/tmp

data:
  num_workers: 4          # a sane value for a laptop
```

For a one-off, the three also read the environment, as `COLLIDE_DATASET_DIR`, `COLLIDE_DATA_DIR` and `COLLIDE_TMP_DIR`, but a file you can read back later is the better record of what a run used.

### Two things that are easy to miss

**The project root.** Hydra resolves paths against the `PROJECT_ROOT` environment variable. The entry points set it themselves: each calls `rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)`, which walks up until it finds the `.project-root` file at the top of the repository. That is why the repository must be cloned whole, and why `.project-root` must not be deleted. If you write your own script that imports from `src/`, start it with the same two lines.

**The event-count file.** `src/utils/nEvents_scan/file_event_counts.json` records how many events each Parquet file contains, so that building a manifest does not have to open every file. It is only used for that: how many events a dataset actually holds is read from the shards. If you use a process it does not list, or the dataset is updated, regenerate it and point `data.event_counts_json` at the new file; the script is `src/utils/nEvents_scan/scan_parquet_nevent.py`. Reading only the Parquet metadata, the scan is far quicker than reading the files. That script imports `tqdm`, which is not in `requirements.txt`, so install it first.

## 3. The dataset

> **Work in progress.** COLLIDE-2V is still being produced and revised. The numbers, file lists and column names below were measured on the version available at the time of writing, and later releases may change them. Treat this section as a description of what you will find today, not as a specification. Re-run the checks described here whenever a new version is published.

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

53 process folders, from single Higgs to top quarks to QCD multijets, several hundred million events in total. This is the complete dataset and the one all published results so far were produced with. It is readable from lxplus and from batch nodes, and it is far too large to copy.

The mapping from the process names used in the configs to these folder names is in `configs/data/collide2v_basic.yaml`, under `process_to_folder`.

### The copy on Hugging Face

[`fastmachinelearning/collide-1m`](https://huggingface.co/datasets/fastmachinelearning/collide-1m), published under MIT, DOI [10.57967/hf/8447](https://doi.org/10.57967/hf/8447). 740 Parquet files over 50 processes, about 953 GB in total, with individual files of 1.2 to 1.6 GB.

This is the copy to use if you do not have the access to CERN account. Do not clone the repository: download the few files you need.

The repository is laid out as one directory per process, with the same names as the folders on EOS. A script does the downloading:

```bash
# what is there, and how many files each process has
python scripts/download_collide_subset.py --list

# three files of QCD, about 4 GB, into data/collide
python scripts/download_collide_subset.py --out data/collide \
    --processes QCD_HT50toInf --files-per-process 3
```

Check the counts before asking for more: processes have between 2 and 100 files, of 1.2 to 1.6 GB each. Three per class is a sensible start — the splits take whole files, so a class with two files cannot fill train, val and test.

The script then writes `data/collide/file_event_counts.json`, by scanning what it downloaded. That file is needed: planning the splits reads event counts, and the snapshot shipped with the repository describes the files on EOS, not these. Pass both paths:

```bash
python src/prepare_data.py experiment=fm_testing_18class_highlevel \
    paths.dataset_dir=data/collide \
    data.event_counts_json=data/collide/file_event_counts.json \
    data.label=hf_subset \
    'data.to_classify=[QCD_inclusive]' \
    'data.train_val_test_split_per_class=[8000, 8000, 8000]'
```

Keep the default feature set, `collide2v_common`: `collide2v_extended_eos` asks for columns this copy does not have, and vectorisation stops before starting and names them.

### EOS and Hugging Face are not the same data

They share file names and process names, which makes it tempting to treat them as interchangeable. They are not, in two separate ways.

**Different events.** The same file name holds different collisions. `QCD_HT50toInf-NEVENT10000-RS26000001.parquet` has 8,221 events on Hugging Face and 8,243 on EOS; `HH_4b-NEVENT10000-RS20000001.parquet` has 9,992 against 9,993, and of the columns those two versions share, not one holds identical values. A number obtained on EOS cannot be compared with a number obtained on Hugging Face. Say which copy you used, in your notes and in anything you publish.

**Different columns.** 173 columns on Hugging Face against 271 on EOS, and the difference is not a subset relation:

| Only on Hugging Face | Only on EOS |
| --- | --- |
| `GenJetAK4`, `GenJetAK8`, `GenPart`, `PFCand`, `PrimaryVertex` | `Event_*`, `Gen_*`, `PFPart`, `Vertex`, `Rho`, `ScalarHT` |

Of the 29 input variables used in the published study, ten exist only on EOS. The list was checked against a downloaded file, not taken from the documentation:

```
JetPuppiAK4_NCharged, JetPuppiAK4_NNeutrals
Electron_Charge, Electron_D0, Electron_DZ
MuonTight_Charge, MuonTight_D0, MuonTight_DZ
PhotonTight_EhadOverEem, PhotonTight_IsolationVarRhoCorr
```

The consequence for this repository: **the default feature set is the 19 variables both copies have**, `configs/data/collide2v_common.yaml`, so the same experiment definition runs on either. `collide2v_extended_eos.yaml` adds the other ten and only works on EOS. Vectorisation reads the schema of one file per class before it starts, so asking for a column the data does not have fails immediately, naming it, instead of inside a batch job.

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

## 4. Vectorisation and preprocessing

Both stages are driven by `configs/vectorize_preprocess.yaml`, which pulls in a `data`, a `preprocess` and an `experiment` config. Read section 4.6 before launching anything large.

The convention is to leave the base configs alone and put everything specific to your study in one file under `configs/experiment/`, selected with `experiment=<name>`. That file is then the whole description of what you ran.

### 4.1 Choosing the features

Two feature sets come with the repository, and an experiment picks one with `override /data:` in its defaults list:

| Data config | Columns | Where it runs |
| --- | --- | --- |
| `collide2v_common` | 20 | EOS and Hugging Face. The default. |
| `collide2v_extended_eos` | 29 | EOS only. The feature set of the published study: it adds the neutral multiplicity of jets, and the charge, D0 and DZ of electrons and muons, and the hadronic fraction and isolation of photons. |

`fm_testing_18class_highlevel` uses the first, `fm_testing_18class_highlevel_eos` the second, and they write to different `label`s so both datasets can exist side by side.

Either way, `datasets_config` is what lists the columns, grouped by object type:

```yaml
jets:
  cols: [FullReco_JetPuppiAK4_PT, FullReco_JetPuppiAK4_Eta, FullReco_JetPuppiAK4_Phi]
  topk: 12      # keep the 12 highest-pT jets, sorted by the first column
  count: true   # add one feature with the number of jets in the event
```

- `topk: k` gives the group *k* slots. An event with fewer objects gets zeros; one with more keeps the *k* highest by the first column and **the rest are dropped**. Check that this does not cut into the signal you care about.
- `topk: null` is for scalars, like MET, where there is nothing to sort.
- `count: true` adds the true multiplicity before truncation. Without it, the model cannot tell 8 objects from 20.

The rest of the configuration:

| Key | Meaning |
| --- | --- |
| `to_classify` | Processes to use, as names from `process_to_folder`. Their order defines the integer labels. |
| `train_val_test_split_per_class` | Target events per class, `[train, val, test]`. |
| `label` | Name of the output directory. **Change it whenever you change anything above**, or new data lands in the same folder as the old and the two are silently mixed. |
| `paths` | Where the Parquet files are read from and where the `.npy` shards go. |

### 4.2 Running it

**Producing a dataset is its own command**, and training does not do it: a run that finds its data missing stops and tells you this. Vectorising the full dataset takes days, and a training job that started doing it would spend its GPU allocation on that.

For a dataset small enough to build on one machine — a few files per class, which is the normal case off lxplus:

```bash
python src/prepare_data.py experiment=fm_testing_binary   # two classes, 50k events each
python src/prepare_data.py experiment=<name> data.label=my_small_test   # a separate one
```

`fm_testing_binary` is the smallest experiment that does something: QCD against ggH→bb, on the common feature set. It takes minutes, and it is the one to start from.

For the full dataset, send the work to HTCondor. The second stage needs the first to be finished:

```bash
# from the project directory on AFS, with the container available
python scripts/submit_vectorization_jobs.py experiment=<name>    # ≤50 files per job
condor_q                                                        # wait until empty
python scripts/submit_preprocessing_jobs.py experiment=<name>    # fits the stats, then submits
```

Pass the same `experiment=` to both, and to the training that follows: it reaches the jobs, so they build the dataset you asked for rather than the one the config happens to default to.

Logs and per-job manifests land in `logs/condor_logs/`. Both scripts are idempotent: files already produced are skipped, so rerunning them submits only what is missing. Both routes read and write the same `split_manifest.json`, so a local run and the batch system cannot disagree about which files the dataset contains.

### 4.3 What lands on disk

```
<eos_data_dir>/<label>/
├── vectorized/
│   ├── feature_map.json        # which column sits at which position
│   ├── split_manifest.json     # which Parquet file went to which split, plus
│   │                            # the seed and strategy that chose them
│   └── train|val|test/<process>/<file>_x.npy, _y.npy
└── preprocessed/
    ├── feature_map.json        # after transforms, so wider than the raw one
    ├── norm_stats.json         # the statistics used, needed to reapply the scaling
    └── train|val|test/<process>/<file>_x.npy, _y.npy
```

`_x.npy` holds `(events, features)` float32, `_y.npy` the integer label of the class. `feature_map.json` is the only way to know what a column means, so read it rather than counting by hand:

```python
import json, numpy as np
fm = json.load(open("feature_map.json"))
X = np.load("<file>_x.npy", mmap_mode="r")
jets = X[:, fm["jets"]["start"]:fm["jets"]["end"]]     # all jet features
```

### 4.4 Preprocessing

Each group of features gets a transform and, independently, a normaliser. Both are set per group in `configs/preprocess/collide2v.yaml`.

| Transform | Applied by default to |
| --- | --- |
| `log1p` | pT, mass, MET — long tails compressed, sign preserved |
| `trig` | φ, replaced by its sine and cosine, so that 0 and 2π are the same point |
| `onehot` | PID, charge, b-tag — categories, not numbers |
| `identity` | everything else |

| Normaliser | Applied by default to |
| --- | --- |
| `robust` | most continuous features: median 0, scaled by the interquartile range, so outliers do not set the scale |
| `minmax` | object counts and the PUPPI b-tag, into [0, 1] |
| `none` | φ and the one-hot groups, which are already on a fixed scale |
| `standard` | nothing by default: mean 0, std 1, available if you want it |

`trig` and `onehot` change the number of columns, which is why the preprocessed feature map is wider than the raw one.

`mode` controls what a run does: `fit_only` computes the statistics, `apply_only` reuses existing ones, `fit_and_apply` does both. The fit reads training files only — `fit_num_files_per_class` of them per class, the first ones alphabetically, which it prints — and the statistics are then applied to all three splits, so val and test never influence the scale.

Rerunning is safe. The pipeline compares the two trees shard by shard: when some vectorised shards have no preprocessed counterpart, as after a batch job that died part way through, it says which ones and applies the **stored** statistics to those alone. It never refits, because refitting on a different set of files would put the new shards on a different scale from the old.

### 4.5 Checking the output

```bash
# distributions from the .npy shards, raw or preprocessed
python scripts/plot_features.py --base_dir <label>/preprocessed --output_dir plots/ \
    --split train --max_files 5

# the same variables in the source Parquet
python scripts/parquet_plotter.py --input_dir <dataset_dir> --output_dir plots/parquet
```

Compare the two. After preprocessing a feature should sit around 0 with a spread of order 1; a column of exact zeros means a `topk` slot never filled, and a lone huge outlier usually means statistics fitted on too few files.

### 4.6 What still bites

- **A Parquet file that cannot be read is skipped.** The error is printed and vectorisation carries on, so a split can come out smaller than asked for while every job reports success. Read the logs, do not just check that they finished.
- **The output paths default under the repository.** Small datasets are fine there, hundreds of gigabytes are not: set them in `configs/local/default.yaml`, as section 2 describes.
- **Checking a dataset reads every shard header.** `has_enough_events` and the loader both open one file per shard, which on EOS costs about a minute and a half for a twelve-class dataset. It is paid once at startup.

## 5. Which files a dataset is made of

A dataset here is not "the COLLIDE-2V sample": it is a particular set of Parquet files, cut into train, val and test, and a particular set of events read from them. Two runs are comparable only if that set is the same, so it is written down, checked, and reproducible from a seed.

### 5.1 The manifest

`split_manifest.json`, next to the vectorised shards, records which file went to which split, and under `_meta` the seed, the strategy and the split sizes that produced it:

```json
{
  "_meta": {"seed": 42, "manifest_strategy": "per_class", "split_counts": [1000000, 100000, 100000]},
  "QCD_HT50toInf": {"train": ["QCD_HT50toInf-NEVENT10000-RS26000001.parquet", "..."],
                    "val": ["..."], "test": ["..."]}
}
```

Files are assigned whole, and a split stops as soon as it holds the events asked for, so the first file of each split is where the previous one stopped. The rules around that file are what matter:

- **An existing manifest is never rewritten.** It is reused as it is, and only *extended*, with classes it does not yet cover.
- **A run that does not match it stops.** Change the seed, the strategy or the split sizes of a dataset that already exists and the run refuses, naming both values, and tells you to use a new `data.label`. The shards on disk were chosen by the stored settings; mixing in files drawn differently would leave one directory holding two samples.
- **Shards the manifest does not list stop the run too.** For a dataset built before the manifest recorded its metadata this is a warning instead, since those predate the rule.
- **The columns are fixed as well.** The manifest says which files a dataset holds, not which features were taken from them, so `feature_map.json` is compared too: change `datasets_config` and keep the `label`, and the run stops rather than adding shards of a different width — or, worse, the same width with different columns in them.

### 5.2 Seeds

`data.seed` decides the draw and is inherited from the global `seed`, so one seed describes a whole run.

`data.manifest_strategy` decides how it is used:

| Strategy | How the files are drawn |
| --- | --- |
| `per_class` (default) | One generator per class, seeded from the class name. The files of a class depend only on that class, the seed and the targets. |
| `legacy` | One generator shared by all classes, in config order. Only to rebuild a dataset produced before `per_class` existed. |

Under `legacy`, adding or removing one class in `to_classify` shifts the draw of every class after it, so a dataset could not be extended or reduced without rebuilding all of it, and two studies that differed by one class shared no files. Under `per_class`, adding a class leaves the others untouched and the manifest is simply extended.

### 5.3 Which events the loader reads

The sample is decided once, in the main process, as a plan of which rows of which shard belong to the epoch, and only then handed to the DataLoader workers. So:

- asking for N events per class gives exactly N, or all of them with a warning when a class has fewer;
- `num_workers` changes how fast an epoch is read and nothing else;
- when one shard of a class is used only partly, its rows are drawn with the seed rather than taken from the top of the file.

Each split draws with its own seed, so `val` does not take the same rows of the same shards as `train`.

### 5.4 Empty events

Some events reach the file with nothing reconstructed in them: no jet, no electron, no muon, no photon above the thresholds at which objects are stored. Vectorised, such an event is a row of zeros. `data.drop_empty_events` decides whether those rows are written at all; it is off by default, so they are.

It belongs in this section because it changes what the dataset contains, so it is part of what identifies one, like the seed. And because it acts during vectorisation, changing your mind means vectorising again, into a new `label` — it cannot be switched at training time.

**How many events it concerns:** about 5% of `QCD_HT50toInf`, and under 0.5% of everything else. QCD is generated with HT > 50 GeV, so a real fraction of it leaves nothing above the storage thresholds; the other processes nearly always produce something.

**When to turn it on.** For a contrastive model: every empty event is the same point, both augmented views of it are identical, and a block of duplicate samples is exactly what those objectives are sensitive to. Also when rebuilding the datasets behind the published study, which were produced with it on — that is the `nosparse` in their labels.

**When to leave it off.** For a classifier, which handles a row of zeros without trouble, and whenever you want the dataset to be what the generator produced. Note that QCD is also the class an anomaly-detection model is trained on as "normal", so this choice moves the soft edge of what it considers normal: say which way you set it when you report a result.

### 5.5 Reproducing a dataset

Keep the `label`, the `seed`, the strategy, the split sizes and `drop_empty_events`, and you get the same files and the same events. Change any of them and give the dataset a new `label`: that is the one rule the rest depends on.

To rebuild a dataset produced before all this, point the config at its directory and its stored manifest is used as it is. If it has no `_meta`, the run says so and uses it anyway.

### 5.6 Checking it yourself

```bash
pytest                                      # the whole suite, nothing needs EOS
pytest tests/test_pipeline_determinism.py   # just the pipeline, on Parquet it writes itself
```

Those tests vectorise and preprocess a small fake dataset twice and require the shards to be identical, read it with several worker counts and require the same events, and delete a preprocessed shard to check it is rebuilt identically without refitting the statistics.

The suite should end green with a dozen tests skipped: those are the ones inherited from the template, which train on the default configuration and therefore need a real dataset. The skip reason says so. If one of them fails rather than skips, that is a bug worth reporting.

GitHub Actions runs the same suite on every push and pull request, on Python 3.10 and 3.11, plus the four `pre-commit` hooks. So a red tick means something real; it was not always so.

---

<!--
  SECTIONS BELOW ARE THE ORIGINAL README AND ARE BEING REWRITTEN.
  Next: training and the models, then extending the code.
-->

### Training Iterations and GPU Usage
If all the preprocessed files are ready, training can simply again be done with `src/train.py`, which will recognize the available files and skip vectorization and preprocessing. Training can also be submitted to compute nodes using the condor batch submission system via `condor_submit src/train_full_pipeline.sub`. Given the corresponding trainer setting, training on GPU is also supported locally and via condor submission. All condor submission logs will by default be saved in `logs/condor_logs`.

#### Supported Models
The currently implemented models are an MLP and a Transformer. Their Lightning source code is available in `src/models`. More custom models can be implemented too and should follow the Lightning framework to be compatible with the rest of the repository.

### Logging and Hyperparameter Sweeps
The repository is build with support of the mlflow logger and the optuna hyperparameter sweeper. Logs of your runs can be accessed on the mlflow user interface by calling `mlflow ui` from inside `logs/mlflow`. If you perform a hyperparameter sweep, its output will be saved in a .db file that can be inspected using `notebooks/optuna_sweep_results.ipynb`.
