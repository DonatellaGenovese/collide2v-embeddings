import os
import random
import numpy as np
import torch
from torch.utils.data import IterableDataset

from src.data.utils import class_seed


class LocalVectorDataset(IterableDataset):
    """Stream a fixed sample of events from preprocessed .npy shards.

    The sample is decided once, in the main process, as a plan of which rows of
    which shard belong to the epoch, and only then handed out to the DataLoader
    workers. That order matters: when each worker picked its own events, the
    sample depended on how many workers were running, and asking for N events
    per class gave roughly N/num_workers of them, taken from the first rows of
    the files. Here the plan is built first, so `num_workers` changes how fast
    the epoch is read and nothing else.

    Args:
        base_dir: directory holding one subdirectory per class
        per_class_limit: events to take per class, or None for all of them
        shuffle_file_order: shuffle the plan, so a batch mixes classes
        classnames: classes to read, in the order their labels were assigned
        folder_map: class name -> directory name
        seed: decides which events are taken and in which order
    """

    def __init__(
        self,
        base_dir,
        per_class_limit=None,
        shuffle_file_order=True,
        classnames=None,
        folder_map=None,
        seed=42,
    ):
        super().__init__()
        self.base_dir = base_dir
        self.per_class_limit = per_class_limit
        self.shuffle = shuffle_file_order
        self.seed = seed

        # Either find all class directories or use provided class names and folder map
        if classnames is None:
            class_dirs = [
                os.path.join(self.base_dir, folder)
                for folder in sorted(os.listdir(self.base_dir))  # sorted: determinism
                if os.path.isdir(os.path.join(self.base_dir, folder))
            ]
        else:
            folder_names = []
            for cname in classnames:
                if folder_map and cname in folder_map:
                    folder_names.append(folder_map[cname])
                else:
                    raise ValueError(f"Class name {cname} not found in folder_map.")
            class_dirs = [os.path.join(self.base_dir, d) for d in folder_names]
            for d in class_dirs:
                if not os.path.isdir(d):
                    raise ValueError(f"Provided folder name {d} is not a directory in base_dir {self.base_dir}")

        self.class_dirs = class_dirs
        self.plan = self._build_plan()
        self.num_events = sum(entry[2] for entry in self.plan)

    # ------------------------------------------------------------------
    # Planning: which rows of which shard make up the epoch
    # ------------------------------------------------------------------

    def _shards_of(self, class_dir):
        """(filename, rows) per shard, sorted by name, rows read from the header."""
        shards = []
        for fx in sorted(f for f in os.listdir(class_dir) if f.endswith("_x.npy")):
            rows = np.load(os.path.join(class_dir, fx), mmap_mode="r").shape[0]
            shards.append((fx, int(rows)))
        return shards

    def _build_plan(self):
        """Plan of (class_dir, filename, n_rows, row_indices or None) entries.

        `row_indices` is None when the whole shard is taken, which is the usual
        case; at most one shard per class is used partially, and there the rows
        are drawn with the seed rather than taken from the top of the file.
        """
        plan = []
        for class_dir in self.class_dirs:
            folder = os.path.basename(class_dir)
            shards = self._shards_of(class_dir)
            if not shards:
                raise ValueError(f"No *_x.npy shards in {class_dir}")

            available = sum(rows for _, rows in shards)
            if self.per_class_limit is None:
                for fx, rows in shards:
                    plan.append((class_dir, fx, rows, None))
                print(f"🟢 {folder}: {available:,} events in {len(shards)} shards (no limit)")
                continue

            wanted = int(self.per_class_limit)
            if available < wanted:
                print(
                    f"⚠️  {folder}: {available:,} events available, {wanted:,} requested. "
                    "Taking all of them; the classes are no longer balanced."
                )
                wanted = available

            # A seeded shard order, derived from the class name so that the
            # sample of one class does not depend on the other classes.
            rng = np.random.default_rng(class_seed(folder, self.seed))
            order = rng.permutation(len(shards))

            taken = 0
            used = 0
            for idx in order:
                if taken >= wanted:
                    break
                fx, rows = shards[idx]
                missing = wanted - taken
                if rows <= missing:
                    plan.append((class_dir, fx, rows, None))
                    taken += rows
                else:
                    chosen = np.sort(rng.permutation(rows)[:missing])
                    plan.append((class_dir, fx, int(len(chosen)), chosen))
                    taken += len(chosen)
                used += 1

            print(f"🟢 {folder}: {taken:,} events from {used} of {len(shards)} shards")

        if self.shuffle:
            rng = np.random.default_rng(self.seed)
            plan = [plan[i] for i in rng.permutation(len(plan))]

        return plan

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    # No __len__ on purpose, although `num_events` is known exactly. Defining it
    # makes len(DataLoader) a number of batches that assumes one stream, while
    # every worker closes its own last, partial batch: with 4 workers and a batch
    # of 7, a 50-event epoch takes 9 batches against a len() of 8. Lightning
    # reads that len() as the length of the epoch and would stop one batch early,
    # dropping events. Read `num_events` for the count instead.

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_id, num_workers = 0, 1
        else:
            worker_id, num_workers = worker_info.id, worker_info.num_workers

        # Interleaved rather than contiguous: with one block per worker, a worker
        # could end up holding only the shards of one class.
        for class_dir, fx, _, rows in self.plan[worker_id::num_workers]:
            X = np.load(os.path.join(class_dir, fx), mmap_mode="r")
            y = np.load(os.path.join(class_dir, fx.replace("_x.npy", "_y.npy")), mmap_mode="r")

            indices = range(len(y)) if rows is None else rows
            for i in indices:
                yield torch.from_numpy(X[i].copy()).float(), torch.tensor(y[i]).long()


class ShuffleBuffer(IterableDataset):
    """Shuffle streaming samples from an IterableDataset.

    `seed` makes the shuffle repeatable; without it the buffer follows the torch
    state, which differs from run to run unless the trainer is seeded.
    """

    def __init__(self, dataset, buffer_size=20000, seed=None):
        super().__init__()
        self.dataset = dataset
        self.buffer_size = buffer_size
        self.seed = seed

    # No __len__ here either, for the reason given in LocalVectorDataset.

    def __iter__(self):
        if self.seed is None:
            seed = torch.initial_seed() % 2**32
        else:
            # Offset per worker, or every worker would draw the same order.
            worker_info = torch.utils.data.get_worker_info()
            worker_id = 0 if worker_info is None else worker_info.id
            seed = self.seed + worker_id

        rng = random.Random(seed)
        buf = []

        for sample in self.dataset:
            buf.append(sample)
            if len(buf) >= self.buffer_size:
                idx = rng.randrange(len(buf))
                yield buf.pop(idx)

        while buf:
            idx = rng.randrange(len(buf))
            yield buf.pop(idx)
