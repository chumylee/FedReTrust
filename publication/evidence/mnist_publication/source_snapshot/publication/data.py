"""Pinned public datasets and paired experiment plans."""
import gzip
import io
import math
import shutil
import struct
import tempfile
from itertools import combinations
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile

import numpy as np
import torch
from torch import nn

from experiment_core import MnistCNN, file_sha256
from non_iid_partitions import LabelSkewConfig, prepare_label_skew
from scenario_utils import seed_for

DATA = Path(__file__).resolve().parent / "data"
FASHION_URL = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"
EMNIST_URL = ("https://huggingface.co/datasets/Royc30ne/emnist-balanced/resolve/"
              "e6fd3608dc5d46960a89a423472f67c9c0f9ec31/")
HAR_URL = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
FASHION = {
    "train-images-idx3-ubyte.gz": "3aede38d61863908ad78613f6a32ed271626dd12800ba2636569512369268a84",
    "train-labels-idx1-ubyte.gz": "a04f17134ac03560a47e3764e11b92fc97de4d1bfaf8ba1a3aa29af54cc90845",
    "t10k-images-idx3-ubyte.gz": "346e55b948d973a97e58d2351dde16a484bd415d4595297633bb08f03db6a073",
    "t10k-labels-idx1-ubyte.gz": "67da17c76eaffca5446c3361aaab5c3cd6d1c2608764d35dfb1850b086bf8dd5",
}
EMNIST = {
    "emnist-balanced-train-images-idx3-ubyte.gz": "74295e75e4e9fa2c102147460ee7eb763653d5132d883a84fb88932bfb8d2db4",
    "emnist-balanced-train-labels-idx1-ubyte.gz": "b46d41e5156a0a8aaaa4d18979fdd1588850fa8571018acffd5d93b325a7d6ed",
    "emnist-balanced-test-images-idx3-ubyte.gz": "117fc12b015e37ea98bfb2cbba3f77c3782c056cdbd219b87abfc8661d7135db",
    "emnist-balanced-test-labels-idx1-ubyte.gz": "a67cac948379b67270b9a4d0a01c8ec661598accf2f2b1964340b7c83cb175cd",
}
HAR_HASH = "c00b803081a5c797cd5e4b83700a9810b38d53d9d84e01917e090e1fdbc81031"
SEEDS = {
    "fashion_mnist": tuple(range(1111201, 1111206)),
    "emnist_balanced": tuple(range(1211201, 1211206)),
    "har": tuple(range(921201, 921206)),
}


def fetch(path, url, digest):
    if path.exists() and file_sha256(path) == digest:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {path.name}", flush=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            with urlopen(Request(url, headers={"User-Agent": "FedReTrust/1.0"}), timeout=90) as response:
                shutil.copyfileobj(response, handle)
            handle.flush()
        except BaseException:
            handle.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        if file_sha256(temporary) != digest:
            raise ValueError(f"Unexpected download checksum for {path.name}: {url}")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def read_idx(images, labels, transpose):
    with gzip.open(images, "rb") as handle:
        magic, n, h, w = struct.unpack(">IIII", handle.read(16))
        if (magic, h, w) != (2051, 28, 28):
            raise ValueError(f"Invalid IDX image header: {images}")
        x = np.frombuffer(handle.read(), dtype=np.uint8).reshape(n, h, w)
    with gzip.open(labels, "rb") as handle:
        if struct.unpack(">II", handle.read(8)) != (2049, n):
            raise ValueError(f"Invalid IDX label header: {labels}")
        y = np.frombuffer(handle.read(), dtype=np.uint8).astype(np.int64)
    return (x.transpose(0, 2, 1) if transpose else x), y


def balanced(labels, seed, count, pool=None):
    rng = np.random.default_rng(seed)
    pool = np.arange(len(labels)) if pool is None else np.asarray(pool)
    return np.sort(np.concatenate([
        rng.choice(pool[labels[pool] == label], count, replace=False)
        for label in np.unique(labels)
    ]))


def load_images(dataset):
    is_emnist = dataset == "emnist_balanced"
    sources, url = (EMNIST, EMNIST_URL) if is_emnist else (FASHION, FASHION_URL)
    files = {name: fetch(DATA / dataset / name, url + name, digest)
             for name, digest in sources.items()}

    def split(prefix):
        return read_idx(files[f"{prefix}-images-idx3-ubyte.gz"],
                        files[f"{prefix}-labels-idx1-ubyte.gz"], is_emnist)

    return (split("emnist-balanced-train"), split("emnist-balanced-test")) if is_emnist else (
        split("train"), split("t10k"))


def image_plan(dataset, loaded, seed):
    (x, labels), (test_x, test_y) = loaded
    classes = 47 if dataset == "emnist_balanced" else 10
    is_emnist = classes == 47
    dev = balanced(labels, seed_for(seed, "development_split"), 20 if is_emnist else 100)
    pool = np.setdiff1d(np.arange(len(labels)), dev)
    populations, _ = prepare_label_skew(
        labels, {str(i): part.tolist() for i, part in enumerate(np.split(pool, 10))},
        LabelSkewConfig(partition_seed=seed_for(seed, "partition"), num_classes=classes),
    )
    clients = {int(key): np.sort(np.random.default_rng(seed_for(seed, "subset", key))
                                .choice(indices, 512, replace=False))
               for key, indices in populations.items()}
    root = balanced(labels, seed_for(seed, "publication_root"), 7 if is_emnist else 30,
                    np.setdiff1d(pool, np.concatenate(list(clients.values()))))
    # Accumulate in float64 without materializing the full float training split.
    sums, squares = 0.0, 0.0
    for chunk in np.array_split(pool, 64):
        values = x[chunk].astype(np.float32) / 255
        sums += values.sum(dtype=np.float64)
        squares += np.square(values).sum(dtype=np.float64)
    count = len(pool) * 28 * 28
    mu = sums / count
    std = math.sqrt(squares / count - mu**2)
    test_ids = balanced(test_y, 1211199 if is_emnist else 1111199, 40 if is_emnist else 100)

    def view(images, indices):
        pixels = torch.from_numpy(images[indices].copy()).unsqueeze(1).float()
        shifted = pixels.clone()
        shifted[..., :7] = 255 - pixels[..., :7]
        shifted[..., -7:] = 255 - pixels[..., -7:]
        return tuple((v / 255 - mu) / std for v in (pixels, shifted))

    train_ids = np.concatenate(list(clients.values()))
    train, drift = view(x, train_ids)
    test, shifted_test = view(test_x, test_ids)
    root_x, _ = view(x, root)
    populations_counts = {i: np.bincount(labels[populations[str(i)]], minlength=classes)
                          for i in clients}
    permutation = torch.arange(classes - 1, -1, -1)
    if is_emnist:
        middle = classes // 2
        permutation[middle - 1:middle + 2] = torch.tensor([middle, middle + 1, middle - 1])
    return assemble(
        seed, classes, lambda: MnistCNN(num_classes=classes), .01,
        train, drift, labels[train_ids], clients, (0, 1, 2), populations_counts,
        test, shifted_test, test_y[test_ids], root_x, labels[root], permutation,
        {"client_indices": {str(i): ids.tolist() for i, ids in clients.items()},
         "root_indices": root.tolist(), "evaluation_indices": test_ids.tolist(),
         "normalization": {"mean": mu, "std": std}},
    )


class HarCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(6, 16, 7, padding=3), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(16, 32, 5, padding=2), nn.ReLU(), nn.AvgPool1d(4),
            nn.Flatten(), nn.Linear(32 * 8, 32), nn.ReLU(), nn.Linear(32, 6),
        )

    def forward(self, signals):
        return self.network(signals)


def load_har():
    directory = DATA / "har"
    path = fetch(directory / "uci-har.zip", HAR_URL, HAR_HASH)

    def load(archive, split):
        base = f"UCI HAR Dataset/{split}"
        streams = []
        for sensor in ("total_acc", "body_gyro"):
            for axis in "xyz":
                with archive.open(f"{base}/Inertial Signals/{sensor}_{axis}_{split}.txt") as file:
                    streams.append(np.loadtxt(file, dtype=np.float32))
        with archive.open(f"{base}/y_{split}.txt") as file:
            labels = np.loadtxt(file, dtype=np.int64) - 1
        with archive.open(f"{base}/subject_{split}.txt") as file:
            subjects = np.loadtxt(file, dtype=np.int64)
        return torch.from_numpy(np.stack(streams, axis=1)), labels, subjects

    with ZipFile(path) as outer:
        with ZipFile(io.BytesIO(outer.read("UCI HAR Dataset.zip"))) as inner:
            return load(inner, "train"), load(inner, "test")


def har_plan(loaded, seed):
    (x, labels, subjects), (test_x, test_y, test_subjects) = loaded
    all_subjects = sorted(set(map(int, subjects)))
    root_pairs = [pair for pair in combinations(all_subjects, 2)
                  if min(np.bincount(labels[np.isin(subjects, pair)], minlength=6)) >= 50]
    root_rng = np.random.default_rng(seed_for(seed, "publication_root_subjects"))
    root_subjects = root_pairs[int(root_rng.integers(len(root_pairs)))]
    root = balanced(labels, seed_for(seed, "publication_root"), 50,
                    np.flatnonzero(np.isin(subjects, root_subjects)))
    clients = {i: np.flatnonzero(subjects == i) for i in all_subjects if i not in root_subjects}
    if set(test_subjects) & set(all_subjects) or len(clients) != 19:
        raise ValueError("Unexpected HAR participant split.")
    affected = tuple(sorted(map(int, np.random.default_rng(seed_for(seed, "subject_selection"))
                               .choice(list(clients), 6, replace=False))))
    train_ids = np.concatenate(list(clients.values()))
    train_x = x[train_ids]
    std, mu = torch.std_mean(train_x, dim=(0, 2), correction=0)
    angle = math.radians(30)
    c, s = math.cos(angle), math.sin(angle)
    rotation = x.new_tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    def normal(signals):
        return (signals - mu.view(1, -1, 1)) / std.view(1, -1, 1)

    def shifted(signals):
        return normal(torch.einsum("ij,nvjt->nvit", rotation,
                                   signals.reshape(-1, 2, 3, 128)).reshape_as(signals))

    return assemble(
        seed, 6, HarCNN, .03, normal(train_x), shifted(train_x), labels[train_ids],
        clients, affected, {i: np.bincount(labels[ids], minlength=6) for i, ids in clients.items()},
        normal(test_x), shifted(test_x), test_y, normal(x[root]), labels[root],
        torch.arange(5, -1, -1),
        {"client_indices": {str(i): ids.tolist() for i, ids in clients.items()},
         "root_indices": root.tolist(), "root_subjects": list(root_subjects),
         "evaluation_subjects": sorted(set(map(int, test_subjects))),
         "normalization": {"mean": mu.tolist(), "std": std.tolist()}},
    )


def assemble(seed, classes, model, lr, train, drift, labels, clients, affected, counts,
             test, shifted_test, test_y, root, root_y, permutation, plan):
    train_ids = np.concatenate(list(clients.values()))
    if len(set(train_ids)) != len(train_ids) or set(train_ids) & set(plan["root_indices"]):
        raise ValueError("Client samples and clean server reference must be disjoint.")
    if len(set(root_y)) != classes or len(set(test_y)) != classes:
        raise ValueError("Reference and test sets must cover all classes.")
    weights = np.array([counts[i] / counts[i].sum() for i in affected]).mean(axis=0)
    rows = {}
    start = 0
    for client, ids in clients.items():
        stop = start + len(ids)
        rng = np.random.default_rng(seed_for(seed, "drift_order", client))
        rows[client] = (train[start:stop], drift[start:stop], torch.from_numpy(labels[start:stop]),
                        torch.from_numpy(rng.permutation(len(ids))))
        start = stop
    return {
        "model": model, "lr": lr, "clients": rows, "affected": affected,
        "test": (test, shifted_test, torch.from_numpy(test_y)), "classes": classes,
        "weights": weights, "root": (root, torch.from_numpy(root_y)),
        "permutation": permutation,
        "plan": {**plan, "seed": seed, "affected_clients": list(affected),
                 "class_profile_weights": weights.tolist(), "test_size": len(test_y)},
    }
