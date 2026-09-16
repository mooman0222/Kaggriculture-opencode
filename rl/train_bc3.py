"""Train Policy3 joint (destination, operation) options from existing Policy2 shards."""
import argparse
import glob
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

import model3
from model3 import Policy3, bc_loss3
from train_bc2 import KEYS, load_shards


def to_tensors(data, indices, device):
    return {
        key: torch.from_numpy(data[key][indices].astype(np.int16) if key in ("tiles", "units") else data[key][indices]).to(device)
        for key in KEYS
    }


def load_compatible(model, path, device):
    source = torch.load(path, map_location=device)
    target = model.state_dict()
    compatible = {key: value for key, value in source.items() if key in target and target[key].shape == value.shape}
    model.load_state_dict(compatible, strict=False)
    print(f"initialized {len(compatible)}/{len(target)} tensors from {path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--bs", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--val", type=float, default=0.1)
    parser.add_argument("--max-games", type=int, default=100000)
    parser.add_argument("--init")
    parser.add_argument("--pass-idle-weight", type=float, default=1.0, help="weight of PASS labels recorded while an unwatered plant / unfed animal remained (bc11: 0.1)")
    args = parser.parse_args()

    files = sorted(glob.glob(args.data, recursive=True))
    random.Random(0).shuffle(files)
    files = files[:args.max_games]
    if len(files) < 2:
        raise ValueError("at least two shard files are required")
    validation_count = max(1, int(len(files) * args.val))
    validation_files, training_files = files[:validation_count], files[validation_count:]
    started = time.time()
    training = load_shards(training_files)
    validation = load_shards(validation_files)
    print(
        f"games train {len(training_files)} val {len(validation_files)}; "
        f"steps train {len(training['dest'])} val {len(validation['dest'])} [{time.time() - started:.0f}s]",
        flush=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    operations = training["dop"][training["dop"] >= 0]
    frequency = np.bincount(operations, minlength=model3.N_OPS).astype(np.float64) + 10
    weights = (frequency.mean() / frequency) ** 0.5
    model3.OP_WEIGHT = torch.tensor(weights / weights.mean(), dtype=torch.float32)
    model3.PASS_IDLE_WEIGHT = args.pass_idle_weight
    model = Policy3(d=args.d, layers=args.layers).to(device)
    if args.init:
        load_compatible(model, args.init, device)
    print("params", sum(parameter.numel() for parameter in model.parameters()), "device", device, flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps_per_epoch = math.ceil(len(training["dest"]) / args.bs)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, args.lr, total_steps=args.epochs * steps_per_epoch)
    for epoch in range(args.epochs):
        model.train()
        permutation = np.random.RandomState(2000 + epoch).permutation(len(training["dest"]))
        total_loss = 0.0
        batches = 0
        epoch_started = time.time()
        for start in range(0, len(permutation), args.bs):
            indices = permutation[start:start + args.bs]
            batch = to_tensors(training, indices, device)
            output = model(batch["tiles"], batch["units"], batch["items"], batch["glob"], dest=batch["dest"].long().clamp(min=0))
            loss, accuracy = bc_loss3(output, batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            batches += 1
            if batches % 200 == 1:
                print(f"  ep {epoch} b {batches}/{steps_per_epoch} loss {loss.item():.3f} acc {accuracy}", flush=True)

        model.eval()
        validation_loss = 0.0
        validation_steps = 0
        accuracies = []
        with torch.no_grad():
            for start in range(0, len(validation["dest"]), args.bs):
                indices = np.arange(start, min(start + args.bs, len(validation["dest"])))
                batch = to_tensors(validation, indices, device)
                output = model(batch["tiles"], batch["units"], batch["items"], batch["glob"], dest=batch["dest"].long().clamp(min=0))
                loss, accuracy = bc_loss3(output, batch)
                validation_loss += loss.item() * len(indices)
                validation_steps += len(indices)
                accuracies.append(accuracy)
        mean_accuracy = {key: round(sum(row[key] for row in accuracies) / len(accuracies), 3) for key in accuracies[0]}
        print(
            f"epoch {epoch} train loss {total_loss / batches:.3f} val loss {validation_loss / validation_steps:.3f} "
            f"val acc {mean_accuracy} [{time.time() - epoch_started:.0f}s]",
            flush=True,
        )
        torch.save(model.state_dict(), args.out)
        torch.save(model.state_dict(), args.out.replace(".pt", f"_ep{epoch}.pt"))


if __name__ == "__main__":
    main()