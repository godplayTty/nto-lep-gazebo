#!/usr/bin/env python3
"""Compare captured /insulators data with the generated reference."""
import argparse
from itertools import combinations, permutations
import json
from math import hypot, isfinite
from pathlib import Path


def evaluate(reference, capture):
    expected = reference["towers"]
    rows = capture["insulators"]
    if capture["frame_id"] != reference["frame_id"]:
        raise ValueError("Разные системы координат")
    detected = {}
    entries = {}
    for row in rows:
        identity = row["tower_id"]
        position = (row["x"], row["y"])
        if not all(map(isfinite, position)):
            raise ValueError(f"Некорректные координаты опоры {identity}")
        if detected.setdefault(identity, position) != position:
            raise ValueError(f"Несогласованные координаты опоры {identity}")
        key = (identity, row["number"])
        if key in entries:
            raise ValueError(f"Повторяющаяся пара {key}")
        entries[key] = (row["color"], row["state"])

    def error(tower, identity):
        x, y = detected[identity]
        return hypot(tower["x"]-x, tower["y"]-y)

    best = []
    for size in range(min(len(detected), len(expected)), 0, -1):
        matches = []
        for towers in combinations(expected, size):
            for identities in permutations(detected, size):
                candidate = [(tower, identity, error(tower, identity))
                             for tower, identity in zip(towers, identities)]
                if all(distance <= .2 for _, _, distance in candidate):
                    matches.append(candidate)
        if matches:
            best = min(matches, key=lambda m: sum(distance for _, _, distance in m))
            break

    correct = 0
    for tower, identity, distance in best:
        hits = sum(entries.get((identity, i["number"])) == (i["color"], i["state"])
                   for i in tower["insulators"])
        correct += hits
        print(f"{identity} → {tower['id']}: {distance:.3f} м, изоляторы {hits}/4")
    print(f"Опоры {len(best)}/{len(expected)}, изоляторы {correct}/12")
    return len(best) == len(expected) and correct == 12


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("capture", type=Path)
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text())
    capture = json.loads(args.capture.read_text())
    raise SystemExit(0 if evaluate(reference, capture) else 1)


if __name__ == "__main__":
    main()
