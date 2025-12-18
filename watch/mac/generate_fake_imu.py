#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate fake IMU JSON samples (for local smoke tests).")
    p.add_argument("--samples", type=int, default=200)
    p.add_argument("--sample-rate-hz", type=float, default=50.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--label", default=None, help="Optional label field to include in each sample.")
    p.add_argument("--realtime", action="store_true", help="Sleep between samples to simulate real-time streaming.")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    random.seed(args.seed)

    sr = float(args.sample_rate_hz)
    if sr <= 0:
        raise SystemExit("--sample-rate-hz must be > 0")
    dt = 1.0 / sr

    t0 = time.time()
    for i in range(int(args.samples)):
        t = t0 + i * dt

        # Simple, plausible motion: gravity on z + small oscillations + noise.
        ax = 0.08 * math.sin(2.0 * math.pi * 1.2 * (i * dt)) + random.uniform(-0.01, 0.01)
        ay = 0.06 * math.sin(2.0 * math.pi * 0.8 * (i * dt + 0.1)) + random.uniform(-0.01, 0.01)
        az = 1.0 + 0.03 * math.sin(2.0 * math.pi * 1.0 * (i * dt + 0.2)) + random.uniform(-0.01, 0.01)

        gx = 0.10 * math.sin(2.0 * math.pi * 0.7 * (i * dt)) + random.uniform(-0.02, 0.02)
        gy = 0.12 * math.sin(2.0 * math.pi * 1.4 * (i * dt + 0.2)) + random.uniform(-0.02, 0.02)
        gz = 0.05 * math.sin(2.0 * math.pi * 0.9 * (i * dt + 0.3)) + random.uniform(-0.02, 0.02)

        msg: dict[str, object] = {
            "t": t,
            "sr": sr,
            "ax": ax,
            "ay": ay,
            "az": az,
            "gx": gx,
            "gy": gy,
            "gz": gz,
        }
        if args.label is not None:
            msg["label"] = str(args.label)

        sys.stdout.write(json.dumps(msg) + "\n")
        sys.stdout.flush()
        if args.realtime:
            time.sleep(dt)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

