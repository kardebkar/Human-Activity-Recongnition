#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import socket
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from har.streaming.features import extract_features, feature_names
from har.streaming.protocol import parse_payload
from har.streaming.sklearn_infer import SklearnWindowClassifier
from har.streaming.stabilizer import StabilizerConfig, StateStabilizer
from har.streaming.windowing import SlidingImuWindow


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Receive Apple Watch IMU (UDP JSON), optionally classify, and output a stabilized state stream.")
    parser.add_argument(
        "--stdin",
        action="store_true",
        help="Read JSON payloads (one per line) from stdin instead of binding a UDP socket (useful for local smoke tests).",
    )
    parser.add_argument("--listen-host", default="0.0.0.0")
    parser.add_argument("--listen-port", type=int, default=5500)

    parser.add_argument("--sample-rate-hz", type=float, default=50.0, help="Fallback sample rate if payload lacks 'sr'.")
    parser.add_argument("--window-seconds", type=float, default=2.0)
    parser.add_argument("--hop-seconds", type=float, default=0.5)

    parser.add_argument("--model-path", default=None, help="Joblib model path from watch/mac/train_imu_model.py")
    parser.add_argument("--min-confidence", type=float, default=0.6)
    parser.add_argument("--k-consecutive", type=int, default=3)
    parser.add_argument("--lockout-seconds", type=float, default=1.0)

    parser.add_argument("--log-raw-csv", default=None, help="Optional CSV to write raw samples (t, ax..gz, label).")
    parser.add_argument("--log-windows-jsonl", default=None, help="Optional JSONL to write per-window features + predictions.")
    parser.add_argument("--print-probabilities", action="store_true", help="Print full probability dict per window.")

    parser.add_argument("--unity-host", default=None, help="Optional Unity UDP host for state output.")
    parser.add_argument("--unity-port", type=int, default=0, help="Optional Unity UDP port for state output.")
    parser.add_argument("--quiet", action="store_true", help="Reduce console output.")
    return parser.parse_args()


def _seconds_now() -> float:
    return time.time()


def main() -> int:
    args = _parse_args()

    window_size = int(round(args.window_seconds * args.sample_rate_hz))
    hop_size = int(round(args.hop_seconds * args.sample_rate_hz))
    if window_size <= 0 or hop_size <= 0:
        raise SystemExit("window-seconds and hop-seconds must be > 0.")

    windowing = SlidingImuWindow(window_size=window_size, hop_size=hop_size)

    classifier: SklearnWindowClassifier | None = None
    if args.model_path:
        classifier = SklearnWindowClassifier.load_joblib(args.model_path)
        # Soft check: feature count must match what training produced.
        expected = classifier.expected_feature_names
        if expected is not None and len(expected) != len(feature_names()):
            raise SystemExit(
                f"Model expects {len(expected)} features, but this server extracts {len(feature_names())}. "
                "Re-train the model with the current feature extractor."
            )

    stabilizer = StateStabilizer(
        config=StabilizerConfig(
            min_confidence=args.min_confidence,
            k_consecutive=args.k_consecutive,
            lockout_seconds=args.lockout_seconds,
        )
    )

    raw_writer = None
    raw_fh = None
    if args.log_raw_csv:
        raw_path = Path(args.log_raw_csv)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_fh = raw_path.open("w", newline="", encoding="utf-8")
        raw_writer = csv.DictWriter(raw_fh, fieldnames=["t", "ax", "ay", "az", "gx", "gy", "gz", "label"])
        raw_writer.writeheader()

    win_fh = None
    if args.log_windows_jsonl:
        win_path = Path(args.log_windows_jsonl)
        win_path.parent.mkdir(parents=True, exist_ok=True)
        win_fh = win_path.open("w", encoding="utf-8")

    unity_sock = None
    unity_addr = None
    if args.unity_host and args.unity_port:
        unity_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        unity_addr = (args.unity_host, int(args.unity_port))

    sock = None
    if not args.stdin:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((args.listen_host, int(args.listen_port)))

    if not args.quiet:
        if args.stdin:
            print("Input: stdin (one JSON payload per line)")
        else:
            print(f"Listening UDP on {args.listen_host}:{args.listen_port}")
        print(f"Window: {args.window_seconds:.2f}s ({window_size} samples) | Hop: {args.hop_seconds:.2f}s ({hop_size} samples)")
        if classifier:
            print(f"Model: {args.model_path}")
        else:
            print("Model: (none) — raw collection only")
        if args.log_raw_csv:
            print(f"Raw CSV: {args.log_raw_csv}")
        if args.log_windows_jsonl:
            print(f"Window JSONL: {args.log_windows_jsonl}")
        if unity_addr:
            print(f"Unity UDP: {unity_addr[0]}:{unity_addr[1]}")
        print()

    try:
        while True:
            if args.stdin:
                line = sys.stdin.readline()
                if not line:
                    break
                payload = line.encode("utf-8")
            else:
                assert sock is not None
                payload, _addr = sock.recvfrom(65535)
            try:
                samples = parse_payload(payload, default_sample_rate_hz=args.sample_rate_hz)
            except Exception as e:
                if not args.quiet:
                    print(f"⚠️ Bad payload: {e}")
                continue

            for s in samples:
                if raw_writer is not None:
                    raw_writer.writerow(
                        {
                            "t": "" if s.t is None else f"{s.t:.6f}",
                            "ax": s.ax,
                            "ay": s.ay,
                            "az": s.az,
                            "gx": s.gx,
                            "gy": s.gy,
                            "gz": s.gz,
                            "label": "" if s.label is None else s.label,
                        }
                    )

                windows = windowing.append(s)
                if not windows:
                    continue

                for w in windows:
                    feats = extract_features(w.x)
                    out: dict[str, object] = {
                        "t_end": w.t_end if w.t_end is not None else _seconds_now(),
                    }

                    if classifier is None:
                        # No inference: emit only timing.
                        if not args.quiet:
                            print(json.dumps(out))
                        continue

                    pred = classifier.predict(feats)
                    stable = stabilizer.update(pred.label, pred.confidence, t=float(out["t_end"]))

                    out.update(
                        {
                            "predicted": pred.label,
                            "confidence": pred.confidence,
                            "stable": stable,
                        }
                    )
                    if args.print_probabilities:
                        out["probabilities"] = pred.probabilities

                    if win_fh is not None:
                        win_fh.write(
                            json.dumps(
                                {
                                    **out,
                                    "features": feats.tolist(),
                                }
                            )
                            + "\n"
                        )
                        win_fh.flush()

                    if not args.quiet:
                        print(json.dumps(out))

                    if unity_sock is not None and unity_addr is not None:
                        unity_sock.sendto(json.dumps(out).encode("utf-8"), unity_addr)

            if raw_fh is not None:
                raw_fh.flush()

    except KeyboardInterrupt:
        if not args.quiet:
            print("\nStopping…")
        return 0
    finally:
        try:
            if sock is not None:
                sock.close()
        except Exception:
            pass
        if unity_sock is not None:
            try:
                unity_sock.close()
            except Exception:
                pass
        if raw_fh is not None:
            raw_fh.close()
        if win_fh is not None:
            win_fh.close()


if __name__ == "__main__":
    raise SystemExit(main())
