from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ImuSample:
    t: float | None
    ax: float
    ay: float
    az: float
    gx: float
    gy: float
    gz: float
    sample_rate_hz: float | None = None
    label: str | None = None


def _normalize_timestamp(value: Any) -> float | None:
    if value is None:
        return None
    try:
        t = float(value)
    except Exception:
        return None

    # Heuristic: milliseconds vs seconds
    # - seconds since epoch ~ 1.7e9
    # - milliseconds since epoch ~ 1.7e12
    if t > 1e11:
        t = t / 1000.0
    return t


def _get_num(d: dict[str, Any], key: str) -> float | None:
    if key not in d:
        return None
    try:
        return float(d[key])
    except Exception:
        return None


def _parse_one(sample_dict: dict[str, Any], *, default_sr: float | None) -> ImuSample:
    t = _normalize_timestamp(sample_dict.get("t") or sample_dict.get("timestamp") or sample_dict.get("ts"))
    sr = _get_num(sample_dict, "sr") or _get_num(sample_dict, "sample_rate_hz") or default_sr

    ax = _get_num(sample_dict, "ax")
    ay = _get_num(sample_dict, "ay")
    az = _get_num(sample_dict, "az")
    gx = _get_num(sample_dict, "gx")
    gy = _get_num(sample_dict, "gy")
    gz = _get_num(sample_dict, "gz")

    missing = [k for k, v in (("ax", ax), ("ay", ay), ("az", az), ("gx", gx), ("gy", gy), ("gz", gz)) if v is None]
    if missing:
        raise ValueError(f"Missing/invalid IMU fields: {missing}. Expected keys ax/ay/az/gx/gy/gz.")

    label = sample_dict.get("label")
    if label is not None:
        label = str(label)

    return ImuSample(
        t=t,
        ax=float(ax),
        ay=float(ay),
        az=float(az),
        gx=float(gx),
        gy=float(gy),
        gz=float(gz),
        sample_rate_hz=sr,
        label=label,
    )


def parse_payload(payload: bytes, *, default_sample_rate_hz: float | None = None) -> list[ImuSample]:
    """Parse a UDP payload into one or more IMU samples.

    Supported JSON payload forms:
    1) Single sample dict: {"t":..., "ax":..., ..., "gz":...}
    2) Batch dict: {"sr": 50, "samples": [ {...}, {...} ]}
    3) Raw list: [ {...}, {...} ]
    """

    text = payload.decode("utf-8", errors="replace").strip()
    if not text:
        return []

    data = json.loads(text)
    if isinstance(data, list):
        return [_parse_one(d, default_sr=default_sample_rate_hz) for d in data]

    if isinstance(data, dict) and isinstance(data.get("samples"), list):
        default_sr = _get_num(data, "sr") or _get_num(data, "sample_rate_hz") or default_sample_rate_hz
        return [_parse_one(d, default_sr=default_sr) for d in data["samples"]]

    if isinstance(data, dict):
        return [_parse_one(data, default_sr=default_sample_rate_hz)]

    raise ValueError("Invalid JSON payload. Expected a dict or list.")

