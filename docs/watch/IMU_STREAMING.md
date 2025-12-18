# Apple Watch IMU → macOS Streaming (Raw) + Classification

Goal: stream **raw IMU** (accelerometer/gyroscope) from Apple Watch, do **classification + state stabilization on macOS**, and (later) drive a Unity avatar.

## Architecture (Recommended)

Apple Watch cannot reliably talk to a Mac directly in most setups, so use the iPhone as a relay:

1) **watchOS app** collects IMU via Core Motion and sends samples to the phone using **WatchConnectivity**  
2) **iOS relay app** forwards samples to the Mac over LAN (UDP recommended)  
3) **Mac (Python)** receives samples, runs windowing + ML inference + stabilization, and outputs a clean state stream

## Message Format (UDP Payload)

Send one JSON object per UDP datagram, either a single sample:

```json
{"t": 1734450000.123, "sr": 50, "ax": 0.01, "ay": -0.02, "az": 0.98, "gx": 0.1, "gy": -0.3, "gz": 0.05}
```

…or a batch (preferred to reduce overhead):

```json
{"sr": 50, "samples": [{"t": 1734450000.123, "ax": 0.01, "ay": -0.02, "az": 0.98, "gx": 0.1, "gy": -0.3, "gz": 0.05}]}
```

Notes:
- `t` can be seconds (float) or milliseconds (int); the Mac script normalizes it.
- Units depend on what you send (Core Motion uses G for accel; rad/s for gyro).
- You can include an optional `label` field while collecting training data.

## Mac Side (Receiver + Classifier)

### 1) Raw collection (no model)

```bash
python watch/mac/run_stream_server.py \
  --listen-port 5500 \
  --sample-rate-hz 50 \
  --window-seconds 2.0 \
  --hop-seconds 0.5 \
  --log-raw-csv data/watch_raw.csv
```

### 2) Real-time inference (with a trained model)

```bash
python watch/mac/run_stream_server.py \
  --listen-port 5500 \
  --sample-rate-hz 50 \
  --window-seconds 2.0 \
  --hop-seconds 0.5 \
  --model-path models/watch_imu_rf.joblib
```

## Training a Model on Collected Raw IMU

After collecting a labeled CSV, train a baseline:

```bash
python watch/mac/train_imu_model.py \
  --raw-csv data/watch_raw_labeled.csv \
  --sample-rate-hz 50 \
  --window-seconds 2.0 \
  --hop-seconds 0.5 \
  --model rf
```

This writes:
- `models/watch_imu_rf.joblib`
- `outputs/watch_imu_confusion_matrix.png`
- `outputs/watch_imu_classification_report.json`

## watchOS Collection (Swift Outline)

You need:
- `CoreMotion` for IMU (`CMMotionManager` or `CMDeviceMotion`)
- a long-running mode (recommended): start an `HKWorkoutSession` to prevent suspension
- `WatchConnectivity` (`WCSession`) to send samples to the phone

Pseudo-outline:

1) Start a workout session on the watch.
2) Start `deviceMotion` updates at 25–50 Hz.
3) Buffer samples and send batches to the phone every ~200–500ms.

## iOS Relay (Swift Outline)

1) Receive data from the watch via `WCSession` delegate.
2) Forward to Mac via UDP using `Network` (`NWConnection` with `.udp`).
3) A simple UI stores the Mac IP/port and shows send stats.

Practical tip: batch samples before sending over WatchConnectivity to avoid message overhead.

## Install + Smoke Test (No Watch Required)

Install Python deps (from repo root):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Smoke test the parsing + windowing path **without UDP**:

```bash
python watch/mac/generate_fake_imu.py --samples 200 --sample-rate-hz 50 \
  | python watch/mac/run_stream_server.py --stdin --sample-rate-hz 50 --window-seconds 2.0 --hop-seconds 0.5
```
