# Apple Watch → iPhone → Mac IMU Streamer (Swift Scaffold)

This folder contains **Swift source scaffolding** for a simple Apple Watch + iPhone relay:

- watchOS: collect **raw IMU** (accel+gravity, gyro) via Core Motion and batch-send to iPhone via WatchConnectivity
- iOS: receive batches and forward them to a Mac over **UDP** on the local network

It is designed to work with the Mac receiver in this repo:
- `watch/mac/run_stream_server.py`

## How To Use

1) Follow `watch/apple/SETUP_XCODE.md` to create an Xcode project (iOS app with Watch app) and add these files.
   - Or open the included sample project: `watch/xcode/Wrist2AvatarStreamer/Wrist2AvatarStreamer.xcodeproj`
2) On the Mac, start the receiver (choose a port, example 5500):

```bash
python watch/mac/run_stream_server.py --listen-port 5500 --log-raw-csv data/watch_raw.csv
```

3) On the iPhone app, enter the Mac’s IP and port, start relay.
4) On the watch app, press Start to begin streaming.

## Payload Format

The iPhone forwards exactly what it receives from the watch as JSON (one UDP datagram per batch):

```json
{"sr":50,"samples":[{"t":1734450000.123,"sr":50,"ax":0.01,"ay":-0.02,"az":0.98,"gx":0.1,"gy":-0.3,"gz":0.05,"label":"SITTING"}]}
```

This matches the parser in `har/streaming/protocol.py`.
