# Xcode Setup (iOS + watchOS) — Raw IMU Streaming

This repo does not commit an `.xcodeproj`. Instead, create the project in Xcode and add the Swift files from `watch/apple/`.

If you prefer to start from a working example project (already set up with iOS + watch targets), open:
- `watch/xcode/Wrist2AvatarStreamer/Wrist2AvatarStreamer.xcodeproj`

## Prereqs

- Xcode (recent; watchOS + iOS SDK installed)
- iPhone paired with your Apple Watch
- Both devices on the **same Wi‑Fi** as your Mac (for UDP relay)

## 1) Create the Project

1. Xcode → **File → New → Project**
2. Choose **iOS → App**
3. Product name: e.g. `Wrist2AvatarStreamer`
4. Interface: **SwiftUI**, Language: **Swift**
5. If you see **“Include a Watch App”**, enable it.

If your Xcode version does **not** show that checkbox (common in newer Xcode versions), do this instead:

1. Create the iOS app project first (steps above).
2. Then: Xcode → **File → New → Target…**
3. Pick **watchOS → App** (or “Watch App for iOS App”, depending on templates available).
4. Name it (e.g. `Wrist2AvatarWatch`) and finish the wizard.

If you don’t see watchOS templates at all:
- Xcode → **Settings… → Platforms** → install/download **watchOS**.

## 2) Add Capabilities / Permissions

### watchOS target
- Add **WatchConnectivity**
- Add `NSMotionUsageDescription` to the watch target’s Info.plist (example):
  - “This app streams motion sensor data for a research demo.”

### iOS target
- Add **WatchConnectivity**
- Add `NSLocalNetworkUsageDescription` to the iOS target’s Info.plist (example):
  - “This app relays watch motion data to a Mac on your local network.”

## 3) Add the Source Files

In Xcode, add these files to the correct targets:

**Shared (both targets)**
- `watch/apple/shared/ImuTypes.swift`

**watchOS target**
- `watch/apple/watchos/WatchImuStreamer.swift`
- `watch/apple/watchos/WatchApp.swift`
- `watch/apple/watchos/WatchContentView.swift`

**iOS target**
- `watch/apple/ios/MacUdpRelay.swift`
- `watch/apple/ios/iOSApp.swift` *(only if you remove the template’s generated `@main` App file)*
- `watch/apple/ios/iOSContentView.swift` *(or copy its contents into your generated `ContentView.swift`)*

Important:
- Each target must have **exactly one** `@main` SwiftUI `App` entry point. If you keep the template-generated `YourAppApp.swift`, do **not** also add `watch/apple/ios/iOSApp.swift` to the iOS target (same for watch).

## 4) Run It End-to-End

1) Start the Mac receiver:

```bash
python watch/mac/run_stream_server.py --listen-port 5500 --log-raw-csv data/watch_raw.csv
```

2) Run the iOS app on your iPhone (Xcode Run).
3) Enter:
- Mac IP (e.g. `192.168.1.25`)
- Port `5500`
4) Tap **Start Relay**.
5) Run the watch app, tap **Start**.

If macOS firewall prompts, allow incoming UDP for Python.

## 4b) Running on Simulator (What to Expect)

- You can run the watch UI on the **Apple Watch Simulator**, but **CoreMotion IMU is typically unavailable** in the watch simulator.
- The watch app may show an error like “DeviceMotion not available” in simulator; that’s expected.
- For a full end-to-end demo without hardware, use the Mac smoke test in `docs/watch/IMU_STREAMING.md`.

## 5) Labeling (Optional, Recommended for Training)

The iOS UI includes a label picker. When you pick a label, the phone sends it to the watch; the watch attaches it to each sample as `label`.

That produces a CSV suitable for:

```bash
python watch/mac/train_imu_model.py --raw-csv data/watch_raw_labeled.csv --model rf
```
