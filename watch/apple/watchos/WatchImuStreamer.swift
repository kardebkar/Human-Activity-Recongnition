import Foundation
import CoreMotion
import WatchConnectivity
import WatchKit

final class WatchImuStreamer: NSObject, ObservableObject {
    @Published var isStreaming: Bool = false
    @Published var samplesSent: Int = 0
    @Published var lastError: String? = nil
    @Published var currentLabel: String = "UNLABELED"

    private let motionManager = CMMotionManager()
    private let motionQueue = OperationQueue()
    private var extendedSession: WKExtendedRuntimeSession?

    private var buffer: [ImuSample] = []
    private var lastSend: Date = .distantPast

    let sampleRateHz: Double = 50.0
    let batchIntervalSeconds: TimeInterval = 0.25
    let maxBatchSize: Int = 25

    override init() {
        super.init()
        motionQueue.qualityOfService = .userInitiated

        if WCSession.isSupported() {
            WCSession.default.delegate = self
            WCSession.default.activate()
        }
    }

    func start() {
        guard !isStreaming else { return }

        lastError = nil
        isStreaming = true
        buffer.removeAll(keepingCapacity: true)
        lastSend = .distantPast

        // Extended runtime helps keep the app running longer while streaming.
        let session = WKExtendedRuntimeSession()
        session.delegate = self
        session.start()
        extendedSession = session

        guard motionManager.isDeviceMotionAvailable else {
            lastError = "DeviceMotion not available on this watch."
            isStreaming = false
            return
        }

        motionManager.deviceMotionUpdateInterval = 1.0 / sampleRateHz
        motionManager.startDeviceMotionUpdates(to: motionQueue) { [weak self] motion, error in
            guard let self else { return }
            if let error {
                DispatchQueue.main.async { self.lastError = error.localizedDescription }
                return
            }
            guard let motion else { return }

            // Use acceleration including gravity (gravity + userAcceleration).
            let t = Date().timeIntervalSince1970
            let acc = motion.userAcceleration
            let grav = motion.gravity
            let rot = motion.rotationRate

            let sample = ImuSample(
                t: t,
                sr: self.sampleRateHz,
                ax: acc.x + grav.x,
                ay: acc.y + grav.y,
                az: acc.z + grav.z,
                gx: rot.x,
                gy: rot.y,
                gz: rot.z,
                label: self.currentLabel == "UNLABELED" ? nil : self.currentLabel
            )

            self.buffer.append(sample)
            if !WCSession.default.isReachable, self.buffer.count > self.maxBatchSize {
                self.buffer.removeFirst(self.buffer.count - self.maxBatchSize)
            }
            self.flushIfNeeded(now: Date())
        }
    }

    func stop() {
        guard isStreaming else { return }
        isStreaming = false
        motionManager.stopDeviceMotionUpdates()
        flush(force: true)
        extendedSession?.invalidate()
        extendedSession = nil
    }

    private func flushIfNeeded(now: Date) {
        guard buffer.count >= maxBatchSize || now.timeIntervalSince(lastSend) >= batchIntervalSeconds else { return }
        flush(force: false)
    }

    private func flush(force: Bool) {
        guard !buffer.isEmpty else { return }
        guard WCSession.default.isReachable else {
            if force {
                buffer.removeAll(keepingCapacity: true)
            }
            return
        }

        let batch = ImuBatch(sr: sampleRateHz, samples: buffer)
        buffer.removeAll(keepingCapacity: true)
        lastSend = Date()

        do {
            let data = try JSONEncoder().encode(batch)
            WCSession.default.sendMessageData(data, replyHandler: nil) { [weak self] error in
                DispatchQueue.main.async { self?.lastError = error.localizedDescription }
            }
            DispatchQueue.main.async { self.samplesSent += batch.samples.count }
        } catch {
            DispatchQueue.main.async { self.lastError = error.localizedDescription }
        }
    }
}

extension WatchImuStreamer: WCSessionDelegate {
    func session(_ session: WCSession, activationDidCompleteWith activationState: WCSessionActivationState, error: (any Error)?) {
        if let error {
            DispatchQueue.main.async { self.lastError = error.localizedDescription }
        }
    }

    func session(_ session: WCSession, didReceiveMessage message: [String: Any]) {
        if let label = message["label"] as? String {
            DispatchQueue.main.async { self.currentLabel = label }
        }
        if let cmd = message["cmd"] as? String {
            if cmd == "start" {
                DispatchQueue.main.async { self.start() }
            } else if cmd == "stop" {
                DispatchQueue.main.async { self.stop() }
            }
        }
    }
}

extension WatchImuStreamer: WKExtendedRuntimeSessionDelegate {
    func extendedRuntimeSessionDidStart(_ extendedRuntimeSession: WKExtendedRuntimeSession) {}

    func extendedRuntimeSessionWillExpire(_ extendedRuntimeSession: WKExtendedRuntimeSession) {
        DispatchQueue.main.async { self.lastError = "Extended runtime expiring; streaming may stop soon." }
    }

    func extendedRuntimeSession(_ extendedRuntimeSession: WKExtendedRuntimeSession, didInvalidateWith reason: WKExtendedRuntimeSessionInvalidationReason, error: (any Error)?) {
        DispatchQueue.main.async {
            if let error { self.lastError = error.localizedDescription }
            self.isStreaming = false
        }
    }
}
