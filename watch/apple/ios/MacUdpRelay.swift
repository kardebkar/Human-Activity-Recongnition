import Foundation
import Network
import WatchConnectivity

final class MacUdpRelay: NSObject, ObservableObject {
    @Published var isRelaying: Bool = false
    @Published var packetsSent: Int = 0
    @Published var lastError: String? = nil

    @Published var macHost: String = ""
    @Published var macPort: UInt16 = 5500
    @Published var currentLabel: String = "UNLABELED"

    private var connection: NWConnection?

    override init() {
        super.init()
        if WCSession.isSupported() {
            WCSession.default.delegate = self
            WCSession.default.activate()
        }
    }

    func startRelay() {
        guard !macHost.isEmpty else {
            lastError = "Set Mac host IP first."
            return
        }
        let port = NWEndpoint.Port(rawValue: macPort) ?? 5500
        let conn = NWConnection(host: NWEndpoint.Host(macHost), port: port, using: .udp)
        conn.stateUpdateHandler = { [weak self] state in
            DispatchQueue.main.async {
                if case .failed(let err) = state { self?.lastError = err.localizedDescription }
            }
        }
        conn.start(queue: .global(qos: .userInitiated))
        connection = conn
        isRelaying = true
        lastError = nil
        sendLabelToWatch()
    }

    func stopRelay() {
        isRelaying = false
        connection?.cancel()
        connection = nil
    }

    func sendLabelToWatch() {
        guard WCSession.default.activationState == .activated else { return }
        guard WCSession.default.isReachable else {
            lastError = "Watch not reachable. Open the watch app to receive label/commands."
            return
        }

        WCSession.default.sendMessage(["label": currentLabel], replyHandler: nil) { [weak self] error in
            DispatchQueue.main.async { self?.lastError = error.localizedDescription }
        }
    }

    func sendCommandToWatch(_ cmd: String) {
        guard WCSession.default.activationState == .activated else { return }
        guard WCSession.default.isReachable else {
            lastError = "Watch not reachable. Open the watch app to receive label/commands."
            return
        }

        WCSession.default.sendMessage(["cmd": cmd], replyHandler: nil) { [weak self] error in
            DispatchQueue.main.async { self?.lastError = error.localizedDescription }
        }
    }
}

extension MacUdpRelay: WCSessionDelegate {
    func sessionDidBecomeInactive(_ session: WCSession) {}
    func sessionDidDeactivate(_ session: WCSession) { session.activate() }

    func session(_ session: WCSession, activationDidCompleteWith activationState: WCSessionActivationState, error: Error?) {
        if let error {
            DispatchQueue.main.async { self.lastError = error.localizedDescription }
        }
    }

    func session(_ session: WCSession, didReceiveMessageData messageData: Data) {
        guard isRelaying else { return }
        guard let conn = connection else { return }

        conn.send(content: messageData, completion: .contentProcessed { [weak self] error in
            DispatchQueue.main.async {
                if let error { self?.lastError = error.localizedDescription }
                self?.packetsSent += 1
            }
        })
    }
}
