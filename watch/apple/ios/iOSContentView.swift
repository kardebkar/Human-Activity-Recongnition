import SwiftUI

struct iOSContentView: View {
    @StateObject private var relay = MacUdpRelay()

    private let labels = ["UNLABELED", "SITTING", "STANDING", "WALKING", "RUNNING", "LYING"]

    var body: some View {
        NavigationView {
            Form {
                Section(header: Text("Mac Target")) {
                    TextField("Mac IP (e.g., 192.168.1.25)", text: $relay.macHost)
                        .keyboardType(.numbersAndPunctuation)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()

                    Stepper(value: $relay.macPort, in: 1...65535) {
                        Text("Port: \(relay.macPort)")
                    }

                    Button(relay.isRelaying ? "Stop Relay" : "Start Relay") {
                        relay.isRelaying ? relay.stopRelay() : relay.startRelay()
                    }
                }

                Section(header: Text("Label (optional)")) {
                    if #available(iOS 17.0, *) {
                        Picker("Current label", selection: $relay.currentLabel) {
                            ForEach(labels, id: \.self) { Text($0) }
                        }
                        .onChange(of: relay.currentLabel) { _, _ in
                            relay.sendLabelToWatch()
                        }
                    } else {
                        Picker("Current label", selection: $relay.currentLabel) {
                            ForEach(labels, id: \.self) { Text($0) }
                        }
                        .onChange(of: relay.currentLabel) { _ in
                            relay.sendLabelToWatch()
                        }
                    }
                }

                Section(header: Text("Watch Controls (optional)")) {
                    HStack {
                        Button("Start") { relay.sendCommandToWatch("start") }
                        Button("Stop") { relay.sendCommandToWatch("stop") }
                    }
                }

                Section(header: Text("Status")) {
                    Text(relay.isRelaying ? "Relaying" : "Stopped")
                    Text("Packets sent: \(relay.packetsSent)")
                    if let err = relay.lastError {
                        Text(err).foregroundColor(.red)
                    }
                }
            }
            .navigationTitle("Watch → Mac Relay")
        }
    }
}
