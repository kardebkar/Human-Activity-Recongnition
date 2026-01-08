import SwiftUI

struct WatchContentView: View {
    @StateObject private var streamer = WatchImuStreamer()

    var body: some View {
        VStack(spacing: 8) {
            Text("IMU Stream")
                .font(.headline)

            Text(streamer.isStreaming ? "Streaming" : "Stopped")
                .foregroundColor(streamer.isStreaming ? .green : .secondary)

            Text("Sent: \(streamer.samplesSent)")
                .font(.footnote)

            Text("Label: \(streamer.currentLabel)")
                .font(.footnote)
                .lineLimit(1)

            if let err = streamer.lastError {
                Text(err)
                    .font(.footnote)
                    .foregroundColor(.red)
                    .lineLimit(2)
            }

            Button(streamer.isStreaming ? "Stop" : "Start") {
                streamer.isStreaming ? streamer.stop() : streamer.start()
            }
        }
        .padding()
    }
}

