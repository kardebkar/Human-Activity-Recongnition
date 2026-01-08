//
//  ImuTypes.swift
//  Wrist2AvatarStreamer
//
//  Created by Saswati_Kar on 12/23/25.
//

import Foundation
struct ImuSample: Codable {
      let t: Double
      let sr: Double
      let ax: Double
      let ay: Double
      let az: Double
      let gx: Double
      let gy: Double
      let gz: Double
      let label: String?
  }

  struct ImuBatch: Codable {
      let sr: Double
      let samples: [ImuSample]
  }
