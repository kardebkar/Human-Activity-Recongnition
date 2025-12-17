# ICSR 2026 Paper Blueprint — Apple Watch → Unity Avatar Control (Hybrid Demo)

This document turns the “hybrid Apple Watch + Unity” concept into a ready-to-write paper outline, figure plan, and results table templates.

## 0) One‑Line Thesis

Use Apple Watch inertial sensing to infer user activity in real time and **drive an embodied avatar’s posture/motion** in Unity, with **uncertainty-aware stabilization** to make the interaction usable, predictable, and suitable for accessibility-oriented scenarios.

## 1) Paper Positioning (Fit to ICSR Topics)

- **Assistive Robotics / Welfare / Social Assistance**: low-effort, embodied control; activity-aware interaction policies.
- **Design, UX & Human Factors**: stability, workload reduction, perceived control.
- **Embodied Intelligence, Adaptation & Lifelong Autonomy**: streaming inference + state stabilization; optional personalization.
- **Trust, Explainability & Transparency**: “why” cues and confidence display; predictable transitions.
- **Gaming / Immersive Media**: Unity interactive demo; embodied agent/avatar.
- **Ethics / Privacy**: on-device inference; minimal retention; user control.

## 2) What You Will Claim (Contributions)

1. **System**: end-to-end architecture (watchOS → iOS inference → Unity) for real-time avatar control.
2. **Robust interaction layer**: uncertainty + hysteresis policy that reduces mode thrashing and improves perceived control.
3. **Evaluation**: user study comparing activity-driven control vs manual control (and optionally transparency ablation).
4. **Design guidance**: practical recommendations for wearable-driven embodied interaction (stability, transparency, privacy).

## 3) Ready-to-Write Paper Outline (Section by Section)

### Title
Pick one:
- “Wrist2Avatar: Real-Time Activity Recognition from Apple Watch for Embodied Avatar Control in Unity”
- “Wearable-Driven Embodied Interaction: Apple Watch Activity Inference for Accessible Avatar Control”
- “Stabilized Activity-Driven Avatar Control: A Hybrid Apple Watch + Unity Interactive Demo”

### Abstract (Checklist)
- Motivation: embodied avatar control is valuable but manual control is burdensome.
- Approach: Apple Watch sensing → inference → stable state → Unity avatar animation.
- Key techniques: uncertainty, hysteresis, lockout, optional transparency cues.
- Evaluation: objective performance + workload/usability/presence/trust.
- Results headline: “reduced workload”, “higher perceived control”, “lower switch rate” (fill with your numbers).
- Contributions summary.

### 1. Introduction
**Goal:** establish the HRI/assistive value; define what “control” means (avatar state/animation), and why stability matters.
- Problem: controlling embodied agents during movement; manual input is hard / distracting.
- Opportunity: ubiquitous wrist sensors provide real-time context and can drive embodied behavior.
- Challenges: domain variability, jitter, false triggers, user trust.
- Proposed solution: hybrid system + stabilization + transparency.
- Contributions (bullet list; mirror Section 2).

### 2. Related Work
Keep this targeted; connect to ICSR topics.
- Wrist-based activity recognition (watch/IMU literature).
- Embodied/Avatar control via wearables (HCI/HRI, VR locomotion proxies).
- Interruptibility-aware assistance + adaptive autonomy in social robotics.
- Explainable/transparent adaptive interfaces (confidence, “why” messaging).
- Accessibility and hands-busy interaction design.

### 3. System Overview
**Goal:** clear pipeline and design choices.
- Hardware: Apple Watch (CoreMotion), iPhone bridge, Unity runtime (desktop or iOS).
- Data flow: sensors → windows → inference → stabilization → Unity state machine.
- Communication: WatchConnectivity (watch→phone) and UDP/WebSocket (phone→Unity).
- Output: `(state, confidence, uncertainty, timestamp)` at ~2–4 Hz.
- Safety + privacy design: no raw data saved by default; optional consented logging.

### 4. Activity Recognition & State Estimation
**Goal:** defend that the classifier is credible for wrist data.
- State set: `STANDING`, `SITTING`, `WALKING`, `STAIRS_UP`, `STAIRS_DOWN`, `UNKNOWN`.
- Windowing: choose 1.0–2.56 s windows; overlap 50%.
- Model options (pick one and justify):
  - Tiny 1D CNN (fast, stable) OR
  - Tiny GRU (temporal dynamics) OR
  - Feature baseline (mean/std/energy) for a simpler first paper.
- Training protocol:
  - Data collection: scripted activities + timestamped labeling.
  - Splits: GroupKFold / leave-one-subject-out.
  - Metrics: macro-F1 + per-class recall (stairs/standing vs sitting).

### 5. Interaction Design: Avatar Control + Stabilization
**Goal:** show the HRI “secret sauce”.
- Mapping: `state` → Unity avatar animation (idle stand / sit / walk / stairs up/down).
- Stabilization policy:
  - Confidence threshold `τ`
  - “k consecutive windows” rule
  - lockout after switch (e.g., 1s)
  - `UNKNOWN` handling: keep current avatar state unless user overrides
- Manual override (optional but helpful): UI button to force a state temporarily.
- Transparency condition (optional):
  - Minimal overlay: state + confidence + short “why” string.
  - “Because you’re walking” reduces confusion and increases trust.

### 6. User Study
**Goal:** evaluate interaction outcomes, not just accuracy.
- Design: within-subject; counterbalanced (Latin square).
- Conditions:
  1) Manual control baseline (buttons/joystick/taps)
  2) Activity-driven control
  3) (optional) Activity-driven + transparency overlay
- Tasks (Unity):
  - Posture gates (stand/sit checkpoints)
  - Mobility sequence (walk → stairs up/down → stop)
  - Social interaction step (approach NPC/coach; state required to proceed)
- Measures:
  - Objective: task time, errors, corrections, mode-switch rate/min, dwell time, latency.
  - Subjective: NASA‑TLX, SUS, embodiment/presence, trust/predictability, perceived control.
- Procedure: training → calibration (optional) → tasks per condition → questionnaires → interview.
- Analysis: repeated-measures tests; effect sizes; qualitative themes for failure modes.

### 7. Results
Structure results so the reader can scan.
1) Model performance (offline): confusion matrix + macro-F1 (subject-independent).
2) System performance (online): end-to-end latency distribution; packet loss; stability metrics.
3) Study outcomes:
   - workload/usability improvements
   - reduced errors / faster completion
   - higher perceived control / presence
4) Ablations (if included): with vs without transparency; with vs without stabilization.

### 8. Discussion
Translate findings into design principles.
- When activity-driven control is beneficial (hands-busy, movement).
- When it fails (edge cases: carrying items, arm swing suppression).
- Why stabilization + `UNKNOWN` matters for trust.
- Implications for socially assistive systems and accessible embodied interaction.

### 9. Limitations & Future Work
- Domain shift: wrist vs other placements; variability across users and activities.
- Limited activity set; “stairs” is hard; need altimeter/pedometer fusion.
- Long-term use and personalization; longitudinal stability.
- Broader embodiments (physical robots, XR).

### 10. Ethics, Privacy, Safety
- On-device inference and minimal data retention.
- Consent for any logging; anonymization.
- Safety precautions for stairs/step-ups; alternatives (walking-in-place).

### 11. Conclusion
- Restate the main win: wearable-driven embodied control can be stable, usable, and measurable.
- Reiterate contributions and next step: robust deployments + broader studies.

## 4) Figure Plan (One-Page, “What to Draw / What to Screenshot”)

**Figure 1 — System Architecture (must-have)**
- Diagram: Apple Watch (IMU) → WatchConnectivity → iPhone (windowing + model + stabilization) → UDP/WebSocket → Unity (avatar state machine).
- Callouts: window size, update rate, stabilization rules, `UNKNOWN` behavior.

**Figure 2 — Unity Demo + Avatar States (must-have)**
- Grid of screenshots: avatar in `STANDING`, `SITTING`, `WALKING`, `STAIRS_UP/DOWN`.
- Show minimal overlay (state/confidence) if using transparency condition.

**Figure 3 — Study Design Timeline (must-have)**
- Within-subject flow: training → condition A tasks → survey → condition B → …
- Show counterbalancing and tasks sequence.

**Figure 4 — Model Confusion Matrix + Streaming Stability (strongly recommended)**
- Left: confusion matrix (subject-independent).
- Right: stability plot: mode-switch rate/min and/or example timeline of predicted state vs stabilized state.

Optional figures (only if space):
- Latency histogram (end-to-end).
- Qualitative themes: a small table with representative quotes.

## 5) Results Table Templates (Drop-in Markdown)

### Table 1 — Participants
| Metric | Value |
|---|---|
| N |  |
| Age (mean ± SD) |  |
| Gender |  |
| Dominant hand |  |
| Prior smartwatch use |  |
| Notes (accessibility-relevant, if applicable) |  |

### Table 2 — Offline Model Performance (Subject-Independent)
| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| STANDING |  |  |  |
| SITTING |  |  |  |
| WALKING |  |  |  |
| STAIRS_UP |  |  |  |
| STAIRS_DOWN |  |  |  |
| Macro Avg |  |  |  |

### Table 3 — Online System Metrics
| Metric | Mean | SD / IQR |
|---|---:|---:|
| End-to-end latency (ms) |  |  |
| Updates per second (Hz) |  |  |
| Packet loss (%) |  |  |
| Mode switches per minute (raw) |  |  |
| Mode switches per minute (stabilized) |  |  |
| Time in UNKNOWN (%) |  |  |

### Table 4 — Task Performance (Within-Subject)
| Outcome | Manual Control | Activity-Driven | (Optional) +Transparency | Test / Effect |
|---|---:|---:|---:|---|
| Completion time (s) |  |  |  |  |
| Errors (#) |  |  |  |  |
| Corrections (#) |  |  |  |  |
| Checkpoint success (%) |  |  |  |  |

### Table 5 — Questionnaires (Lower is better for TLX; higher is better for SUS, trust, presence)
| Scale | Manual | Activity-Driven | (Optional) +Transparency | Test / Effect |
|---|---:|---:|---:|---|
| NASA‑TLX |  |  |  |  |
| SUS |  |  |  |  |
| Perceived control |  |  |  |  |
| Trust/predictability |  |  |  |  |
| Presence/embodiment |  |  |  |  |

## 6) Appendix Checklist (Optional but Helpful)

- Full questionnaire items + Likert anchors.
- Stabilization parameters (`τ`, `k`, `t_hold`, lockout).
- Details of model architecture + training hyperparameters.
- Consent language for optional data logging.
- Demo operator script (for live conference demos).

## 7) “Minimum Publishable Demo” MVP

If you need a tight scope:
- States: `STANDING`, `WALKING`, `SITTING`, `UNKNOWN` (add stairs later).
- One Unity scene with 3 checkpoint types and an NPC interaction.
- Two conditions: manual vs activity-driven (add transparency as ablation if time).

## 8) Note on the Dataverse Apple Watch / Fitbit CSVs (Optional Starting Point)

If you are using engineered, tabular CSVs like `aw_fb_data.csv` / `data_for_weka_aw.csv` / `data_for_weka_fb.csv`:

- These files typically contain labels such as `Lying`, `Sitting`, `Self Pace walk`, and `Running 3/5/7 METs` (no explicit `Standing`).
- They are useful for a quick baseline and for mapping to avatar states (lie/sit/walk/run speeds), but they are **not raw IMU time-series**.
- For a “stand vs sit” interaction (and for robust wrist-based HRI claims), plan to collect a small wrist IMU dataset with an explicit `Standing` label and subject-independent evaluation.
