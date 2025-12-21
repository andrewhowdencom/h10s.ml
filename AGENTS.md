# H10s Agent Configuration

This document serves as the primary context for agents working on this repository. It outlines the project's ultimate goals, hardware constraints, and necessary assumptions.

## Project Destination
The ultimate goal of this project is to deploy an arrhythmia classification model to mobile devices (**Android** first, potentially **iOS** later).

## Target Hardware: Polar H10
The model will process ECG data streamed from a **Polar H10** heart rate sensor. All development, training, and testing must account for the specific characteristics of this hardware.

### Hardware Assumptions & Constraints
*   **Sampling Rate**: **130 Hz**.
    *   *Implication*: All training data (e.g., PTB-XL, which is often 500Hz or 1000Hz) **must be downsampled to 130 Hz** to match the inference-time input.
*   **Lead Configuration**: **Single Lead (Chest Strap)**.
    *   *Implication*: The sensor sits on the chest. This is widely considered analogous to **Lead I** (Right Arm to Left Arm) in standard 12-lead ECGs, though exact positioning varies.
    *   *Action*: When selecting leads from datasets like PTB-XL, prioritize **Lead I**.
*   **Data Quality**: Ambulatory / Noisy.
    *   *Implication*: The user will be moving. Expect baseline wander, muscle noise (EMG), and motion artifacts. Preprocessing pipelines must be robust to this.
*   **Bandwidth/Resolution**: 
    *   The Polar H10 SDK typically provides raw ECG values in microvolts (µV).
    *   The resolution is generally sufficient for R-peak detection and basic morphology, but finer details may be lost compared to clinical 12-lead machines.

## Development Guidelines
1.  **Mobile-First Inference**: The final model must run on mobile (e.g., TensorFlow Lite). Avoid heavy architectures that cannot run in real-time on a phone.
2.  **Preprocessing Consistency**: The preprocessing pipeline used during training (in Python) must be exactly reproducible in the mobile app (Kotlin/Swift). Keep it simple (e.g., simple bandpass filters, standardization).
3.  **Data Mismatch**: We are training on clinical data (PTB-XL) but deploying on consumer hardware. We need to bridge this domain gap (e.g., by adding synthetic noise or using specific augmentation).
