# Desktop Feature Research (OmniVoice)

Updated: 2026-05-12

## Goal

Build a desktop-first experience for frequent TTS workflows with low friction:
- Fast setup for non-technical users
- Full control for power users
- Stable long-running tasks (model loading, generation, batch jobs)

## User Workflow Analysis

Primary workflows observed from repo capabilities:

1. **Single synthesis**  
   Input text -> pick mode (clone/design/auto) -> generate -> listen -> export.

2. **Iterative tuning**  
   Try multiple parameter combinations quickly (`num_step`, guidance, speed/duration).

3. **Batch production**  
   Run JSONL jobs, monitor logs, collect many wavs.

4. **Occasional troubleshooting**  
   Device mismatch, model load errors, missing reference files.

## Feature Prioritization

## Must-have (implemented)

- Model load panel (checkpoint, device, dtype, optional ASR)
- One-screen synthesis for clone/design/auto voice
- Generation controls for quality/performance tradeoff
- Non-blocking background tasks (UI remains responsive)
- In-app audio playback + save exported wav
- Synthesis history and quick replay
- Batch JSONL runner with logs
- Clear guide tab for quick onboarding

## Should-have (implemented)

- Quality profiles (Fast/Balanced/High Quality)
- Auto mode inference from provided fields (`ref_audio`, `instruct`)
- Unified status/progress signal

## Nice-to-have (backlog)

- Waveform/spectrogram visualization
- Preset management and reusable project sessions
- Drag-and-drop input + batch queue manager
- System notifications on long batch completion
- One-click model download/cache diagnostics

## UX Decisions

- **Two-level structure**:
  - `Studio`: interactive creative flow
  - `Batch`: production flow
- **Low cognitive load**:
  - sensible defaults (`Balanced`, `auto` device)
  - optional advanced controls without hiding critical options
- **Fast feedback**:
  - progress bar + runtime log + immediate playback
- **Error transparency**:
  - show traceback tail in dialog + full trace in log

## Delivered Scope

Implemented desktop app entrypoint:

- `omnivoice-desktop` -> `omnivoice.desktop.app:main`

Core files:

- `omnivoice/desktop/app.py`
- `docs/desktop_app.md`

