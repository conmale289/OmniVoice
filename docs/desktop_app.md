# Desktop App

`omnivoice-desktop` provides a native desktop workflow for daily TTS usage.

## Install

```bash
pip install "omnivoice[desktop]"
# or
uv sync --extra desktop
```

## Launch

```bash
omnivoice-desktop
```

## Features

- **Studio tab**
  - Load model checkpoint (`k2-fsa/OmniVoice` or local path)
  - Pick device (`auto`, `cuda`, `mps`, `cpu`) and dtype
  - Voice clone / voice design / auto voice in one screen
  - Generation controls: `num_step`, `guidance_scale`, `speed`, `duration`, `t_shift`
  - Runtime controls: `denoise`, `preprocess_prompt`, `postprocess_output`
  - Built-in playback (`Play`, `Stop`) and `Save As...`
  - Audio history list (double-click to replay)

- **Batch tab**
  - Run batch inference from JSONL via `omnivoice.cli.infer_batch`
  - Configure workers/GPU, batch duration, batch size
  - View full command output logs inside app

- **Guide tab**
  - Multi-level guide for different audiences:
    - Beginner quick start
    - Role-based playbooks (education/content/business/technical)
    - Detailed batch JSONL guidance
    - Troubleshooting and accessibility notes

## Input Notes

- **Voice clone**: provide `ref_audio` (3-10 seconds recommended)
- **Voice design**: provide `instruct`, no reference audio required
- **Auto voice**: provide text only
- If both `duration` and `speed` are set, `duration` takes priority

## Accessibility & Readability

- High-contrast light theme optimized for long reading sessions
- Larger default typography and bigger interactive controls
- Improved checkbox visibility and log readability
- Structured hints near critical fields (mode detection, reference audio quality)
