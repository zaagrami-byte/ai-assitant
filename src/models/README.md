# `src/models/` — AI model files

This folder holds the **local AI models** used by the assistant. Large binary
model files are **never committed to Git** (see the root `.gitignore`) — only
this documentation and small config/metadata files live here.

Two different engines store their files here, in two different sub-locations:

```text
src/models/
├── README.md                        <- this file (tracked by Git)
├── fr_FR-siwis-medium.onnx.json     <- Piper voice config (tracked, small)
├── fr_FR-siwis-medium.onnx          <- Piper voice weights (NOT tracked — you download it)
└── whisper/                         <- faster-whisper model cache (NOT tracked — auto-downloaded)
```

## 9.1 Piper (TTS — text to speech)

Piper needs **two files per voice**: a `.onnx` weights file and a matching
`.onnx.json` config file. The `.onnx.json` for the default French voice
(`fr_FR-siwis-medium`) is already included in this repository — it's a small
text file, safe to version. The `.onnx` weights file (~60 MB) is **not**
included and must be downloaded once:

```bash
cd src/models
curl -L -o fr_FR-siwis-medium.onnx \
  "https://huggingface.co/rhasspy/piper-voices/resolve/main/fr/fr_FR/siwis/medium/fr_FR-siwis-medium.onnx"
```

The `.onnx.json` is already there, so you normally don't need to re-download
it. If you ever need it again (e.g. you picked a different voice), grab it
from the same folder on Hugging Face:
`https://huggingface.co/rhasspy/piper-voices/resolve/main/fr/fr_FR/siwis/medium/fr_FR-siwis-medium.onnx.json`

To use a different voice/language, browse the full catalog:
https://github.com/rhasspy/piper/blob/master/VOICES.md, download both files
for the voice you want into `src/models/`, and point `TTS_VOICE_NAME` (in
`config.py`, or via the `TTS_VOICE_NAME` environment variable) at the new
voice's file name (without the extension).

Controlled by these variables in `config.py`:

| Variable          | Default                     | Purpose                              |
|--------------------|-----------------------------|---------------------------------------|
| `TTS_MODELS_DIR`   | `src/models`                | Folder Piper looks in for voices      |
| `TTS_VOICE_NAME`   | `fr_FR-siwis-medium`        | Voice file name (no extension)        |

## 9.2 faster-whisper (STT — speech to text)

Unlike Piper, faster-whisper **downloads and caches its own model
automatically** the first time it runs — there is nothing to download by
hand. It saves the model files under `src/models/whisper/`, which is
git-ignored.

The model size is controlled by `STT_MODEL` in `config.py` (default:
`"small"`, a good CPU/Raspberry Pi trade-off between speed and accuracy).
Valid values include `tiny`, `base`, `small`, `medium`, `large-v3` — bigger
means better accuracy but slower and more RAM. See the faster-whisper
documentation for the full list.

Controlled by these variables in `config.py`:

| Variable        | Default   | Purpose                                  |
|-------------------|-----------|-------------------------------------------|
| `STT_MODEL`       | `small`   | Whisper model size to download/use         |
| `STT_MODEL_DIR`   | `src/models/whisper` | Cache folder for the downloaded model |
| `STT_DEVICE`      | `cpu`     | `cpu` (default) or `cuda` if you have a GPU |
| `STT_COMPUTE_TYPE`| `int8`    | Quantization — `int8` is fastest on CPU     |

## What if a model is missing?

- **Piper `.onnx` missing** → `tts.py` raises `TTSModelNotFoundError` with the
  exact path it looked for. Download it as shown above.
- **faster-whisper model missing** → it downloads automatically on first run
  (requires internet access once). If the machine has no internet access
  (e.g. an offline robot), download it once on a machine that does, then copy
  the resulting `src/models/whisper/` folder over.

## Why aren't these files in Git?

Model weights are large binary files that don't diff well and would bloat
the repository for every team and every clone. The `.gitignore` at the root
of the project blocks common model extensions (`*.onnx`, `*.bin`, `*.pt`,
`*.pth`, `*.safetensors`, `*.gguf`) as well as the whole `whisper/` cache
folder. Small, human-readable config files (like the Piper `.onnx.json`)
are fine to keep in Git since they're tiny and make onboarding easier.
