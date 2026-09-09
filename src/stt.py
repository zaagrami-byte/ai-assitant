"""
stt.py
------
Module de reconnaissance vocale (Speech-to-Text) local, base sur faster-whisper.

VAD : webrtcvad (classification parole/non-parole par frame), remplace
l'ancien seuil d'energie RMS — plus robuste au bruit ambiant, plus rapide
a couper la fin de la phrase, moins d'hallucinations Whisper sur du bruit.
"""

from __future__ import annotations

import collections
import logging
import os
import tempfile
import time
import wave
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field

import numpy as np

import config

try:
    import sounddevice as sd
except Exception:
    sd = None

try:
    import webrtcvad
except Exception:
    webrtcvad = None


os.makedirs(config.LOG_DIR, exist_ok=True)

logger = logging.getLogger("robot_assistant.stt")
logger.setLevel(getattr(logging, config.LOG_LEVEL, logging.INFO))

if not logger.handlers:
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    file_handler = logging.FileHandler(config.LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


class STTError(Exception):
    """Erreur generique du module STT."""


class STTMicrophoneError(STTError):
    pass


class STTModelError(STTError):
    pass


class STTTranscriptionError(STTError):
    pass


class STTTimeoutError(STTError):
    pass


class STTEmptyAudioError(STTError):
    pass


@dataclass
class STTEngine:
    model_name: str = config.STT_MODEL
    device: str = config.STT_DEVICE
    compute_type: str = config.STT_COMPUTE_TYPE
    language: str = config.STT_LANGUAGE
    sample_rate: int = config.STT_SAMPLE_RATE
    channels: int = config.STT_CHANNELS
    max_record_seconds: int = config.STT_MAX_RECORD_SECONDS
    silence_duration: float = config.STT_SILENCE_DURATION
    model_dir: str = config.STT_MODEL_DIR
    tmp_dir: str = config.STT_TMP_DIR
    timeout: int = config.STT_TIMEOUT

    # VAD (webrtcvad)
    vad_mode: int = config.STT_VAD_MODE
    vad_frame_ms: int = config.STT_VAD_FRAME_MS
    vad_start_window_ms: int = config.STT_VAD_START_WINDOW_MS
    vad_start_ratio: float = config.STT_VAD_START_RATIO
    vad_end_ratio: float = config.STT_VAD_END_RATIO

    # Anti-hallucination faster-whisper
    vad_filter: bool = config.STT_VAD_FILTER
    no_speech_threshold: float = config.STT_NO_SPEECH_THRESHOLD
    log_prob_threshold: float = config.STT_LOG_PROB_THRESHOLD
    compression_ratio_threshold: float = config.STT_COMPRESSION_RATIO_THRESHOLD
    condition_on_previous_text: bool = config.STT_CONDITION_ON_PREVIOUS_TEXT
    beam_size: int = config.STT_BEAM_SIZE

    _model: object = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        if self.vad_frame_ms not in (10, 20, 30):
            raise ValueError("vad_frame_ms doit valoir 10, 20 ou 30 (contrainte webrtcvad).")
        logger.info(
            "STTEngine initialise (modele=%s, device=%s, compute_type=%s, langue=%s, vad_mode=%s)",
            self.model_name, self.device, self.compute_type, self.language, self.vad_mode,
        )

    # ------------------------------------------------------------------
    # API publique
    # ------------------------------------------------------------------

    def listen(self) -> str:
        model_load_time = 0.0
        if self._model is None:
            model_load_time = self._load_model_timed()

        logger.info("Enregistrement demarre.")
        record_start = time.perf_counter()
        audio = self._record_audio()
        recording_time = time.perf_counter() - record_start
        logger.info("Enregistrement termine (%.2fs).", recording_time)

        logger.info("Transcription demarree.")
        transcription_start = time.perf_counter()
        text = self._transcribe_with_timeout(audio)
        transcription_time = time.perf_counter() - transcription_start

        total_time = model_load_time + recording_time + transcription_time
        logger.info(
            "STT latency: model=%.2fs record=%.2fs transcribe=%.2fs total=%.2fs | texte='%s'",
            model_load_time, recording_time, transcription_time, total_time, self._truncate(text),
        )
        return text

    def transcribe_audio(self, audio) -> str:
        if audio is None or (isinstance(audio, np.ndarray) and audio.size == 0):
            logger.warning("transcribe_audio appele avec un audio vide, ignore.")
            raise STTEmptyAudioError("L'audio fourni est vide.")

        if self._model is None:
            self._load_model_timed()

        return self._transcribe_with_timeout(audio)

    def reset_model(self) -> None:
        self._model = None
        logger.info("Modele STT decharge.")

    # ------------------------------------------------------------------
    # Internes — chargement du modele
    # ------------------------------------------------------------------

    def _load_model_timed(self) -> float:
        start = time.perf_counter()
        self._model = self._load_model()
        load_time = time.perf_counter() - start
        logger.info("Modele Whisper charge en %.2fs.", load_time)
        return load_time

    def _load_model(self):
        os.makedirs(self.model_dir, exist_ok=True)
        try:
            from faster_whisper import WhisperModel
            return WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type,
                download_root=self.model_dir,
            )
        except Exception as exc:
            logger.exception("Echec du chargement du modele faster-whisper.")
            raise STTModelError(
                f"Impossible de charger le modele '{self.model_name}' : {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Internes — microphone + VAD (webrtcvad)
    # ------------------------------------------------------------------

    def _record_audio(self) -> np.ndarray:
        if sd is None:
            raise STTMicrophoneError(
                "sounddevice/PortAudio indisponible. Installe la lib systeme "
                "('sudo apt install libportaudio2') puis reinstalle sounddevice."
            )
        if webrtcvad is None:
            raise STTMicrophoneError(
                "webrtcvad n'est pas installe. Lance : pip install webrtcvad"
            )

        vad = webrtcvad.Vad(self.vad_mode)
        frame_size = int(self.sample_rate * self.vad_frame_ms / 1000)

        num_start_frames = max(1, round(self.vad_start_window_ms / self.vad_frame_ms))
        num_end_frames = max(1, round((self.silence_duration * 1000) / self.vad_frame_ms))

        start_window: collections.deque = collections.deque(maxlen=num_start_frames)
        end_window: collections.deque = collections.deque(maxlen=num_end_frames)

        voiced_frames: list[np.ndarray] = []
        triggered = False
        elapsed_ms = 0
        max_ms = self.max_record_seconds * 1000

        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="int16",
            ) as stream:
                while elapsed_ms < max_ms:
                    block, _overflowed = stream.read(frame_size)
                    elapsed_ms += self.vad_frame_ms

                    is_speech = vad.is_speech(block.tobytes(), self.sample_rate)

                    if not triggered:
                        start_window.append((block, is_speech))
                        voiced_count = sum(1 for _, s in start_window if s)
                        if voiced_count >= self.vad_start_ratio * start_window.maxlen:
                            triggered = True
                            # on garde le contexte pre-declenchement (debut de mot non coupe)
                            voiced_frames.extend(b for b, _ in start_window)
                            start_window.clear()
                    else:
                        voiced_frames.append(block)
                        end_window.append(is_speech)
                        if len(end_window) == end_window.maxlen:
                            unvoiced_count = sum(1 for s in end_window if not s)
                            if unvoiced_count >= self.vad_end_ratio * end_window.maxlen:
                                break
        except STTMicrophoneError:
            raise
        except Exception as exc:
            logger.exception("Erreur microphone pendant l'enregistrement.")
            raise STTMicrophoneError(f"Erreur d'enregistrement audio : {exc}") from exc

        if not voiced_frames:
            raise STTEmptyAudioError("Aucune parole detectee (silence uniquement).")

        return np.concatenate(voiced_frames, axis=0)

    # ------------------------------------------------------------------
    # Internes — transcription
    # ------------------------------------------------------------------

    def _transcribe_with_timeout(self, audio) -> str:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._transcribe, audio)
            try:
                return future.result(timeout=self.timeout)
            except FutureTimeoutError as exc:
                logger.error("Timeout (>%ss) pendant la transcription.", self.timeout)
                raise STTTimeoutError(
                    f"La transcription n'a pas termine en moins de {self.timeout}s."
                ) from exc

    def _transcribe(self, audio) -> str:
        wav_path = None
        try:
            if isinstance(audio, np.ndarray):
                wav_path = self._save_temp_wav(audio)
                source = wav_path
            else:
                source = audio

            segments, _info = self._model.transcribe(
                source,
                language=self.language,
                beam_size=self.beam_size,
                vad_filter=self.vad_filter,
                vad_parameters=dict(min_silence_duration_ms=500),
                no_speech_threshold=self.no_speech_threshold,
                log_prob_threshold=self.log_prob_threshold,
                compression_ratio_threshold=self.compression_ratio_threshold,
                condition_on_previous_text=self.condition_on_previous_text,
            )
            text = " ".join(segment.text.strip() for segment in segments).strip()
        except STTError:
            raise
        except Exception as exc:
            logger.exception("Echec de la transcription.")
            raise STTTranscriptionError(f"Echec de la transcription : {exc}") from exc
        finally:
            if wav_path and os.path.exists(wav_path):
                try:
                    os.remove(wav_path)
                except OSError:
                    pass

        if not text:
            raise STTTranscriptionError("Le modele n'a reconnu aucun texte.")

        logger.info("Texte reconnu : %s", self._truncate(text))
        return text

    def _save_temp_wav(self, audio: np.ndarray) -> str:
        os.makedirs(self.tmp_dir, exist_ok=True)
        fd, wav_path = tempfile.mkstemp(suffix=".wav", prefix="stt_", dir=self.tmp_dir)
        os.close(fd)
        with wave.open(wav_path, "wb") as wav_file:
            wav_file.setnchannels(self.channels)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(audio.tobytes())
        return wav_path

    @staticmethod
    def _truncate(text: str, length: int = 60) -> str:
        return text if len(text) <= length else text[: length - 1] + "…"


if __name__ == "__main__":
    print("=== Test manuel STT ===")
    print(f"Modele : {config.STT_MODEL} | Device : {config.STT_DEVICE}\n")

    engine = STTEngine()

    print("Chargement du modele...")
    try:
        engine._load_model_timed()
        print("Modele charge.\n")
    except STTError as e:
        print(f"[ERREUR] {e}")
        raise SystemExit(1)

    try:
        while True:
            input("Appuyez sur Entree puis parlez (Ctrl+C pour quitter)...")
            print("Parlez maintenant...")
            try:
                start = time.perf_counter()
                texte = engine.listen()
                elapsed = time.perf_counter() - start
                print(f'Texte reconnu : "{texte}"')
                print(f"(latence totale : {elapsed:.2f}s)\n")
            except STTError as e:
                print(f"[ERREUR] {e}\n")
    except KeyboardInterrupt:
        print("\n=== Fin du test ===")