"""
tests/test_stt.py
------------------
Tests du module stt.py.
"""

import os
import sys
import time
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from stt import (  # noqa: E402
    STTEngine,
    STTEmptyAudioError,
    STTMicrophoneError,
    STTModelError,
    STTTranscriptionError,
    STTTimeoutError,
)


def make_engine() -> STTEngine:
    engine = STTEngine()
    engine._model = MagicMock()
    return engine


def fake_segments(text: str):
    return [SimpleNamespace(text=text)]


class FakeStream:
    """Simule sd.InputStream : renvoie une sequence de blocs predefinis."""

    def __init__(self, blocks):
        self._blocks = list(blocks)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, frame_size):
        if not self._blocks:
            # plus rien a lire : on renvoie du silence pour ne pas planter le test
            return np.zeros((frame_size, 1), dtype=np.int16), False
        return self._blocks.pop(0), False


def make_vad_sequence(n_speech: int, n_silence: int, frame_size: int):
    """n_speech blocs 'voix' suivis de n_silence blocs 'silence'."""
    speech_block = np.full((frame_size, 1), 3000, dtype=np.int16)
    silence_block = np.zeros((frame_size, 1), dtype=np.int16)
    return [speech_block] * n_speech + [silence_block] * n_silence


class TestSTTEngineMocked(unittest.TestCase):

    def test_transcribe_none_audio_raises(self):
        engine = make_engine()
        with self.assertRaises(STTEmptyAudioError):
            engine.transcribe_audio(None)

    def test_transcribe_empty_array_raises(self):
        engine = make_engine()
        with self.assertRaises(STTEmptyAudioError):
            engine.transcribe_audio(np.array([]))

    def test_microphone_unavailable_raises(self):
        engine = make_engine()
        with patch("stt.sd", None):
            with self.assertRaises(STTMicrophoneError):
                engine._record_audio()

    def test_webrtcvad_unavailable_raises(self):
        engine = make_engine()
        with patch("stt.webrtcvad", None):
            with self.assertRaises(STTMicrophoneError):
                engine._record_audio()

    def test_model_load_failure_raises(self):
        engine = STTEngine()
        with patch("faster_whisper.WhisperModel", side_effect=OSError("download failed")):
            with self.assertRaises(STTModelError):
                engine._load_model()

    def test_transcribe_audio_returns_text(self):
        engine = make_engine()
        engine._model.transcribe.return_value = (fake_segments("Bonjour Rami"), None)
        texte = engine.transcribe_audio("dummy_path.wav")
        self.assertEqual(texte, "Bonjour Rami")

    def test_transcribe_passes_anti_hallucination_params(self):
        engine = make_engine()
        engine._model.transcribe.return_value = (fake_segments("ok"), None)
        engine.transcribe_audio("dummy_path.wav")

        kwargs = engine._model.transcribe.call_args.kwargs
        self.assertEqual(kwargs["vad_filter"], engine.vad_filter)
        self.assertEqual(kwargs["no_speech_threshold"], engine.no_speech_threshold)
        self.assertEqual(kwargs["log_prob_threshold"], engine.log_prob_threshold)
        self.assertEqual(kwargs["compression_ratio_threshold"], engine.compression_ratio_threshold)
        self.assertEqual(kwargs["condition_on_previous_text"], engine.condition_on_previous_text)

    def test_transcription_error_raises(self):
        engine = make_engine()
        engine._model.transcribe.side_effect = RuntimeError("modele en erreur")
        with self.assertRaises(STTTranscriptionError):
            engine.transcribe_audio("dummy_path.wav")

    def test_empty_transcription_result_raises(self):
        engine = make_engine()
        engine._model.transcribe.return_value = (fake_segments(""), None)
        with self.assertRaises(STTTranscriptionError):
            engine.transcribe_audio("dummy_path.wav")

    def test_transcription_timeout_raises(self):
        engine = make_engine()
        engine.timeout = 0.05

        def slow_transcribe(*args, **kwargs):
            time.sleep(0.3)
            return (fake_segments("trop tard"), None)

        engine._model.transcribe.side_effect = slow_transcribe
        with self.assertRaises(STTTimeoutError):
            engine.transcribe_audio("dummy_path.wav")

    def test_temp_wav_is_cleaned_up(self):
        engine = make_engine()
        engine._model.transcribe.return_value = (fake_segments("test nettoyage"), None)

        captured_path = {}
        original_save = engine._save_temp_wav

        def spy_save(audio):
            path = original_save(audio)
            captured_path["path"] = path
            return path

        engine._save_temp_wav = spy_save

        audio = np.zeros(1600, dtype=np.int16)
        engine.transcribe_audio(audio)

        self.assertIn("path", captured_path)
        self.assertFalse(os.path.exists(captured_path["path"]))

    def test_configuration_values_are_loaded(self):
        engine = STTEngine()
        self.assertEqual(engine.model_name, config.STT_MODEL)
        self.assertEqual(engine.device, config.STT_DEVICE)
        self.assertEqual(engine.compute_type, config.STT_COMPUTE_TYPE)
        self.assertEqual(engine.language, config.STT_LANGUAGE)

    # ---------------- VAD (webrtcvad) ----------------

    def test_vad_rejects_invalid_frame_ms(self):
        with self.assertRaises(ValueError):
            STTEngine(vad_frame_ms=25)

    def test_no_speech_at_all_raises_empty_audio(self):
        engine = STTEngine()
        frame_size = int(engine.sample_rate * engine.vad_frame_ms / 1000)
        n_frames = int(engine.max_record_seconds * 1000 / engine.vad_frame_ms)
        blocks = make_vad_sequence(n_speech=0, n_silence=n_frames, frame_size=frame_size)

        fake_vad = MagicMock()
        fake_vad.is_speech.return_value = False

        with patch("stt.sd") as fake_sd, patch("stt.webrtcvad") as fake_webrtcvad:
            fake_sd.InputStream.return_value = FakeStream(blocks)
            fake_webrtcvad.Vad.return_value = fake_vad
            with self.assertRaises(STTEmptyAudioError):
                engine._record_audio()

    def test_speech_then_silence_stops_before_max_duration(self):
        engine = STTEngine()
        frame_size = int(engine.sample_rate * engine.vad_frame_ms / 1000)

        # assez de frames "voix" pour declencher le debut, puis assez de
        # frames "silence" pour declencher la fin — bien avant max_record_seconds.
        n_start = max(1, round(engine.vad_start_window_ms / engine.vad_frame_ms)) + 5
        n_end = max(1, round((engine.silence_duration * 1000) / engine.vad_frame_ms)) + 5
        blocks = make_vad_sequence(n_speech=n_start, n_silence=n_end, frame_size=frame_size)

        speech_block = blocks[0]

        def fake_is_speech(frame_bytes, sample_rate):
            block = np.frombuffer(frame_bytes, dtype=np.int16)
            return bool(np.any(block == speech_block[0, 0]))

        fake_vad = MagicMock()
        fake_vad.is_speech.side_effect = fake_is_speech

        with patch("stt.sd") as fake_sd, patch("stt.webrtcvad") as fake_webrtcvad:
            fake_sd.InputStream.return_value = FakeStream(list(blocks))
            fake_webrtcvad.Vad.return_value = fake_vad

            audio = engine._record_audio()

        expected_max_frames = int(engine.max_record_seconds * 1000 / engine.vad_frame_ms)
        actual_frames = audio.shape[0] // frame_size
        self.assertLess(actual_frames, expected_max_frames)
        self.assertGreater(audio.shape[0], 0)


@unittest.skipUnless(
    os.environ.get("RUN_INTEGRATION") == "1",
    "Test d'integration desactive par defaut (necessite microphone + modele "
    "faster-whisper telecharge). Lancer avec RUN_INTEGRATION=1 pour l'activer.",
)
class TestSTTEngineIntegration(unittest.TestCase):

    def test_real_listen_returns_nonempty_text(self):
        engine = STTEngine()
        print("\nParlez une phrase courte en francais, puis marquez une pause naturelle...")
        texte = engine.listen()
        self.assertIsInstance(texte, str)
        self.assertTrue(len(texte) > 0)


if __name__ == "__main__":
    unittest.main()