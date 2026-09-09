"""
tests/test_tts.py
------------------
Tests du module tts.py.

Comme pour le LLM, on distingue :
  1. Tests mockés (aucun besoin de haut-parleur ni de modèle Piper réel) :
     vérifient la gestion d'erreurs (texte vide, modèle absent, pas de
     lecteur audio disponible).
  2. Un test d'intégration réel (nécessite le modèle Piper téléchargé ET
     un haut-parleur fonctionnel), désactivé par défaut.

Lancer les tests mockés :
    python -m unittest tests.test_tts -v

Lancer aussi le test réel (avec son) :
    RUN_INTEGRATION=1 python -m unittest tests.test_tts -v
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tts  # noqa: E402
from tts import (  # noqa: E402
    TTSEmptyTextError,
    TTSModelNotFoundError,
    TTSPlaybackError,
)


class TestTTSMocked(unittest.TestCase):

    def setUp(self):
        # Repartir d'une voix "non chargée" à chaque test pour ne pas
        # dépendre de l'ordre d'exécution.
        tts._voice = None

    def test_empty_text_raises(self):
        with self.assertRaises(TTSEmptyTextError):
            tts.speak("   ")

    def test_missing_model_files_raise_clear_error(self):
        with patch("config.TTS_MODEL_PATH", "/chemin/inexistant.onnx"), \
             patch("config.TTS_CONFIG_PATH", "/chemin/inexistant.onnx.json"):
            with self.assertRaises(TTSModelNotFoundError):
                tts.speak("Bonjour")

    def test_no_audio_player_raises_playback_error(self):
        with patch("shutil.which", return_value=None), \
             patch("config.TTS_PLAYER_CMD", None):
            with self.assertRaises(TTSPlaybackError):
                tts._detect_player()


    def test_explicit_player_cmd_is_respected(self):
        with patch("config.TTS_PLAYER_CMD", "mon_lecteur_custom"):
            self.assertEqual(tts._detect_player(), "mon_lecteur_custom")


@unittest.skipUnless(
    os.environ.get("RUN_INTEGRATION") == "1",
    "Test d'intégration désactivé par défaut (nécessite le modèle Piper "
    "téléchargé + un haut-parleur fonctionnel). Lancer avec RUN_INTEGRATION=1.",
)
class TestTTSIntegration(unittest.TestCase):
    """Test réel : synthèse ET lecture audio sur le haut-parleur de la machine."""

    def test_real_speak_short_sentence(self):
        metrics = tts.speak("Ceci est un test.")
        self.assertIn("synthesis_time", metrics)
        self.assertGreater(metrics["synthesis_time"], 0)

    def test_real_synthesis_without_playback(self):
        # Utile pour mesurer la latence de synthèse seule, sans dépendre
        # du haut-parleur / lecteur audio.
        metrics = tts.speak("Test sans lecture audio.", play_audio=False)
        self.assertGreater(metrics["synthesis_time"], 0)
        self.assertEqual(metrics["playback_time"], 0.0)


if __name__ == "__main__":
    unittest.main()
