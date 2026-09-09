"""
tests/test_assistant.py
------------------------
Tests du module assistant.py.

Uniquement des tests mockes : on injecte des faux moteurs STT/LLM/TTS pour
verifier la logique d'orchestration (enchainement, gestion d'erreurs,
detection de la commande d'arret) sans dependre du materiel reel.

Lancer :
    python -m unittest tests.test_assistant -v
"""

import os
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from assistant import Assistant, is_stop_command  # noqa: E402
from llm import LLMError  # noqa: E402
from stt import STTError, STTEmptyAudioError  # noqa: E402
from tts import TTSError  # noqa: E402


def make_assistant(stt_text="Bonjour", llm_response="Bonjour !"):
    stt = MagicMock()
    stt.listen.return_value = stt_text
    llm = MagicMock()
    llm.generate_response.return_value = llm_response
    speak_fn = MagicMock()
    return Assistant(stt_engine=stt, llm_engine=llm, speak_fn=speak_fn), stt, llm, speak_fn


class TestStopCommandDetection(unittest.TestCase):

    def test_detects_stop(self):
        self.assertTrue(is_stop_command("stop"))
        self.assertTrue(is_stop_command("Stop."))

    def test_detects_arret_with_accent(self):
        self.assertTrue(is_stop_command("arrêt"))
        self.assertTrue(is_stop_command("Arrête !"))

    def test_regular_sentence_is_not_stop(self):
        self.assertFalse(is_stop_command("Bonjour, comment allez-vous ?"))
        self.assertFalse(is_stop_command("stopper la voiture"))


class TestAssistantTurn(unittest.TestCase):

    def test_normal_turn_calls_all_three_engines(self):
        assistant, stt, llm, speak_fn = make_assistant(
            stt_text="Bonjour", llm_response="Bonjour, comment puis-je vous aider ?"
        )
        should_continue = assistant.run_one_turn()

        self.assertTrue(should_continue)
        stt.listen.assert_called_once()
        llm.generate_response.assert_called_once_with("Bonjour")
        speak_fn.assert_called_once_with("Bonjour, comment puis-je vous aider ?")

    def test_stop_command_ends_conversation_without_calling_llm(self):
        assistant, stt, llm, speak_fn = make_assistant(stt_text="stop")
        should_continue = assistant.run_one_turn()

        self.assertFalse(should_continue)
        llm.generate_response.assert_not_called()
        speak_fn.assert_called_once()  # message d'au revoir

    def test_empty_audio_just_reloops_silently(self):
        assistant, stt, llm, speak_fn = make_assistant()
        stt.listen.side_effect = STTEmptyAudioError("silence")

        should_continue = assistant.run_one_turn()

        self.assertTrue(should_continue)
        llm.generate_response.assert_not_called()
        speak_fn.assert_not_called()

    def test_stt_error_speaks_error_message_and_continues(self):
        assistant, stt, llm, speak_fn = make_assistant()
        stt.listen.side_effect = STTError("micro en panne")

        should_continue = assistant.run_one_turn()

        self.assertTrue(should_continue)
        llm.generate_response.assert_not_called()
        speak_fn.assert_called_once()

    def test_llm_error_speaks_error_message_and_continues(self):
        assistant, stt, llm, speak_fn = make_assistant(stt_text="Bonjour")
        llm.generate_response.side_effect = LLMError("ollama indisponible")

        should_continue = assistant.run_one_turn()

        self.assertTrue(should_continue)
        speak_fn.assert_called_once()

    def test_tts_error_does_not_crash_turn(self):
        assistant, stt, llm, speak_fn = make_assistant(
            stt_text="Bonjour", llm_response="Reponse"
        )
        speak_fn.side_effect = TTSError("pas de haut-parleur")

        # Ne doit lever aucune exception
        should_continue = assistant.run_one_turn()
        self.assertTrue(should_continue)


if __name__ == "__main__":
    unittest.main()
