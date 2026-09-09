"""
tests/test_llm.py
------------------
Tests du module llm.py.

Deux catégories :
  1. Tests unitaires "mockés" (ne nécessitent PAS Ollama démarré) :
     vérifient la gestion d'erreurs, l'historique, le texte vide, etc.
  2. Un test d'intégration réel (nécessite Ollama + qwen2.5:3b actifs),
     désactivé par défaut pour ne pas faire échouer la CI si Ollama
     n'est pas disponible sur la machine qui lance les tests.

Lancer uniquement les tests mockés (rapide, toujours possible) :
    python -m unittest tests.test_llm -v

Lancer aussi le test d'intégration réel :
    RUN_INTEGRATION=1 python -m unittest tests.test_llm -v
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Permet de lancer ce fichier directement depuis tests/ ou depuis la racine
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ollama  # noqa: E402

from llm import (  # noqa: E402
    LLMEngine,
    LLMEmptyResponseError,
    LLMModelNotFoundError,
    LLMUnavailableError,
)


def make_engine() -> LLMEngine:
    """Crée un LLMEngine avec un client Ollama mocké (aucun appel réseau réel)."""
    engine = LLMEngine()
    engine._client = MagicMock()
    return engine


class TestLLMEngineMocked(unittest.TestCase):
    """Tests qui ne touchent jamais le vrai service Ollama."""

    def test_empty_user_text_raises(self):
        engine = make_engine()
        with self.assertRaises(LLMEmptyResponseError):
            engine.generate_response("   ")

    def test_generate_response_returns_content(self):
        engine = make_engine()
        engine._client.chat.return_value = {
            "message": {"role": "assistant", "content": "Bonjour Rami !"}
        }
        reponse = engine.generate_response("Bonjour")
        self.assertEqual(reponse, "Bonjour Rami !")

    def test_empty_model_response_raises(self):
        engine = make_engine()
        engine._client.chat.return_value = {"message": {"role": "assistant", "content": ""}}
        with self.assertRaises(LLMEmptyResponseError):
            engine.generate_response("Bonjour")

    def test_model_not_found_maps_to_specific_error(self):
        engine = make_engine()
        engine._client.chat.side_effect = ollama.ResponseError(
            error="model not found", status_code=404
        )
        with self.assertRaises(LLMModelNotFoundError):
            engine.generate_response("Bonjour")

    def test_connection_error_maps_to_unavailable(self):
        engine = make_engine()
        engine._client.chat.side_effect = ConnectionError("refused")
        with self.assertRaises(LLMUnavailableError):
            engine.generate_response("Bonjour")

    def test_history_accumulates_and_is_sent_to_model(self):
        engine = make_engine()
        engine._client.chat.return_value = {
            "message": {"role": "assistant", "content": "Enchanté Rami."}
        }
        engine.generate_response("Je m'appelle Rami.")

        history = engine.get_history()
        self.assertEqual(len(history), 2)  # 1 message user + 1 message assistant
        self.assertEqual(history[0]["content"], "Je m'appelle Rami.")

        # Le prochain appel doit inclure l'historique dans les messages envoyés
        engine._client.chat.return_value = {
            "message": {"role": "assistant", "content": "Rami."}
        }
        engine.generate_response("Quel est mon prénom ?")

        sent_messages = engine._client.chat.call_args.kwargs["messages"]
        contents = [m["content"] for m in sent_messages]
        self.assertIn("Je m'appelle Rami.", contents)

    def test_history_is_truncated_to_max_history(self):
        engine = make_engine()
        engine.max_history = 2  # garde seulement 2 échanges = 4 messages
        engine._client.chat.return_value = {
            "message": {"role": "assistant", "content": "ok"}
        }
        for i in range(5):
            engine.generate_response(f"message {i}")

        self.assertEqual(len(engine.get_history()), 4)

    def test_reset_history_clears_memory(self):
        engine = make_engine()
        engine._client.chat.return_value = {"message": {"role": "assistant", "content": "ok"}}
        engine.generate_response("Bonjour")
        self.assertTrue(engine.get_history())

        engine.reset_history()
        self.assertEqual(engine.get_history(), [])


@unittest.skipUnless(
    os.environ.get("RUN_INTEGRATION") == "1",
    "Test d'intégration désactivé par défaut (nécessite Ollama + qwen2.5:3b actifs). "
    "Lancer avec RUN_INTEGRATION=1 pour l'activer.",
)
class TestLLMEngineIntegration(unittest.TestCase):
    """Test réel contre Ollama. Nécessite : ollama serve + qwen2.5:3b installé."""

    def test_real_call_returns_nonempty_response(self):
        engine = LLMEngine()
        reponse = engine.generate_response("Réponds uniquement par 'ok'.")
        self.assertIsInstance(reponse, str)
        self.assertTrue(len(reponse) > 0)


if __name__ == "__main__":
    unittest.main()
