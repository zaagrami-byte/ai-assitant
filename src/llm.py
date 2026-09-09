"""
llm.py
------
Module de communication avec le LLM local (Qwen2.5:3B via Ollama).

Rôle : recevoir un texte utilisateur, l'envoyer au modèle avec le
system prompt + l'historique de conversation, et retourner uniquement
le texte de la réponse finale — prêt à être transmis au module TTS.

Ce module ne connaît RIEN du micro, du STT ni du TTS : il est totalement
indépendant (testable seul), ce qui permettra plus tard de le brancher
tel quel dans une architecture ROS 2 sans le modifier.

Utilisation minimale :

    from llm import LLMEngine

    engine = LLMEngine()
    reponse = engine.generate_response("Bonjour, qui êtes-vous ?")
    print(reponse)
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field

import ollama

import config

# ─────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────

os.makedirs(config.LOG_DIR, exist_ok=True)

logger = logging.getLogger("robot_assistant.llm")
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


# ─────────────────────────────────────────────────────────────
# ERREURS DÉDIÉES
# ─────────────────────────────────────────────────────────────


class LLMError(Exception):
    """Erreur générique du module LLM. Toutes les erreurs ci-dessous en héritent,
    ce qui permet à assistant.py de faire un seul `except LLMError` s'il le souhaite."""


class LLMUnavailableError(LLMError):
    """Ollama n'est pas joignable (service arrêté, mauvais host, etc.)."""


class LLMModelNotFoundError(LLMError):
    """Le modèle demandé n'est pas installé/disponible dans Ollama."""


class LLMTimeoutError(LLMError):
    """Le modèle a mis trop de temps à répondre."""


class LLMEmptyResponseError(LLMError):
    """Ollama a répondu, mais sans contenu exploitable."""


# ─────────────────────────────────────────────────────────────
# MOTEUR LLM
# ─────────────────────────────────────────────────────────────


@dataclass
class LLMEngine:
    """Encapsule la conversation avec Qwen2.5:3B.

    Garde l'historique en mémoire (liste de dicts {role, content}) afin que
    le modèle se souvienne du contexte d'une phrase à l'autre, sans base
    de données : c'est la mémoire "légère" demandée pour cette phase.
    """

    model: str = config.LLM_MODEL
    system_prompt: str = config.SYSTEM_PROMPT
    max_history: int = config.MAX_HISTORY
    timeout: int = config.LLM_TIMEOUT
    host: str = config.OLLAMA_HOST

    _history: list[dict] = field(default_factory=list, init=False, repr=False)
    _client: ollama.Client = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._client = ollama.Client(host=self.host, timeout=self.timeout)
        logger.info(
            "LLMEngine initialisé (modèle=%s, host=%s, timeout=%ss, max_history=%s)",
            self.model, self.host, self.timeout, self.max_history,
        )

    # ------------------------------------------------------------------
    # API publique
    # ------------------------------------------------------------------

    def generate_response(self, user_text: str) -> str:
        """Envoie `user_text` au modèle et retourne la réponse texte finale.

        Lève une sous-classe de LLMError en cas de problème — à charge de
        l'appelant (assistant.py) de décider quoi dire à l'utilisateur
        (ex : "Je rencontre un problème technique, un instant.").
        """
        if not user_text or not user_text.strip():
            logger.warning("generate_response appelé avec un texte vide, ignoré.")
            raise LLMEmptyResponseError("Le texte utilisateur est vide.")

        user_text = user_text.strip()
        messages = self._build_messages(user_text)

        start = time.perf_counter()
        try:
            result = self._client.chat(model=self.model, messages=messages)
        except ollama.ResponseError as exc:
            # Ollama répond mais signale une erreur (ex : modèle absent -> 404)
            if exc.status_code == 404:
                logger.error("Modèle '%s' introuvable dans Ollama.", self.model)
                raise LLMModelNotFoundError(
                    f"Le modèle '{self.model}' n'est pas installé. "
                    f"Lance : ollama pull {self.model}"
                ) from exc
            logger.error("Erreur Ollama (status=%s): %s", exc.status_code, exc.error)
            raise LLMError(f"Erreur Ollama : {exc.error}") from exc
        except TimeoutError as exc:
            logger.error("Timeout (>%ss) en attendant la réponse d'Ollama.", self.timeout)
            raise LLMTimeoutError(
                f"Le modèle n'a pas répondu en moins de {self.timeout}s."
            ) from exc
        except ConnectionError as exc:
            logger.error("Impossible de joindre Ollama sur %s : %s", self.host, exc)
            raise LLMUnavailableError(
                f"Ollama est injoignable sur {self.host}. Le service est-il démarré ? "
                f"(commande : 'ollama serve')"
            ) from exc
        except Exception as exc:  # filet de sécurité : ne jamais planter le robot
            logger.exception("Erreur inattendue lors de l'appel LLM.")
            raise LLMError(f"Erreur inattendue : {exc}") from exc

        latency = time.perf_counter() - start

        content = (result.get("message") or {}).get("content", "").strip()
        if not content:
            logger.warning("Réponse vide reçue d'Ollama après %.2fs.", latency)
            raise LLMEmptyResponseError("Le modèle a renvoyé une réponse vide.")

        logger.info("LLM latency: %.2fs | user='%s' -> reponse='%s'",
                     latency, self._truncate(user_text), self._truncate(content))

        self._update_history(user_text, content)
        return content

    def reset_history(self) -> None:
        """Efface la mémoire de conversation (ex : sur commande 'nouvelle conversation')."""
        self._history.clear()
        logger.info("Historique de conversation réinitialisé.")

    def get_history(self) -> list[dict]:
        """Retourne une copie de l'historique actuel (utile pour debug/tests)."""
        return list(self._history)

    # ------------------------------------------------------------------
    # Internes
    # ------------------------------------------------------------------

    def _build_messages(self, user_text: str) -> list[dict]:
        return (
            [{"role": "system", "content": self.system_prompt}]
            + self._history
            + [{"role": "user", "content": user_text}]
        )

    def _update_history(self, user_text: str, assistant_text: str) -> None:
        self._history.append({"role": "user", "content": user_text})
        self._history.append({"role": "assistant", "content": assistant_text})

        # On garde max_history ÉCHANGES, donc max_history * 2 messages.
        max_messages = self.max_history * 2
        if len(self._history) > max_messages:
            overflow = len(self._history) - max_messages
            del self._history[:overflow]
            logger.debug("Historique tronqué (%d messages retirés).", overflow)

    @staticmethod
    def _truncate(text: str, length: int = 60) -> str:
        return text if len(text) <= length else text[: length - 1] + "…"


# ─────────────────────────────────────────────────────────────
# TEST MANUEL RAPIDE : python llm.py
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Test manuel de llm.py ===")
    print(f"Modèle : {config.LLM_MODEL} | Host : {config.OLLAMA_HOST}\n")

    engine = LLMEngine()

    test_questions = [
        "Bonjour, qui êtes-vous ?",
        "Je m'appelle Rami.",
        "Quel est mon prénom ?",
    ]

    for question in test_questions:
        print(f"Utilisateur : {question}")
        try:
            start = time.perf_counter()
            reponse = engine.generate_response(question)
            elapsed = time.perf_counter() - start
            print(f"Assistant   : {reponse}")
            print(f"(latence: {elapsed:.2f}s)\n")
        except LLMError as e:
            print(f"[ERREUR] {e}\n")

    print("=== Fin du test ===")
