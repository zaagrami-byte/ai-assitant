"""
assistant.py
------------
Orchestration du pipeline complet : STT -> LLM -> TTS.

Ce module ne reimplemente aucune logique deja validee dans stt.py, llm.py
ou tts.py : il se contente d'enchainer les trois, de gerer les erreurs de
chaque etage sans jamais planter le robot, et de permettre l'arret propre
de la conversation via un mot-cle prononce ("stop" / "arret").

Lancer la conversation :

    python assistant.py

Arreter en parlant :

    "stop" ou "arrêt" (ou Ctrl+C dans le terminal)
"""

from __future__ import annotations

import logging
import os
import time
import unicodedata

import config
from llm import LLMEngine, LLMError
from stt import STTEngine, STTError, STTEmptyAudioError
from tts import speak, TTSError

# ─────────────────────────────────────────────────────────────
# LOGGING (meme fichier que les autres modules, historique unifie)
# ─────────────────────────────────────────────────────────────

os.makedirs(config.LOG_DIR, exist_ok=True)

logger = logging.getLogger("robot_assistant.assistant")
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
# UTILITAIRE — normalisation de texte pour detecter "stop"/"arret"
# ─────────────────────────────────────────────────────────────


def _normalize(text: str) -> str:
    """Minuscules + sans accents, pour comparer robustement a ASSISTANT_STOP_WORDS.
    Ex: 'Arrêt !' -> 'arret !' """
    text = text.lower().strip()
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if not unicodedata.combining(c))


def is_stop_command(text: str) -> bool:
    """Vrai si le texte reconnu correspond a une commande d'arret."""
    normalized = _normalize(text)
    # On retire la ponctuation finale simple pour comparer proprement
    normalized = normalized.rstrip(" !.?")
    return normalized in config.ASSISTANT_STOP_WORDS


# ─────────────────────────────────────────────────────────────
# ORCHESTRATEUR
# ─────────────────────────────────────────────────────────────


class Assistant:
    """Enchaine STT -> LLM -> TTS pour une conversation vocale continue.

    Les moteurs sont injectables (utile pour les tests) mais utilisent par
    defaut les implementations reelles deja validees.
    """

    def __init__(
        self,
        stt_engine: STTEngine | None = None,
        llm_engine: LLMEngine | None = None,
        speak_fn=speak,
    ) -> None:
        self.stt = stt_engine or STTEngine()
        self.llm = llm_engine or LLMEngine()
        self.speak_fn = speak_fn
        logger.info("Assistant initialise.")

    # ------------------------------------------------------------------
    # Boucle principale
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Lance la conversation en continu jusqu'a une commande d'arret
        ou une interruption clavier (Ctrl+C)."""
        print("Assistant demarre. Parlez a tout moment (dites 'stop' pour arreter).\n")
        logger.info("Boucle assistant demarree.")

        try:
            while True:
                should_continue = self.run_one_turn()
                if not should_continue:
                    break
        except KeyboardInterrupt:
            print("\nAssistant arrete (Ctrl+C).")
            logger.info("Assistant arrete par l'utilisateur (Ctrl+C).")

    # ------------------------------------------------------------------
    # Un seul echange complet (facilement testable isolement)
    # ------------------------------------------------------------------

    def run_one_turn(self) -> bool:
        """Execute un cycle Ecoute -> LLM -> Parole.

        Retourne False si la conversation doit s'arreter (commande d'arret
        reconnue), True sinon. Ne leve jamais d'exception : toute erreur
        d'un etage est loguee et signalee vocalement quand possible.
        """
        print("Ecoute...")
        try:
            start = time.perf_counter()
            user_text = self.stt.listen()
            elapsed = time.perf_counter() - start
        except STTEmptyAudioError:
            # Silence : rien a faire, on reecoute simplement (pas une vraie erreur).
            logger.debug("Silence detecte, nouvelle ecoute.")
            return True
        except STTError as exc:
            logger.error("Erreur STT : %s", exc)
            self._safe_speak(config.ASSISTANT_ERROR_MESSAGE)
            return True

        print(f"Utilisateur : {user_text}  (latence STT: {elapsed:.2f}s)")
        logger.info("STT -> texte reconnu : %s", user_text)

        if is_stop_command(user_text):
            logger.info("Commande d'arret reconnue : '%s'", user_text)
            self._safe_speak("À bientôt !")
            print("Assistant arrete (commande vocale).")
            return False

        try:
            start = time.perf_counter()
            response_text = self.llm.generate_response(user_text)
            elapsed = time.perf_counter() - start
        except LLMError as exc:
            logger.error("Erreur LLM : %s", exc)
            self._safe_speak(config.ASSISTANT_ERROR_MESSAGE)
            return True

        print(f"Assistant   : {response_text}  (latence LLM: {elapsed:.2f}s)")
        logger.info("LLM -> reponse generee : %s", response_text)

        self._safe_speak(response_text)
        return True

    # ------------------------------------------------------------------
    # Internes
    # ------------------------------------------------------------------

    def _safe_speak(self, text: str) -> None:
        """Appelle le TTS sans jamais faire planter la boucle assistant."""
        try:
            self.speak_fn(text)
        except TTSError as exc:
            logger.error("Erreur TTS : %s", exc)
            print(f"[TTS indisponible] {text}")


# ─────────────────────────────────────────────────────────────
# POINT D'ENTREE
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    assistant = Assistant()
    assistant.run()