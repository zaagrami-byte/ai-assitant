"""
tts.py
------
Module de synthese vocale (Text-to-Speech) local, base sur Piper.

Role : recevoir un texte (la reponse du LLM) et le faire entendre via le
haut-parleur. Ne connait rien du micro, du STT ni du LLM : independant et
testable seul, comme llm.py.

Utilisation minimale :

    from tts import speak

    speak("Bonjour, je suis l'assistant d'accueil.")

----------------------------------------------------------------------
Pourquoi Piper (et pas autre chose) ?
----------------------------------------------------------------------
Comparatif rapide des solutions TTS locales envisageables pour ce projet :

  Piper (ONNX, CPU)   : qualite bonne (voix naturelle), latence faible
                        (proche du temps reel), ~150 Mo RAM par voix,
                        pense pour tourner sur Raspberry Pi (projet
                        officiel Home Assistant qui cible ce materiel),
                        API Python simple, un seul fichier modele/voix.

  eSpeak-NG           : latence tres faible, ~20 Mo RAM, mais rendu tres
                        robotique - peu agreable pour un robot d'accueil.

  Coqui TTS (XTTS-v2) : excellente qualite mais tres lourd (1-4 Go+ RAM,
                        plusieurs secondes par phrase en CPU-only) -
                        irrealiste sur Raspberry Pi 4.

  gTTS / solutions cloud : bonne qualite mais necessitent Internet -
                        exclu par la contrainte "fonctionnement local".

Choix retenu : Piper, voix francaise "fr_FR-siwis-medium". C'est le seul
candidat qui coche toutes les cases : qualite convenable, latence
compatible avec une conversation, 100% local, leger, et deja valide par
la communaute sur Raspberry Pi 4.

----------------------------------------------------------------------
Installation (a faire une seule fois) :
----------------------------------------------------------------------
    pip install -r requirements.txt

    mkdir -p models
    curl -L -o models/fr_FR-siwis-medium.onnx \
      "https://huggingface.co/rhasspy/piper-voices/resolve/main/fr/fr_FR/siwis/medium/fr_FR-siwis-medium.onnx"
    curl -L -o models/fr_FR-siwis-medium.onnx.json \
      "https://huggingface.co/rhasspy/piper-voices/resolve/main/fr/fr_FR/siwis/medium/fr_FR-siwis-medium.onnx.json"

    # Verifier qu'un lecteur audio est disponible (l'un des deux suffit) :
    which paplay || which aplay
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import time
import wave

import config

# ─────────────────────────────────────────────────────────────
# LOGGING (meme fichier de log que llm.py, pour un historique unifie)
# ─────────────────────────────────────────────────────────────

os.makedirs(config.LOG_DIR, exist_ok=True)

logger = logging.getLogger("robot_assistant.tts")
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
# ERREURS DEDIEES
# ─────────────────────────────────────────────────────────────


class TTSError(Exception):
    """Erreur generique du module TTS."""


class TTSEmptyTextError(TTSError):
    """Texte vide ou None passe a speak()."""


class TTSModelNotFoundError(TTSError):
    """Fichiers de modele Piper (.onnx / .onnx.json) introuvables."""


class TTSSynthesisError(TTSError):
    """La generation audio a echoue (modele corrompu, texte invalide, etc.)."""


class TTSPlaybackError(TTSError):
    """Impossible de lire l'audio (pas de lecteur, peripherique absent, etc.)."""


# ─────────────────────────────────────────────────────────────
# CHARGEMENT DE LA VOIX (une seule fois, en cache)
# ─────────────────────────────────────────────────────────────

_voice = None  # instance PiperVoice, chargee paresseusement


def _load_voice():
    """Charge le modele Piper en memoire (une seule fois par process)."""
    global _voice
    if _voice is not None:
        return _voice

    if not os.path.isfile(config.TTS_MODEL_PATH) or not os.path.isfile(config.TTS_CONFIG_PATH):
        logger.error(
            "Modele Piper introuvable : %s / %s",
            config.TTS_MODEL_PATH, config.TTS_CONFIG_PATH,
        )
        raise TTSModelNotFoundError(
            f"Modele de voix introuvable dans '{config.TTS_MODELS_DIR}'. "
            f"Voir les instructions de telechargement en haut de tts.py."
        )

    try:
        # Import differe : evite de charger onnxruntime si tts.py n'est
        # importe que pour des tests qui ne touchent pas la vraie synthese.
        from piper.voice import PiperVoice

        start = time.perf_counter()
        _voice = PiperVoice.load(config.TTS_MODEL_PATH, config.TTS_CONFIG_PATH)
        logger.info("Voix Piper chargee en %.2fs (%s).",
                     time.perf_counter() - start, config.TTS_VOICE_NAME)
        return _voice
    except TTSModelNotFoundError:
        raise
    except Exception as exc:
        logger.exception("Echec du chargement du modele Piper.")
        raise TTSModelNotFoundError(f"Impossible de charger le modele Piper : {exc}") from exc


def _detect_player() -> str:
    """Determine la commande systeme a utiliser pour lire un fichier .wav."""
    if config.TTS_PLAYER_CMD:
        return config.TTS_PLAYER_CMD
    for candidate in ("paplay", "aplay"):
        if shutil.which(candidate):
            return candidate
    raise TTSPlaybackError(
        "Aucun lecteur audio trouve (ni 'paplay' ni 'aplay'). "
        "Installe alsa-utils ('sudo apt install alsa-utils') ou pulseaudio-utils."
    )


# ─────────────────────────────────────────────────────────────
# API PUBLIQUE
# ─────────────────────────────────────────────────────────────


def speak(text: str, play_audio: bool = True) -> dict:
    """Synthetise `text` en audio et le joue sur le haut-parleur.

    Retourne un petit dict de metriques utile pour les logs/tests :
        {"synthesis_time": float, "playback_time": float, "wav_path": str}

    `play_audio=False` permet de ne faire QUE la synthese (utile pour des
    tests automatises sans haut-parleur, ou pour un futur mode "genere les
    fichiers audio a l'avance").

    Leve une sous-classe de TTSError en cas de probleme.
    """
    if not text or not text.strip():
        logger.warning("speak() appele avec un texte vide, ignore.")
        raise TTSEmptyTextError("Le texte a synthetiser est vide.")

    text = text.strip()
    voice = _load_voice()

    os.makedirs(config.TTS_TMP_DIR, exist_ok=True)
    wav_path = None

    # ---- Synthese ----
    synth_start = time.perf_counter()
    try:
        fd, wav_path = tempfile.mkstemp(suffix=".wav", dir=config.TTS_TMP_DIR)
        os.close(fd)
        with wave.open(wav_path, "wb") as wav_file:
            voice.synthesize_wav(text, wav_file)
    except Exception as exc:
        logger.exception("Echec de la synthese audio.")
        if wav_path and os.path.exists(wav_path):
            os.remove(wav_path)
        raise TTSSynthesisError(f"Echec de la synthese audio : {exc}") from exc
    synthesis_time = time.perf_counter() - synth_start

    # ---- Lecture ----
    playback_time = 0.0
    if play_audio:
        player = _detect_player()
        playback_start = time.perf_counter()
        try:
            subprocess.run(
                [player, wav_path],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                timeout=30,
            )
        except subprocess.CalledProcessError as exc:
            logger.error("Le lecteur audio '%s' a echoue : %s", player, exc.stderr)
            raise TTSPlaybackError(f"Echec de lecture avec '{player}': {exc.stderr}") from exc
        except subprocess.TimeoutExpired as exc:
            logger.error("Timeout de lecture audio (>30s) avec '%s'.", player)
            raise TTSPlaybackError("La lecture audio a depasse le delai autorise.") from exc
        finally:
            playback_time = time.perf_counter() - playback_start
    else:
        logger.debug("play_audio=False : synthese uniquement, pas de lecture.")

    total = synthesis_time + playback_time
    logger.info(
        "TTS latency: synth=%.2fs play=%.2fs total=%.2fs | texte='%s'",
        synthesis_time, playback_time, total, _truncate(text),
    )

    if play_audio and wav_path and os.path.exists(wav_path):
        try:
            os.remove(wav_path)
        except OSError:
            pass  # non bloquant : au pire le fichier reste dans tmp_audio/

    return {"synthesis_time": synthesis_time, "playback_time": playback_time, "wav_path": wav_path}


def _truncate(text: str, length: int = 60) -> str:
    return text if len(text) <= length else text[: length - 1] + "…"


# ─────────────────────────────────────────────────────────────
# TEST MANUEL RAPIDE : python tts.py
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Test manuel de tts.py ===")
    print(f"Voix : {config.TTS_VOICE_NAME}\n")

    test_cases = [
        ("Phrase courte", "Bonjour, bienvenue."),
        ("Phrase longue", (
            "Bonjour et bienvenue. Je suis l'assistant vocal d'accueil de ce "
            "batiment. N'hesitez pas a me poser vos questions, je ferai de mon "
            "mieux pour vous aider rapidement."
        )),
        ("Caracteres accentues", "Elephant, foret, hopital, a cote, ou etes-vous ?"),
        ("Texte vide (doit echouer proprement)", ""),
    ]

    for label, text in test_cases:
        print(f"[{label}]")
        try:
            metrics = speak(text)
            print(
                f"  OK - synthese: {metrics['synthesis_time']:.2f}s, "
                f"lecture: {metrics['playback_time']:.2f}s\n"
            )
        except TTSError as e:
            print(f"  [ERREUR attendue ou reelle] {e}\n")

    print("=== Fin du test ===")
