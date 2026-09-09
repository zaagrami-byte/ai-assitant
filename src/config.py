"""
config.py
---------
Configuration centrale du robot d'accueil.

Objectif : éviter de disperser des paramètres "en dur" dans plusieurs
fichiers Python. Toute valeur qu'on pourrait vouloir changer plus tard
(modèle, prompt, timeout, taille d'historique...) vit ici.

Pour l'instant les valeurs sont des constantes Python simples.
Si besoin plus tard (secrets, déploiement Pi vs PC), on pourra basculer
sur des variables d'environnement (.env) sans changer le reste du code,
car tous les autres modules importent depuis CE fichier uniquement.
"""

import os

# ─────────────────────────────────────────────────────────────
# OLLAMA / LLM
# ─────────────────────────────────────────────────────────────

# Adresse du serveur Ollama local (par défaut, inchangée sauf cas particulier)
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

# Modèle utilisé pour la génération de réponses
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen2.5:3b")

# Délai maximum (en secondes) qu'on autorise à Ollama pour répondre
# avant de considérer que quelque chose ne va pas (service bloqué, etc.)
LLM_TIMEOUT = 20

# Nombre max d'échanges (paire utilisateur + assistant) conservés en mémoire.
# Limite la consommation de RAM et de tokens envoyés au modèle à chaque appel.
MAX_HISTORY = 10

# ─────────────────────────────────────────────────────────────
# SYSTEM PROMPT — comportement du robot d'accueil
# ─────────────────────────────────────────────────────────────
#
# Chaque ligne du prompt cible une contrainte précise du cahier des charges :
#  - identité claire et rôle fixé -> évite les dérives ("je suis un LLM...")
#  - consigne de brièveté -> essentiel en vocal (une réponse de 10 lignes
#    est pénible à écouter et fait exploser la latence TTS)
#  - interdiction d'exposer le raisonnement interne -> le modèle ne doit
#    jamais produire de "réflexion" façon chain-of-thought à voix haute
#  - consigne de langue -> répond dans la langue de l'utilisateur, avec le
#    français comme langue par défaut
#  - ton poli/chaleureux -> cohérent avec un rôle d'accueil (hôpital, etc.)
SYSTEM_PROMPT = (
    "Tu es l'assistant vocal d'un robot physique. "
    "Tu accueilles des personnes en face à face, à voix haute. "
    "Règles strictes :\n"
    "1. Réponds toujours de manière brève (1 à 3 phrases maximum), claire et naturelle, "
    "comme dans une vraie conversation orale — jamais de longues listes ni de blocs de texte.\n"
    "2. Sois poli, chaleureux et professionnel, sans être familier.\n"
    "3. Réponds dans la langue utilisée par l'utilisateur ; si tu n'es pas certain, réponds en français.\n"
    "4. Ne montre jamais ton raisonnement interne, tes étapes de réflexion, ni de balises "
    "techniques : donne uniquement la réponse finale destinée à être entendue.\n"
    "5. Ne parle de ton fonctionnement technique (modèle, IA, système) que si on te le demande "
    "explicitement.\n"
    "6. Si tu ne sais pas répondre, dis-le simplement et propose d'orienter la personne vers un humain."
)

# ─────────────────────────────────────────────────────────────
# TTS (Piper)
# ─────────────────────────────────────────────────────────────

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Dossier où sont stockés les modèles de voix Piper (.onnx + .onnx.json)
TTS_MODELS_DIR = os.environ.get("TTS_MODELS_DIR", os.path.join(_BASE_DIR, "models"))

# Nom de la voix Piper utilisée (voix française, qualité "medium" = bon
# compromis qualité/vitesse/RAM, adaptée au Raspberry Pi 4).
TTS_VOICE_NAME = os.environ.get("TTS_VOICE_NAME", "fr_FR-siwis-medium")

TTS_MODEL_PATH = os.path.join(TTS_MODELS_DIR, f"{TTS_VOICE_NAME}.onnx")
TTS_CONFIG_PATH = os.path.join(TTS_MODELS_DIR, f"{TTS_VOICE_NAME}.onnx.json")

# Dossier temporaire pour les fichiers .wav générés avant lecture
TTS_TMP_DIR = os.path.join(_BASE_DIR, "tmp_audio")

# Commande de lecture audio à utiliser explicitement (None = auto-détection
# entre 'paplay' et 'aplay' selon ce qui est disponible sur le système).
TTS_PLAYER_CMD = os.environ.get("TTS_PLAYER_CMD", None)

# ─────────────────────────────────────────────────────────────
# STT (faster-whisper)
# ─────────────────────────────────────────────────────────────

STT_MODEL = os.environ.get("STT_MODEL", "small")
STT_DEVICE = os.environ.get("STT_DEVICE", "cpu")
STT_COMPUTE_TYPE = os.environ.get("STT_COMPUTE_TYPE", "int8")
STT_LANGUAGE = os.environ.get("STT_LANGUAGE", "fr")

STT_SAMPLE_RATE = int(os.environ.get("STT_SAMPLE_RATE", "16000"))
STT_CHANNELS = int(os.environ.get("STT_CHANNELS", "1"))

# Duree max d'un enregistrement, et duree de silence apres parole
# consideree comme "fin de phrase" (VAD simple par seuil d'energie).
STT_MAX_RECORD_SECONDS = int(os.environ.get("STT_MAX_RECORD_SECONDS", "10"))
STT_SILENCE_THRESHOLD = float(os.environ.get("STT_SILENCE_THRESHOLD", "500"))
STT_SILENCE_DURATION = float(os.environ.get("STT_SILENCE_DURATION", "1.2"))

# Duree de mesure du bruit ambiant avant chaque enregistrement (VAD adaptatif).
STT_CALIBRATION_DURATION = float(os.environ.get("STT_CALIBRATION_DURATION", "0.3"))

# Multiplicateur applique au bruit ambiant mesure pour obtenir le seuil de
# detection de parole. Remplace l'usage direct de STT_SILENCE_THRESHOLD comme
# valeur fixe : le seuil s'adapte desormais a l'environnement reel.
STT_THRESHOLD_MULTIPLIER = float(os.environ.get("STT_THRESHOLD_MULTIPLIER", "2.0"))

# Seuil plancher : meme dans une piece tres silencieuse, ne jamais descendre
# en dessous de cette valeur (evite qu'un seuil trop bas confonde le bruit de
# fond residuel avec de la parole).
STT_MIN_THRESHOLD = float(os.environ.get("STT_MIN_THRESHOLD", "150"))

# Dossier dedie au modele Whisper (separe des modeles Piper)
STT_MODEL_DIR = os.environ.get(
    "STT_MODEL_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "whisper"),
)

# Dossier temporaire pour les .wav enregistres avant transcription
STT_TMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp_audio")

# Delai max (secondes) autorise pour la transcription d'une phrase
STT_TIMEOUT = int(os.environ.get("STT_TIMEOUT", "30"))
# ─────────────────────────────────────────────────────────────
# STT — VAD (webrtcvad) : remplace l'ancien seuil d'énergie
# ─────────────────────────────────────────────────────────────

# Agressivité du classifieur webrtcvad : 0 (permissif, laisse passer plus
# de bruit) à 3 (agressif, filtre fort). 3 recommandé pour un environnement
# bruyant type hall d'accueil.
STT_VAD_MODE = int(os.environ.get("STT_VAD_MODE", "3"))

# Taille des frames analysées par webrtcvad. Valeurs autorisées : 10, 20, 30 ms uniquement.
STT_VAD_FRAME_MS = int(os.environ.get("STT_VAD_FRAME_MS", "30"))

# Fenêtre glissante utilisée pour confirmer le DÉBUT de parole (évite qu'un
# bruit ponctuel déclenche un faux départ).
STT_VAD_START_WINDOW_MS = int(os.environ.get("STT_VAD_START_WINDOW_MS", "300"))
STT_VAD_START_RATIO = float(os.environ.get("STT_VAD_START_RATIO", "0.7"))

# Ratio de frames "non-voix" requis dans la fenêtre de fin (durée = STT_SILENCE_DURATION)
# pour considérer que l'utilisateur a fini de parler.
STT_VAD_END_RATIO = float(os.environ.get("STT_VAD_END_RATIO", "0.85"))

# ─────────────────────────────────────────────────────────────
# STT — anti-hallucination faster-whisper
# ─────────────────────────────────────────────────────────────

STT_VAD_FILTER = os.environ.get("STT_VAD_FILTER", "1") == "1"
STT_NO_SPEECH_THRESHOLD = float(os.environ.get("STT_NO_SPEECH_THRESHOLD", "0.6"))
STT_LOG_PROB_THRESHOLD = float(os.environ.get("STT_LOG_PROB_THRESHOLD", "-1.0"))
STT_COMPRESSION_RATIO_THRESHOLD = float(os.environ.get("STT_COMPRESSION_RATIO_THRESHOLD", "2.4"))
STT_CONDITION_ON_PREVIOUS_TEXT = os.environ.get("STT_CONDITION_ON_PREVIOUS_TEXT", "0") == "1"
# beam_size=1 (greedy) pour la latence minimale ; monter à 5 si la qualité
# de transcription pose problème (coût CPU plus élevé, à tester sur Pi 4).
STT_BEAM_SIZE = int(os.environ.get("STT_BEAM_SIZE", "1"))


# ─────────────────────────────────────────────────────────────
# ASSISTANT — orchestration
# ─────────────────────────────────────────────────────────────

# Mots-cles (normalises en minuscules, sans accents) qui arretent la conversation.
ASSISTANT_STOP_WORDS = {"stop", "arret", "arrete", "au revoir"}

# Message parle en cas de probleme technique (LLM, TTS ou STT en erreur recuperable)
ASSISTANT_ERROR_MESSAGE = (
    "Je rencontre un petit problème technique, un instant s'il vous plaît."
)
# ─────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────

LOG_LEVEL = "INFO"
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
LOG_FILE = os.path.join(LOG_DIR, "assistant.log")
