# -*- coding: utf-8 -*-
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import random
import logging
import json
import uuid
import uvicorn
from spellchecker import SpellChecker
import os
from pathlib import Path
from typing import Optional

# Initialisation de l'application
app = FastAPI(
    title="API Chatbot Intelligent",
    version="2.0",
    description="API de chatbot utilisant DistilBERT pour la compréhension du langage naturel"
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # À restreindre en production
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Configuration des chemins
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "model_cache"  # Changement de nom pour plus de clarté
DATA_PATH = BASE_DIR / "data.json"

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("chatbot.log")
    ]
)
logger = logging.getLogger(__name__)

# Optimisation des performances
torch.set_num_threads(1)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Modèles Pydantic
class ChatRequest(BaseModel):
    text: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    intent: str
    confidence: float
    error: Optional[str] = None
    error_id: Optional[str] = None

# Variables globales
tokenizer = None
model = None
spell = None
intents_data = None
label_map = None
reverse_label_map = None

def initialize_components():
    """Initialise tous les composants nécessaires"""
    global tokenizer, model, spell, intents_data, label_map, reverse_label_map
    
    try:
        # Création du dossier de cache
        os.makedirs(MODEL_PATH, exist_ok=True)
        
        logger.info("Initialisation du tokenizer et du modèle...")
        
        # Charge le tokenizer et le modèle depuis Hugging Face
        model_name = "distilbert-base-multilingual-cased"  # Modèle optimisé multilingue
        
        tokenizer = DistilBertTokenizer.from_pretrained(
            model_name,
            cache_dir=str(MODEL_PATH)
        
        model = DistilBertForSequenceClassification.from_pretrained(
            model_name,
            cache_dir=str(MODEL_PATH),
            torch_dtype=torch.float16  # Réduction de l'utilisation mémoire
        ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        model.eval()

        # Charge les données d'intention locales
        logger.info("Chargement des données d'intention...")
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            intents_data = json.load(f)
        
        # Crée les mappings d'étiquettes
        label_map = {intent["intent"]: idx for idx, intent in enumerate(intents_data["intents"])}
        reverse_label_map = {v: k for k, v in label_map.items()}

        # Initialise le correcteur orthographique français
        spell = SpellChecker(language='fr')
        
        logger.info("Initialisation terminée avec succès")

    except Exception as e:
        logger.critical(f"Échec critique de l'initialisation: {str(e)}", exc_info=True)
        raise

@app.on_event("startup")
async def startup_event():
    """Exécuté au démarrage de l'application"""
    initialize_components()
    logger.info("Application prête à recevoir des requêtes")

def correct_spelling(text: str) -> str:
    """Corrige l'orthographe du texte en français"""
    try:
        words = text.split()
        corrected = [spell.correction(word) or word for word in words]
        return " ".join(corrected)
    except Exception as e:
        logger.warning(f"Échec de la correction orthographique: {str(e)}")
        return text

@app.post("/chat", response_model=ChatResponse, summary="Interagir avec le chatbot")
async def chat_endpoint(request: ChatRequest):
    """
    Endpoint principal pour interagir avec le chatbot.
    
    Args:
        request: Contient le texte de l'utilisateur et un ID de session optionnel
    
    Returns:
        Réponse structurée du chatbot avec intention détectée et niveau de confiance
    """
    try:
        # Validation de l'entrée
        if not request.text.strip():
            raise ValueError("Le texte ne peut pas être vide")
        
        logger.info(f"Requête reçue (session: {request.session_id}): {request.text[:50]}...")

        # Correction orthographique
        corrected_text = correct_spelling(request.text)
        if corrected_text != request.text:
            logger.debug(f"Texte corrigé: {corrected_text}")

        # Tokenization et prédiction
        inputs = tokenizer(
            corrected_text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128  # Augmenté pour meilleure compréhension
        ).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs)
        
        # Traitement des résultats
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        confidence = torch.max(probs).item()
        intent = reverse_label_map.get(torch.argmax(probs).item(), "fallback")

        # Gestion des faibles confiances
        if confidence < 0.3:
            intent = "fallback"
            logger.warning(f"Confiance faible ({confidence:.2%}) - Utilisation du fallback")

        # Sélection de la réponse
        intent_data = next(
            (item for item in intents_data["intents"] if item["intent"] == intent), 
            None
        )
        if not intent_data:
            raise ValueError(f"Intention '{intent}' non trouvée dans les données")
            
        response = random.choice(intent_data["responses"])

        logger.info(f"Réponse générée - Intent: {intent}, Confiance: {confidence:.2%}")

        return ChatResponse(
            response=response,
            intent=intent,
            confidence=confidence
        )

    except Exception as e:
        error_id = uuid.uuid4().hex
        logger.error(f"Erreur [{error_id}]: {str(e)}", exc_info=True)
        return ChatResponse(
            response="Désolé, une erreur technique est survenue. Veuillez réessayer.",
            intent="error",
            confidence=0.0,
            error=str(e),
            error_id=error_id
        )

@app.get("/health", summary="Vérifier l'état de l'API")
async def health_check():
    """
    Endpoint de vérification de santé de l'application
    
    Returns:
        État de santé et composants chargés
    """
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "components_ready": all([
            tokenizer is not None,
            model is not None,
            intents_data is not None
        ])
    }

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", "10000")),
        workers=int(os.getenv("WORKERS", "1")),
        log_level="info",
        timeout_keep_alive=120
    )