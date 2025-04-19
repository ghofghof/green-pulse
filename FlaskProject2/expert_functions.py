import numpy as np
import pandas as pd
import re
import json
from sentence_transformers import SentenceTransformer, util
import torch
import requests
from config import Config

embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)


def load_referential(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_equipment_data(filepath):
    return pd.read_csv(filepath)


def ask_mistral_llm(categorie, sous_categorie, query):
    headers = {
        "Authorization": f"Bearer {Config.MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }

    prompt = f"""Tu es un expert en durabilité industrielle. Pour le problème suivant :
"{query}" (Catégorie: {categorie}/{sous_categorie}), 
donne 5 recommandations concrètes sous forme de liste numérotée."""

    data = {
        "model": Config.MISTRAL_MODEL_ID,
        "messages": [
            {"role": "system", "content": "Expert en solutions industrielles durables"},
            {"role": "user", "content": prompt}
        ]
    }

    response = requests.post(
        "https://api.mistral.ai/v1/chat/completions",
        headers=headers,
        json=data
    )
    return response.json()['choices'][0]['message']['content']


def find_closest_problem_by_input(query, referentiel, threshold=0.5):
    texts = [p["Model_input"] for p in referentiel]
    query_embed = embedding_model.encode(query, convert_to_tensor=True)
    ref_embeds = embedding_model.encode(texts, convert_to_tensor=True)

    scores = util.cos_sim(query_embed, ref_embeds)[0]
    best_idx = torch.argmax(scores).item()

    return referentiel[best_idx] if scores[best_idx] > threshold else None


def apply_thresholds(df, problem):
    # Implémentation de la logique des seuils
    filtered = df.copy()
    for indicator, condition in problem["Seuils"].items():
        if indicator in df.columns:
            try:
                if '>' in condition:
                    filtered = filtered[filtered[indicator] > float(condition.split('>')[1])]
                elif '<' in condition:
                    filtered = filtered[filtered[indicator] < float(condition.split('<')[1])]
            except:
                continue
    return filtered.to_dict('records')