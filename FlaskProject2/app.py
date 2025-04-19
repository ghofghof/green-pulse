from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import re
from sentence_transformers import SentenceTransformer, util
import torch
import requests
import json

app = Flask(__name__)

# === Configuration ===
MISTRAL_API_KEY = "xYaq8vnQzSbmKjZGvnefLa2ugS1jZnj2"
MISTRAL_MODEL_ID = "ft:open-mistral-7b:70f38c7a:20250416:2b82656c"
EMBEDDING_MODEL = "paraphrase-MiniLM-L6-v2"

# === Initialisation des modèles ===
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

# === Chargement des données ===
with open('data/referentiel.json', 'r', encoding='utf-8') as f:
    problemes_referentiel = json.load(f)

df_merged = pd.read_csv('data/equipment_data.csv')


def ask_mistral_llm(categorie, sous_categorie, query):
    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }

    prompt = f"""
Tu es un expert en durabilité industrielle.

Tu reçois une requête liée à un problème : "{query}"
Catégorie : {categorie}
Sous-catégorie : {sous_categorie}

Fournis **au maximum 5 actions correctives ou recommandations** concrètes et efficaces, sous forme de liste numérotée.

Ne donne que les actions (pas de texte introductif ou conclusion).
"""

    data = {
        "model": MISTRAL_MODEL_ID,
        "messages": [
            {"role": "system",
             "content": "Tu es un expert en durabilité industrielle. Tu proposes des actions écologiques concrètes."},
            {"role": "user", "content": prompt}
        ]
    }

    response = requests.post(url, headers=headers, json=data)
    return response.json()['choices'][0]['message']['content']


def calculate_dynamic_thresholds(df, problem):
    thresholds = {}
    for indicateur, condition in problem["Seuils"].items():
        if 'P' in condition:
            percentile = float(re.findall(r'P(\d+)', condition)[0])
            thresholds[indicateur] = np.percentile(df[indicateur], percentile)
        elif 'x référence' in condition:
            ref_value = df[indicateur].median()
            factor = float(re.findall(r'(\d+(\.\d+)?)', condition)[0][0])
            thresholds[indicateur] = ref_value * factor
        elif any(op in condition for op in ['>', '<', '==', '!=']):
            thresholds[indicateur] = condition
        else:
            thresholds[indicateur] = float(re.findall(r'\d+', condition)[0])
    return thresholds


def apply_thresholds(df, problem):
    thresholds = calculate_dynamic_thresholds(df, problem)
    filtered = df.copy()
    for indicateur, condition in problem["Seuils"].items():
        try:
            if 'P' in condition or 'x référence' in condition:
                filtered = filtered[filtered[indicateur] > thresholds[indicateur]]
            elif any(op in condition for op in ['>', '<', '==', '!=']):
                filtered = filtered.query(f"{indicateur} {condition}")
            else:
                filtered = filtered[filtered[indicateur] > thresholds[indicateur]]
        except Exception as e:
            print(f"Erreur de filtrage pour {indicateur}: {str(e)}")
    return filtered


def find_closest_problem_by_input(query, threshold=0.5):
    referential_texts = [p["Model_input"] for p in problemes_referentiel]
    query_embedding = embedding_model.encode(query, convert_to_tensor=True)
    referential_embeddings = embedding_model.encode(referential_texts, convert_to_tensor=True)
    cos_scores = util.cos_sim(query_embedding, referential_embeddings)[0]
    best_idx = int(torch.argmax(cos_scores))
    if cos_scores[best_idx] > threshold:
        return problemes_referentiel[best_idx]
    return None


@app.route('/')
def home():
    return render_template('chat.html')


@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json()
    query = data.get('message', '').strip()

    if not query:
        return jsonify({"error": "Veuillez entrer une requête"}), 400

    try:
        # Générer réponse avec Mistral
        mistral_output = ask_mistral_llm("Générique", "N/A", query)

        # Trouver le problème correspondant
        matched_problem = find_closest_problem_by_input(query)
        anomalies = None

        if matched_problem:
            df_filtered = apply_thresholds(df_merged, matched_problem)
            cols = ['Equipment_ID', 'Equipment_Name', 'Supplier_Name'] + matched_problem['Indicateurs']
            anomalies = df_filtered[cols].to_dict('records')

        return jsonify({
            "response": mistral_output,
            "problem": matched_problem,
            "anomalies": anomalies
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)