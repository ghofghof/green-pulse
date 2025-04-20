from flask import Flask, render_template, request, jsonify, send_file
from datetime import datetime
import io
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
import joblib
import warnings
import re
from sentence_transformers import SentenceTransformer, util
import torch
import requests
import json
warnings.filterwarnings('ignore')

app = Flask(__name__)


class EnergyAIAssistant:
    def __init__(self):
        self.energy_factors = {
            'Fioul': {'consumption': 1.2, 'emission': 0.25},
            'Gaz': {'consumption': 1.0, 'emission': 0.15},
            'Electricité': {'consumption': 0.8, 'emission': 0.10},
            'Solaire': {'consumption': 0.5, 'emission': 0.02}
        }

        self.equipment_impact = {
            'Production': 1.0,
            'Transformation': 0.9,
            'Refrigeration': 0.8,
            'Conditionnement': 0.6,
            'Nettoyage': 0.4,
            'Controle Qualite': 0.3
        }

    def calculate_smart_consumption(self, energy_type, equipment_type, lifetime):
        base = 3000
        energy_factor = self.energy_factors[energy_type]['consumption']
        equipment_factor = self.equipment_impact[equipment_type]
        age_factor = 1 + (lifetime * 0.02)
        return base * energy_factor * equipment_factor * age_factor

    def calculate_smart_emissions(self, energy_type, equipment_type, consumption):
        emission_factor = self.energy_factors[energy_type]['emission']
        equipment_factor = self.equipment_impact[equipment_type]
        return consumption * emission_factor * equipment_factor


class SmartEnergySystem:
    def __init__(self):
        self.assistant = EnergyAIAssistant()
        self.model = None
        self.load_model()

    def load_model(self):
        try:
            self.model = joblib.load('smart_energy_model_xgboost.pkl')
        except:
            self.train_model()

    def train_model(self):
        data = self.generate_smart_data()
        X = data[['Type_Energy', 'Category_Equipment', 'Estimated_Lifetime_Years']]
        y = data[['Energy_Consumption_kWh', 'CO2_Emissions_kg']]

        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore'), ['Type_Energy', 'Category_Equipment'])
            ],
            remainder='passthrough'
        )

        self.model = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', XGBRegressor(
                n_estimators=150,
                learning_rate=0.1,
                max_depth=6,
                objective='reg:squarederror',
                random_state=42
            ))
        ])

        self.model.fit(X, y)
        joblib.dump(self.model, 'smart_energy_model_xgboost.pkl')

    def predict_energy(self, data):
        input_df = pd.DataFrame([data])
        prediction = self.model.predict(input_df)

        # Convert numpy types to Python native types
        return {
            'kwh': float(prediction[0][0]),
            'co2': float(prediction[0][1]),
            'cost': float(prediction[0][0] * 0.15)
        }

    def generate_recommendations(self, data, results):
        recommendations = []
        energy_type = data['Type_Energy']
        equip_type = data['Category_Equipment']

        if energy_type == 'Fioul':
            recommendations.append(
                "Consider switching to a less polluting energy source like Gaz or Electricité to reduce your carbon footprint by 30-50%.")
        elif energy_type == 'Gaz':
            recommendations.append(
                "To further reduce your impact, consider renewable energies like Solaire which could decrease your emissions by 60%.")

        if equip_type in ['Production', 'Transformation']:
            recommendations.append(
                "Optimize your production processes with more efficient technologies. Modernization could reduce your consumption by 15-25%.")
        elif equip_type == 'Refrigeration':
            recommendations.append(
                "Modern high-efficiency refrigeration systems could reduce your consumption by 20-30%.")

        if results['kwh'] > 3500:
            recommendations.append(
                f"Your consumption is high ({results['kwh']:.2f} kWh). A complete energy audit is recommended to identify savings opportunities.")
        elif results['kwh'] < 2000:
            recommendations.append(
                "Good energy performance. Maintain your best practices and consider energy certifications.")

        recommendations.append("Contact a Green Pulse expert for a more detailed analysis and customized solutions.")

        return recommendations


# Initialize the system
system = SmartEnergySystem()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    try:
        energy_type = data['energyType']
        equipment_type = data['equipmentType']
        lifetime = float(data['lifetime'])

        input_data = {
            'Type_Energy': energy_type,
            'Category_Equipment': equipment_type,
            'Estimated_Lifetime_Years': lifetime
        }

        results = system.predict_energy(input_data)
        recommendations = system.generate_recommendations(input_data, results)

        return jsonify({
            'status': 'success',
            'results': {
                'consumption': round(results['kwh'], 2),
                'emission': round(results['co2'], 2),
                'cost': round(results['cost'], 2)
            },
            'recommendations': recommendations
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400


@app.route('/generate_pdf', methods=['POST'])
def generate_pdf():
    data = request.json
    try:
        energy_type = data['energyType']
        equipment_type = data['equipmentType']
        lifetime = float(data['lifetime'])

        input_data = {
            'Type_Energy': energy_type,
            'Category_Equipment': equipment_type,
            'Estimated_Lifetime_Years': lifetime
        }

        results = system.predict_energy(input_data)

        # In a real implementation, you would generate the PDF here
        # For now, we'll return a success message
        return jsonify({
            'status': 'success',
            'message': 'PDF generation would happen here in a full implementation'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400


######



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

Respond only with the actions, in English.

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


@app.route('/system1')
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