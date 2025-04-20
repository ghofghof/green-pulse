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


if __name__ == '__main__':
    app.run(debug=True)