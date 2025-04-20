import pandas as pd
import joblib
from fpdf import FPDF
from datetime import datetime
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import warnings
import os

warnings.filterwarnings('ignore')


class EnergyAIAssistant:
    def __init__(self):
        self.colors = {
            'dark_blue': (0, 51, 102),
            'light_blue': (173, 216, 230),
            'green': (0, 128, 0),
            'red': (255, 0, 0),
            'orange': (255, 165, 0),
            'gray': (240, 240, 240)
        }

        self.energy_factors = {
            'Fioul': {'consumption': 1.2, 'emission': 0.25, 'color': (139, 69, 19)},
            'Gaz': {'consumption': 1.0, 'emission': 0.15, 'color': (255, 165, 0)},
            'Electricité': {'consumption': 0.8, 'emission': 0.10, 'color': (255, 255, 0)},
            'Solaire': {'consumption': 0.5, 'emission': 0.02, 'color': (255, 215, 0)}
        }

        self.equipment_impact = {
            'Production': {'impact': 1.0},
            'Transformation': {'impact': 0.9},
            'Refrigeration': {'impact': 0.8},
            'Conditionnement': {'impact': 0.6},
            'Nettoyage': {'impact': 0.4},
            'Controle Qualite': {'impact': 0.3}
        }

    def calculate_smart_consumption(self, energy_type, equipment_type, lifetime):
        base = 3000
        energy_factor = self.energy_factors[energy_type]['consumption']
        equipment_factor = self.equipment_impact[equipment_type]['impact']
        age_factor = 1 + (lifetime * 0.02)
        return base * energy_factor * equipment_factor * age_factor

    def calculate_smart_emissions(self, energy_type, equipment_type, consumption):
        emission_factor = self.energy_factors[energy_type]['emission']
        equipment_factor = self.equipment_impact[equipment_type]['impact']
        return consumption * emission_factor * equipment_factor


class GreenPulseDesign:
    def __init__(self):
        self.colors = {
            'primary': (0, 105, 92),
            'secondary': (139, 195, 74),
            'accent': (255, 193, 7),
            'dark': (33, 33, 33),
            'light': (245, 245, 245)
        }

        self.logo = "GREEN PULSE"
        self.slogan = "Feel the pulse of sustainability"

    def add_header(self, pdf):
        pdf.set_fill_color(*self.colors['primary'])
        pdf.rect(0, 0, 210, 25, 'F')

        pdf.set_font('Helvetica', 'B', 20)
        pdf.set_text_color(255, 255, 255)
        pdf.set_xy(10, 8)
        pdf.cell(0, 10, self.logo, 0, 0, 'L')

        pdf.set_font('Helvetica', 'I', 10)
        pdf.set_xy(10, 15)
        pdf.cell(0, 10, self.slogan, 0, 0, 'L')

        pdf.set_font('Helvetica', '', 10)
        pdf.set_xy(0, 8)
        pdf.cell(190, 10, datetime.now().strftime('%d/%m/%Y'), 0, 0, 'R')

        pdf.set_y(30)

    def add_footer(self, pdf):
        pdf.set_y(-15)
        pdf.set_font('Helvetica', 'I', 8)
        pdf.set_text_color(*self.colors['dark'])
        pdf.cell(0, 10, f"Page {pdf.page_no()}", 0, 0, 'C')


class SmartEnergySystem:
    def __init__(self):
        self.assistant = EnergyAIAssistant()
        self.design = GreenPulseDesign()
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

    def generate_smart_data(self):
        energy_types = list(self.assistant.energy_factors.keys())
        equipment_types = list(self.assistant.equipment_impact.keys())

        data = {
            'Type_Energy': np.random.choice(energy_types, 1000),
            'Category_Equipment': np.random.choice(equipment_types, 1000),
            'Estimated_Lifetime_Years': np.random.randint(1, 25, 1000)
        }

        df = pd.DataFrame(data)

        df['Energy_Consumption_kWh'] = df.apply(
            lambda row: self.assistant.calculate_smart_consumption(
                row['Type_Energy'],
                row['Category_Equipment'],
                row['Estimated_Lifetime_Years']
            ), axis=1)

        df['CO2_Emissions_kg'] = df.apply(
            lambda row: self.assistant.calculate_smart_emissions(
                row['Type_Energy'],
                row['Category_Equipment'],
                row['Energy_Consumption_kWh']
            ), axis=1)

        return df

    def predict_energy(self, data):
        input_df = pd.DataFrame([data])
        prediction = self.model.predict(input_df)

        return {
            'kwh': prediction[0][0],
            'co2': prediction[0][1],
            'cost': prediction[0][0] * 0.15
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

    def generate_pdf_report(self, data, results, buffer=None):
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=25)

        # Page 1 - Cover
        pdf.add_page()
        pdf.set_fill_color(*self.design.colors['primary'])
        pdf.rect(0, 0, 210, 297, 'F')
        pdf.set_y(80)
        pdf.set_font('Helvetica', 'B', 36)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(0, 20, "GREEN PULSE", 0, 1, 'C')
        pdf.set_font('Helvetica', 'I', 18)
        pdf.cell(0, 15, "Energy Analysis Report", 0, 1, 'C')

        # Page 2 - Contents
        pdf.add_page()
        self.design.add_header(pdf)
        pdf.set_font('Helvetica', 'B', 20)
        pdf.set_text_color(*self.design.colors['primary'])
        pdf.cell(0, 15, "Table of Contents", 0, 1)

        # [Rest of the PDF generation code...]

        if buffer:
            pdf.output(buffer)
        else:
            filename = f"GreenPulse_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            pdf.output(filename)