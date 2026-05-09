import random
from typing import Dict, List
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Síntomas disponibles para el diagnóstico
SYMPTOMS = [
    "fiebre", "dolor_cabeza", "dolores_musculares", "dolores_articulares",
    "fatiga", "malestar_general", "nauseas", "vomitos", "diarrea",
    "secrecion_nasal", "dificultad_respiratoria", "perdida_olfato",
    "perdida_gusto", "tos", "dolor_garganta", "dolor_ojos", "sarpullido",
    "dolor_abdominal"
]

# Probabilidades para generar datos de entrenamiento
SYMPTOM_PROBS = {
    "fiebre": {"covid": 0.85, "dengue": 0.90},
    "dolor_cabeza": {"covid": 0.70, "dengue": 0.75},
    "dolores_musculares": {"covid": 0.60, "dengue": 0.70},
    "dolores_articulares": {"covid": 0.50, "dengue": 0.65},
    "fatiga": {"covid": 0.75, "dengue": 0.80},
    "malestar_general": {"covid": 0.80, "dengue": 0.85},
    "nauseas": {"covid": 0.30, "dengue": 0.40},
    "vomitos": {"covid": 0.20, "dengue": 0.35},
    "diarrea": {"covid": 0.25, "dengue": 0.30},
    "secrecion_nasal": {"covid": 0.65, "dengue": 0.10},
    "dificultad_respiratoria": {"covid": 0.55, "dengue": 0.05},
    "perdida_olfato": {"covid": 0.70, "dengue": 0.02},
    "perdida_gusto": {"covid": 0.65, "dengue": 0.02},
    "tos": {"covid": 0.80, "dengue": 0.15},
    "dolor_garganta": {"covid": 0.60, "dengue": 0.10},
    "dolor_ojos": {"covid": 0.10, "dengue": 0.70},
    "sarpullido": {"covid": 0.02, "dengue": 0.50},
    "dolor_abdominal": {"covid": 0.15, "dengue": 0.40},
}


def generate_training_data(n_samples: int = 500) -> tuple:
    """Genera datos de entrenamiento sintéticos basados en probabilidades.
    
    Args:
        n_samples: Número de muestras a generar.
    
    Returns:
        Tuple con (X, y) datos de entrenamiento.
    """
    diseases = ["covid", "dengue"]
    X = []
    y = []

    for _ in range(n_samples):
        disease = random.choice(diseases)
        symptom_vector = []

        for symptom in SYMPTOMS:
            prob = SYMPTOM_PROBS[symptom].get(disease, 0.5)
            present = random.random() < prob
            symptom_vector.append(1 if present else 0)

        X.append(symptom_vector)
        y.append(disease)

    return np.array(X), np.array(y)


class RandomForestDiagnostic:
    """Clasificador Random Forest para diagnóstico de enfermedades."""
    
    def __init__(self):
        """Inicializa y entrena el modelo Random Forest."""
        self.symptoms = SYMPTOMS
        self.diseases = ["covid", "dengue"]
        
        X, y = generate_training_data(n_samples=1000)
        
        self.classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight="balanced"
        )
        self.classifier.fit(X, y)
        
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.diseases)
    
    def predict(self, symptoms: List[str]) -> Dict:
        """Predice la enfermedad basada en los síntomas dados.
        
        Args:
            symptoms: Lista de síntomas presentes.
        
        Returns:
            Dict con diagnóstico, confianza y probabilidades.
        """
        symptom_vector = []
        for symptom in self.symptoms:
            symptom_vector.append(1 if symptom in symptoms else 0)
        
        symptom_array = np.array([symptom_vector])
        
        probabilities = self.classifier.predict_proba(symptom_array)[0]
        prediction = self.classifier.predict(symptom_array)[0]
        
        prob_dict = {}
        for i, disease in enumerate(self.classifier.classes_):
            prob_dict[disease] = probabilities[i]
        
        confidence = max(probabilities) * 100
        
        return {
            "diagnosis": prediction,
            "confidence": confidence,
            "probabilities": {k: round(v * 100, 2) for k, v in prob_dict.items()}
        }
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Retorna la importancia de cada síntoma en el modelo.
        
        Returns:
            Dict con síntoma y su importancia.
        """
        importances = self.classifier.feature_importances_
        return {
            symptom: round(importance, 4) 
            for symptom, importance in zip(self.symptoms, importances)
        }


def analyze_patient(symptoms_input: str, model: RandomForestDiagnostic) -> Dict:
    """Procesa los síntomas ingresados y devuelve el diagnóstico.
    
    Args:
        symptoms_input: Cadena de síntomas separados por coma.
        model: Modelo Random Forest entrenado.
    
    Returns:
        Dict con resultado del diagnóstico.
    """
    symptoms = [s.strip().lower().replace(" ", "_") for s in symptoms_input.split(",")]
    symptoms = [s for s in symptoms if s]

    valid_symptoms = [s for s in symptoms if s in SYMPTOMS]
    invalid = [s for s in symptoms if s not in SYMPTOMS]

    if invalid:
        print(f"ADVERTENCIA: Sintomas no reconocidos: {', '.join(invalid)}")

    if not valid_symptoms:
        return {"error": "No se proporcionaron sintomas validos"}

    print(f"\n-> Sintomas reportados: {', '.join(valid_symptoms)}")

    result = model.predict(valid_symptoms)
    result["symptoms_used"] = valid_symptoms

    return result


def display_results(result: Dict) -> None:
    """Imprime el resultado del diagnóstico en formato legible."""
    if "error" in result:
        print(f"\nERROR: {result['error']}")
        return

    print("\n" + "=" * 45)
    print(">>> RESULTADO DEL DIAGNOSTICO")
    print("=" * 45)
    print(f"\nEnfermedad mas probable: {result['diagnosis'].upper()}")
    print(f"Confianza: {result['confidence']:.1f}%")
    print("\nProbabilidades:")
    for disease, prob in result["probabilities"].items():
        bar = "#" * int(prob / 2)
        print(f"  {disease:12} | {bar} {prob:.1f}%")


def main():
    """Función principal que ejecuta el programa interactivo."""
    print("=" * 50)
    print("* DETECTOR COVID - DENGUE (Random Forest)")
    print("=" * 50)
    print("\nSintomas disponibles:")
    print("-" * 45)
    print("COVID: secrecion_nasal, dificultad_respiratoria,")
    print("       perdida_olfato, perdida_gusto, tos,")
    print("       dolor_garganta")
    print("DENGUE: dolor_ojos, sarpullido, dolor_abdominal")
    print("COMUNES: fiebre, dolor_cabeza, dolores_musculares,")
    print("         dolores_articulares, fatiga, malestar_general,")
    print("         nauseas, vomitos, diarrea")
    print("-" * 45)
    
    print("\n>>> Entrenando modelo Random Forest...")
    model = RandomForestDiagnostic()
    print(">>> Modelo entrenado correctamente\n")
    
    while True:
        print("\nIngrese sus sintomas (separados por coma)")
        print("O presione Enter para salir")

        user_input = input("\n> ").strip()

        if not user_input:
            print("\n--- Programa finalizado.")
            break

        result = analyze_patient(user_input, model)
        display_results(result)


if __name__ == "__main__":
    main()