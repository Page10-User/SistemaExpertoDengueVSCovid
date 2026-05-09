import math
from typing import Dict, List

# Diccionario de síntomas con probabilidades condicionadas para cada enfermedad.
# "combined" indica si el síntoma se considera común a más de una enfermedad.
SYMPTOMS = {
    "fiebre": {"covid": 0.85, "dengue": 0.90, "combined": True},
    "dolor_cabeza": {"covid": 0.70, "dengue": 0.75, "combined": True},
    "dolores_musculares": {"covid": 0.60, "dengue": 0.70, "combined": True},
    "dolores_articulares": {"covid": 0.50, "dengue": 0.65, "combined": True},
    "fatiga": {"covid": 0.75, "dengue": 0.80, "combined": True},
    "malestar_general": {"covid": 0.80, "dengue": 0.85, "combined": True},
    "nauseas": {"covid": 0.30, "dengue": 0.40, "combined": True},
    "vomitos": {"covid": 0.20, "dengue": 0.35, "combined": True},
    "diarrea": {"covid": 0.25, "dengue": 0.30, "combined": True},
    "secrecion_nasal": {"covid": 0.65, "dengue": 0.10, "combined": False},
    "dificultad_respiratoria": {"covid": 0.55, "dengue": 0.05, "combined": False},
    "perdida_olfato": {"covid": 0.70, "dengue": 0.02, "combined": False},
    "perdida_gusto": {"covid": 0.65, "dengue": 0.02, "combined": False},
    "tos": {"covid": 0.80, "dengue": 0.15, "combined": False},
    "dolor_garganta": {"covid": 0.60, "dengue": 0.10, "combined": False},
    "dolor_ojos": {"covid": 0.10, "dengue": 0.70, "combined": False},
    "sarpullido": {"covid": 0.02, "dengue": 0.50, "combined": False},
    "dolor_abdominal": {"covid": 0.15, "dengue": 0.40, "combined": False},
}

# Probabilidades previas para cada diagnóstico antes de observar los síntomas.
PRIOR_PROBS = {"covid": 0.50, "dengue": 0.50}

# Parámetro de suavizado para evitar logaritmos de cero.
LAPLACE_ALPHA = 1.0


def calculate_conditional_prob(symptom: str, disease: str) -> float:
    """Devuelve P(síntoma | enfermedad) usando la tabla de síntomas.

    Para síntomas desconocidos, se devuelve una probabilidad neutra de 0.5.
    """
    if symptom not in SYMPTOMS:
        # Síntoma desconocido -> probabilidad neutra.
        return 0.5

    prob = SYMPTOMS[symptom].get(disease, 0.5)

    if SYMPTOMS[symptom]["combined"]:
        # Para síntomas compartidos, se mantiene el valor original.
        combined_weight = 0.5
        return prob * (1 - combined_weight) + prob * combined_weight

    return prob


def naive_bayes(patient_symptoms: List[str]) -> Dict[str, float]:
    """Calcula la probabilidad posterior para cada diagnóstico.
    
    IMPORTANTE: Solo considera síntomas PRESENTES (reportados explícitamente).
    Los síntomas ausentes NO se usan como evidencia, ya que en diagnóstico médico
    real, solo los síntomas que el paciente reporta son información relevante.
    Ignorar síntomas ausentes evita el sesgo de considerar la ausencia de síntomas
    específicos como evidencia fuerte para otra enfermedad.
    """
    log_posteriors = {}

    for disease in ["covid", "dengue"]:
        # Iniciar con la probabilidad previa en escala logarítmica.
        log_prob = math.log(PRIOR_PROBS[disease])

        # SOLO iterar sobre síntomas PRESENTES (reportados por el paciente)
        for symptom in patient_symptoms:
            if symptom in SYMPTOMS:
                p_present = calculate_conditional_prob(symptom, disease)
                # Sumar la probabilidad del síntoma presente
                log_prob += math.log(p_present + LAPLACE_ALPHA)

        log_posteriors[disease] = log_prob

    # Convertir los logaritmos a probabilidades normalizadas.
    max_log = max(log_posteriors.values())
    exp_logs = {d: math.exp(l - max_log) for d, l in log_posteriors.items()}
    total = sum(exp_logs.values())
    posteriors = {d: e / total for d, e in exp_logs.items()}

    return posteriors


def analyze_patient(symptoms_input: str) -> Dict:
    """Procesa los síntomas ingresados y devuelve el diagnóstico generado."""
    symptoms = [s.strip().lower().replace(" ", "_") for s in symptoms_input.split(",")]
    symptoms = [s for s in symptoms if s]

    valid_symptoms = [s for s in symptoms if s in SYMPTOMS]
    invalid = [s for s in symptoms if s not in SYMPTOMS]

    if invalid:
        # Informar síntomas que no están en la lista de referencia.
        print(f"⚠️ Síntomas no reconocidos: {', '.join(invalid)}")

    if not valid_symptoms:
        # Si no hay síntomas válidos, no se puede computar un diagnóstico.
        return {"error": "No se proporcionaron síntomas válidos"}

    print(f"\n📋 Síntomas reportados: {', '.join(valid_symptoms)}")

    probabilities = naive_bayes(valid_symptoms)

    diagnosis = max(probabilities, key=probabilities.get)
    confidence = probabilities[diagnosis] * 100

    return {
        "diagnosis": diagnosis,
        "confidence": confidence,
        "probabilities": {k: round(v * 100, 2) for k, v in probabilities.items()},
        "symptoms_used": valid_symptoms,
    }


def display_results(result: Dict) -> None:
    """Imprime el resultado y la confianza del diagnóstico en un formato legible."""
    if "error" in result:
        print(f"\n❌ Error: {result['error']}")
        return

    print("\n" + "=" * 45)
    print("🩺 RESULTADO DEL DIAGNÓSTICO")
    print("=" * 45)
    print(f"\nEnfermedad más probable: {result['diagnosis'].upper()}")
    print(f"Confianza: {result['confidence']:.1f}%")
    print("\n📊 Probabilidades:")
    for disease, prob in result["probabilities"].items():
        bar = "█" * int(prob / 2)
        print(f"  {disease:12} | {bar} {prob:.1f}%")


def main():
    """Bucle principal que solicita síntomas y muestra los resultados."""
    print("=" * 50)
    print("🦠 DETECTOR COVID - DENGUE (Naive Bayes)")
    print("=" * 50)
    print("\nSíntomas disponibles:")
    print("-" * 45)
    print("COVID: secrecion_nasal, dificultad_respiratoria,")
    print("       perdida_olfato, perdida_gusto, tos,")
    print("       dolor_garganta")
    print("DENGUE: dolor_ojos, sarpullido, dolor_abdominal")
    print("COMUNES: fiebre, dolor_cabeza, dolores_musculares,")
    print("         dolores_articulares, fatiga, malestar_general,")
    print("         nauseas, vomitos, diarrea")
    print("-" * 45)

    while True:
        print("\nIngrese sus síntomas (separados por coma)")
        print("O presione Enter para salir")

        user_input = input("\n> ").strip()

        if not user_input:
            print("\n👋 Programa finalizado.")
            break

        result = analyze_patient(user_input)
        display_results(result)


if __name__ == "__main__":
    main()