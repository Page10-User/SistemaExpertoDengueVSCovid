import time
import sys
import os
import importlib.util

# Función para cargar módulos desde archivos con nombres especiales
def load_module_from_file(filepath, module_name):
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Cargar los módulos
nb_module = load_module_from_file("C&D&O(Naive Bayes).py", "nb")
rf_module = load_module_from_file("C&D&O(Random Forest).py", "rf")
bn_module = load_module_from_file("C&D&O(Redes Bayesianas).py", "bn")

# Funciones de análisis
nb_analyze = nb_module.analyze_patient
rf_analyze = rf_module.analyze_patient
bn_analyze = bn_module.analyze_patient

# Modelo Random Forest
rf_model = rf_module.RandomForestDiagnostic()

# Función para medir tiempo de ejecución
def measure_execution_time(func, *args, **kwargs):
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    execution_time = end_time - start_time
    return result, execution_time

# Casos de prueba con síntomas
test_cases = [
    "fiebre, tos, dificultad_respiratoria",  # Caso típico COVID
    "fiebre, dolor_cabeza, dolor_ojos, sarpullido",  # Caso típico Dengue
    "fiebre, dolor_cabeza, fatiga",  # Caso común
    "secrecion_nasal, perdida_olfato, perdida_gusto",  # COVID específico
    "dolor_ojos, sarpullido, dolor_abdominal",  # Dengue específico
]

# Ventajas y desventajas de cada enfoque
advantages_disadvantages = {
    "Naive Bayes": {
        "ventajas": [
            "Simple y rápido de implementar",
            "Requiere pocos datos para entrenar",
            "Fácil de interpretar",
            "Eficiente computacionalmente"
        ],
        "desventajas": [
            "Asume independencia entre características (síntomas), lo cual puede no ser realista",
            "Sensible a características irrelevantes",
            "No maneja bien relaciones complejas entre variables"
        ]
    },
    "Random Forest": {
        "ventajas": [
            "Maneja bien relaciones no lineales y complejas",
            "Robusto contra overfitting con buen ajuste de parámetros",
            "Puede manejar características irrelevantes",
            "Proporciona importancia de características"
        ],
        "desventajas": [
            "Más complejo de interpretar que modelos simples",
            "Requiere más datos para entrenar bien",
            "Puede ser computacionalmente intensivo",
            "Necesita ajuste de hiperparámetros"
        ]
    },
    "Redes Bayesianas": {
        "ventajas": [
            "Modela explícitamente relaciones causales entre variables",
            "Permite razonamiento probabilístico natural",
            "Puede manejar incertidumbre de manera elegante",
            "Fácil de actualizar con nueva evidencia"
        ],
        "desventajas": [
            "Difícil de construir y requiere conocimiento experto",
            "Computacionalmente costoso para redes grandes",
            "Sensible a la calidad de las probabilidades prior",
            "Complejo de mantener y actualizar"
        ]
    }
}

def run_comparisons():
    print("=" * 60)
    print("COMPARACIÓN DE ENFOQUES PROBABILÍSTICOS")
    print("Naive Bayes, Random Forest y Redes Bayesianas")
    print("=" * 60)

    # Modelo Random Forest ya entrenado

    results_summary = {method: {"times": [], "diagnoses": [], "confidences": []} for method in ["Naive Bayes", "Random Forest", "Redes Bayesianas"]}

    for i, symptoms in enumerate(test_cases, 1):
        print(f"\n--- Caso de Prueba {i}: {symptoms} ---")

        # Naive Bayes
        result_nb, time_nb = measure_execution_time(nb_analyze, symptoms)
        results_summary["Naive Bayes"]["times"].append(time_nb)
        if "error" not in result_nb:
            results_summary["Naive Bayes"]["diagnoses"].append(result_nb["diagnosis"])
            results_summary["Naive Bayes"]["confidences"].append(result_nb["confidence"])
            print(f"Naive Bayes: {result_nb['diagnosis']} ({result_nb['confidence']:.1f}%) - Tiempo: {time_nb:.4f}s")
        else:
            print(f"Naive Bayes: Error - {result_nb['error']}")

        # Random Forest
        result_rf, time_rf = measure_execution_time(rf_analyze, symptoms, rf_model)
        results_summary["Random Forest"]["times"].append(time_rf)
        if "error" not in result_rf:
            results_summary["Random Forest"]["diagnoses"].append(result_rf["diagnosis"])
            results_summary["Random Forest"]["confidences"].append(result_rf["confidence"])
            print(f"Random Forest: {result_rf['diagnosis']} ({result_rf['confidence']:.1f}%) - Tiempo: {time_rf:.4f}s")
        else:
            print(f"Random Forest: Error - {result_rf['error']}")

        # Redes Bayesianas
        result_bn, time_bn = measure_execution_time(bn_analyze, symptoms)
        results_summary["Redes Bayesianas"]["times"].append(time_bn)
        if "error" not in result_bn:
            results_summary["Redes Bayesianas"]["diagnoses"].append(result_bn["diagnosis"])
            results_summary["Redes Bayesianas"]["confidences"].append(result_bn["confidence"])
            print(f"Redes Bayesianas: {result_bn['diagnosis']} ({result_bn['confidence']:.1f}%) - Tiempo: {time_bn:.4f}s")
        else:
            print(f"Redes Bayesianas: Error - {result_bn['error']}")

    # Resumen de rendimiento
    print("\n" + "=" * 60)
    print("RESUMEN DE RENDIMIENTO")
    print("=" * 60)

    for method in results_summary:
        times = results_summary[method]["times"]
        if times:
            avg_time = sum(times) / len(times)
            diagnoses = results_summary[method]["diagnoses"]
            confidences = results_summary[method]["confidences"]
            if confidences:
                avg_confidence = sum(confidences) / len(confidences)
                print(f"\n{method}:")
                print(f"  Tiempo promedio de ejecución: {avg_time:.4f}s")
                print(f"  Confianza promedio: {avg_confidence:.1f}%")
                print(f"  Diagnósticos: {diagnoses}")
            else:
                print(f"\n{method}: Errores en todos los casos")

    # Ventajas y desventajas
    print("\n" + "=" * 60)
    print("VENTAJAS Y DESVENTAJAS")
    print("=" * 60)

    for method, info in advantages_disadvantages.items():
        print(f"\n{method}:")
        print("  Ventajas:")
        for adv in info["ventajas"]:
            print(f"    - {adv}")
        print("  Desventajas:")
        for dis in info["desventajas"]:
            print(f"    - {dis}")

    print("\n" + "=" * 60)
    print("CONCLUSIÓN")
    print("=" * 60)
    print("Cada enfoque tiene sus fortalezas:")
    print("- Naive Bayes: Ideal para problemas simples con independencia asumida.")
    print("- Random Forest: Bueno para datos complejos y relaciones no lineales.")
    print("- Redes Bayesianas: Excelente para modelar conocimiento experto y relaciones causales.")
    print("La elección depende del problema específico, disponibilidad de datos y expertise.")

if __name__ == "__main__":
    run_comparisons()