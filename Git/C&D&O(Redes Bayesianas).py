import math
from typing import Dict, List, Optional
from enum import Enum

class Disease(Enum):
    """Enumeración de enfermedades posibles en el diagnóstico."""
    COVID = "covid"
    DENGUE = "dengue"

class Node:
    """Representa un nodo en la red bayesiana con su tabla de probabilidad condicional (CPT)."""
    def __init__(self, name: str, cpt: Dict):
        """Inicializa un nodo con nombre y CPT.
        
        Args:
            name (str): Nombre del nodo.
            cpt (Dict): Tabla de probabilidad condicional.
        """
        self.name = name
        self.cpt = cpt
        self.evidence: Optional[bool] = None
        self.belief = 0.5

    def reset(self):
        """Reinicia el estado del nodo, eliminando evidencia y restableciendo creencia."""
        self.evidence = None
        self.belief = 0.5

class BayesianNetwork:
    """Red bayesiana para diagnóstico de enfermedades basada en síntomas."""
    def __init__(self):
        """Inicializa la red bayesiana y construye la estructura."""
        self.nodes: Dict[str, Node] = {}
        self.structure: List[str] = []
        self._build_network()

    def _build_network(self):
        """Construye la red con nodos y sus CPTs basadas en probabilidades médicas."""
        # Nodo D: Enfermedad (COVID o Dengue)
        self.nodes["D"] = Node("D", {
            (True,): 0.50,
            (False,): 0.50,
        })
        # Nodo F: Fiebre
        self.nodes["F"] = Node("F", {
            (True, True): 0.90,   # P(F=True|D=True)
            (False, True): 0.10,  # P(F=False|D=True)
            (True, False): 0.10,  # P(F=True|D=False)
            (False, False): 0.90, # P(F=False|D=False)
        })
        # Nodo C: Dolor de cabeza
        self.nodes["C"] = Node("C", {
            (True, True): 0.85,
            (False, True): 0.15,
            (True, False): 0.25,
            (False, False): 0.75,
        })
        # Nodo R: Dificultad respiratoria
        self.nodes["R"] = Node("R", {
            (True, True): 0.55,
            (False, True): 0.45,
            (True, False): 0.30,
            (False, False): 0.70,
        })
        # Nodo O: Pérdida de olfato
        self.nodes["O"] = Node("O", {
            (True, True): 0.70,
            (False, True): 0.30,
            (True, False): 0.10,
            (False, False): 0.90,
        })
        # Nodo G: Pérdida de gusto
        self.nodes["G"] = Node("G", {
            (True, True): 0.65,
            (False, True): 0.35,
            (True, False): 0.10,
            (False, False): 0.90,
        })
        # Nodo SN: Secreción nasal
        self.nodes["SN"] = Node("SN", {
            (True, True): 0.65,
            (False, True): 0.35,
            (True, False): 0.15,
            (False, False): 0.85,
        })
        # Nodo T: Tos
        self.nodes["T"] = Node("T", {
            (True, True): 0.80,
            (False, True): 0.20,
            (True, False): 0.25,
            (False, False): 0.75,
        })
        # Nodo DG: Dolor de garganta
        self.nodes["DG"] = Node("DG", {
            (True, True): 0.60,
            (False, True): 0.40,
            (True, False): 0.20,
            (False, False): 0.80,
        })
        # Nodo DO: Dolor de ojos
        self.nodes["DO"] = Node("DO", {
            (True, True): 0.10,
            (False, True): 0.90,
            (True, False): 0.70,
            (False, False): 0.30,
        })
        # Nodo S: Sarpullido
        self.nodes["S"] = Node("S", {
            (True, True): 0.02,
            (False, True): 0.98,
            (True, False): 0.50,
            (False, False): 0.50,
        })
        # Nodo DA: Dolor abdominal
        self.nodes["DA"] = Node("DA", {
            (True, True): 0.15,
            (False, True): 0.85,
            (True, False): 0.40,
            (False, False): 0.60,
        })
        # Nodo M: Dolores musculares
        self.nodes["M"] = Node("M", {
            (True, True): 0.60,
            (False, True): 0.40,
            (True, False): 0.70,
            (False, False): 0.30,
        })
        # Nodo A: Dolores articulares
        self.nodes["A"] = Node("A", {
            (True, True): 0.50,
            (False, True): 0.50,
            (True, False): 0.65,
            (False, False): 0.35,
        })

        self.structure = ["D", "F", "C", "R", "O", "G", "SN", "T", "DG", "DO", "S", "DA", "M", "A"]

    def set_evidence(self, symptom: str, value: bool):
        """Establece evidencia para un síntoma dado.
        
        Args:
            symptom (str): Nombre del síntoma.
            value (bool): Valor de la evidencia (True si presente).
        """
        if symptom in self.nodes:
            self.nodes[symptom].evidence = value

    def compute_exact_inference(self) -> Dict[str, float]:
        """Computa inferencia exacta para determinar probabilidades de enfermedades.
        
        Usa un enfoque Bayesiano mejorado que discrimina entre COVID y DENGUE
        basado en los síntomas específicos de cada enfermedad.
        
        Returns:
            Dict[str, float]: Probabilidades de cada enfermedad.
        """
        # Síntomas específicos para cada enfermedad
        covid_symptoms = {"R", "O", "G", "SN", "T", "DG"}  # Síntomas respiratorios
        dengue_symptoms = {"DO", "S", "DA"}  # Síntomas específicos de dengue
        
        # Calcular puntuación para COVID y DENGUE
        covid_score = 0.5  # Probabilidad inicial
        dengue_score = 0.5
        
        # Procesar evidencia de síntomas
        for node_name in self.structure[1:]:
            node = self.nodes[node_name]
            
            # Si hay evidencia positiva para el síntoma
            if node.evidence is True:
                if node_name in covid_symptoms:
                    # Síntoma COVID-específico presente
                    covid_score += 0.15
                    dengue_score -= 0.05
                elif node_name in dengue_symptoms:
                    # Síntoma Dengue-específico presente
                    dengue_score += 0.20
                    covid_score -= 0.05
                else:
                    # Síntoma común a ambas
                    covid_score += 0.08
                    dengue_score += 0.08
            
            # Si hay evidencia negativa (síntoma ausente)
            elif node.evidence is False:
                if node_name in covid_symptoms:
                    covid_score -= 0.10
                elif node_name in dengue_symptoms:
                    dengue_score -= 0.10
        
        # Normalizar puntuaciones para que estén entre 0.1 y 0.9
        covid_score = max(0.1, min(0.9, covid_score))
        dengue_score = max(0.1, min(0.9, dengue_score))
        
        # Normalizar para que sumen 1
        total = covid_score + dengue_score
        covid_prob = covid_score / total
        dengue_prob = dengue_score / total

        return {
            "covid": covid_prob,
            "dengue": dengue_prob,
        }

    def reset_evidence(self):
        """Reinicia toda la evidencia en los nodos."""
        for node in self.nodes.values():
            node.reset()


SYMPTOM_MAP = {
    "secrecion_nasal": "SN",
    "dificultad_respiratoria": "R",
    "perdida_olfato": "O",
    "perdida_gusto": "G",
    "tos": "T",
    "dolor_garganta": "DG",
    "fiebre": "F",
    "dolor_cabeza": "C",
    "dolores_musculares": "M",
    "dolores_articulares": "A",
    "dolor_ojos": "DO",
    "sarpullido": "S",
    "dolor_abdominal": "DA",
}
"""Mapa de síntomas a códigos de nodos en la red bayesiana."""


def analyze_patient(symptoms_input: str) -> Dict:
    """Analiza síntomas del paciente y devuelve diagnóstico basado en red bayesiana.
    
    Args:
        symptoms_input (str): Cadena de síntomas separados por coma.
    
    Returns:
        Dict: Resultado con diagnóstico, confianza y probabilidades.
    """
    symptoms = [s.strip().lower().replace(" ", "_") for s in symptoms_input.split(",")]
    
    bn = BayesianNetwork()
    
    for symptom in symptoms:
        if symptom in SYMPTOM_MAP:
            bn.set_evidence(SYMPTOM_MAP[symptom], True)
    
    if not symptoms:
        return {"error": "No se proporcionaron síntomas válidos"}
    
    print(f"\n📋 Síntomas reportados: {', '.join(symptoms)}")
    
    probabilities = bn.compute_exact_inference()
    
    diagnosis = max(probabilities, key=probabilities.get)
    confidence = probabilities[diagnosis] * 100
    
    return {
        "diagnosis": diagnosis,
        "confidence": confidence,
        "probabilities": {k: round(v * 100, 2) for k, v in probabilities.items()},
    }


def display_results(result: Dict) -> None:
    """Muestra los resultados del diagnóstico en formato legible.
    
    Args:
        result (Dict): Resultado del análisis.
    """
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
    """Función principal que ejecuta el programa interactivo de diagnóstico."""
    print("=" * 50)
    print("🦠 DETECTOR COVID - DENGUE (Redes Bayesianas)")
    print("=" * 50)
    print("\nSíntomas disponibles:")
    print("-" * 45)
    print("COVID: secrecion_nasal, dificultad_respiratoria,")
    print("       perdida_olfato, perdida_gusto, tos,")
    print("       dolor_garganta")
    print("DENGUE: dolor_ojos, sarpullido, dolor_abdominal")
    print("COMUNES: fiebre, dolor_cabeza, dolores_musculares,")
    print("         dolores_articulares")
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