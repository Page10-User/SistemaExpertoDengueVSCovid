# Sistema Experto: COVID vs Dengue 

Sistema de diagnóstico médico probabilístico capaz de distinguir entre COVID-19 y Dengue basándose en los síntomas del paciente.

##  Descripción

Este proyecto implementa tres enfoques diferentes de clasificación probabilística para el diagnóstico diferencial entre COVID-19 y Dengue:

| Enfoque | Archivo | Descripción |
|---------|---------|-------------|
| **Naive Bayes** | `C&D&O(Naive Bayes).py` | Clasificador probabilístico clásico basado en el teorema de Bayes |
| **Redes Bayesianas** | `C&O(Redes Bayesianas).py` | Modelo gráfico probabilístico con estructura causal |
| **Random Forest** | `C&D&O(Random Forest).py` | Ensemble de árboles de decisión |

##  Características

- **18 síntomas** analizables
- **Diagnóstico probabilístico** con porcentaje de confianza
- **Comparador visual** para ver las diferencias entre métodos (`comparaciones.py`)
- Interfaz interactiva/simple para probar casos

##  Uso

### Naive Bayes
```bash
python "C&D&O(Naive Bayes).py"
```
Ingresa los síntomas separados por coma cuando se te indique.

### Redes Bayesianas
```bash
python "C&D&O(Redes Bayesianas).py"
```

### Random Forest
```bash
python "C&D&O(Random Forest).py"
```

### Comparador (recomendado para ver diferencias)
```bash
python comparaciones.py
```
Muestra una comparación visual de los tres métodos con casos de prueba.

##  Síntomas Disponibles

**Síntomas COVID específicos:**
- secrecion_nasal, dificultad_respiratoria, perdida_olfato, perdida_gusto, tos, dolor_garganta

**Síntomas Dengue específicos:**
- dolor_ojos, sarpullido, dolor_abdominal

**Síntomas comunes (ambas enfermedades):**
- fiebre, dolor_cabeza, dolores_musculares, dolores_articulares, fatiga, malestar_general, nauseas, vomitos, diarrea

##  Comparación de Métodos

### Naive Bayes
- ✅ Rápido y simple
- ✅ Alta interpretabilidad
- ✅ Funciona con pocos datos
- ❌ Supuesto de independencia entre síntomas

### Redes Bayesianas
- ✅ Estructura causal visible
- ✅ Maneja dependencias entre síntomas
- ❌ Complejo de construir
- ❌ Inferencia costosa con muchas variables

### Random Forest
- ✅ Alta precisión
- ✅ Maneja relaciones no lineales
- ❌ "Caja negra" (difícil de interpretar)
- ❌ Requiere más datos

##  Estructura del Proyecto

```
Git/
├── C&D&O(Naive Bayes).py       # Clasificador Naive Bayes
├── C&D&O(Redes Bayesianas).py  # Redes Bayesianas
├── C&D&O(Random Forest).py     # Random Forest
├── comparaciones.py            # Comparador visual
├── README.md                   # Este archivo
└── LICENCIA                    # Licencia MIT
```

**Autores:** [Acosta Lopez Gonzalo Nahuel & Cesar pietro barrios calathaki]
**Versión:** 1.0.0
**Fecha:** 2026
