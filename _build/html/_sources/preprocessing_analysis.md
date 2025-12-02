# Pipeline de Preprocesamiento y Data Augmentation: Análisis Técnico

## 1. Arquitectura del Pipeline

El sistema de preparación de datos implementa un pipeline multietapa con tres niveles progresivos de limpieza textual y un módulo de data augmentation basado en back-translation. La arquitectura está diseñada para maximizar la preservación semántica mientras normaliza el texto para consumo por diferentes arquitecturas de modelos.

### 1.1 Estructura General

```
Dataset Original (2,484 muestras)
    ↓
[Limpieza Básica] → text_basic
    ↓
[Limpieza Media] → text_medium
    ↓
[Limpieza Avanzada] → text_advanced
    ↓
[Data Augmentation] → Dataset Expandido (2,621 muestras)
    ↓
[Stratified Group Split] → Train (2,104) | Val (262) | Test (255)
```

---

## 2. Niveles de Limpieza Textual

### 2.1 Limpieza Básica (text_basic)

**Objetivo**: Normalización sintáctica sin pérdida de información semántica.

**Operaciones**:
1. **Decodificación HTML**: `html.unescape()` para convertir entidades HTML (&amp; → &)
2. **Eliminación de tags HTML**: Regex `r'<[^>]+>'` para remover markup residual
3. **Eliminación de URLs**: Regex `r'http\S+|www\S+` para limpiar enlaces
4. **Normalización de guiones Unicode**: Conversión de variantes tipográficas (–, —, －) a `-` estándar
5. **Eliminación de caracteres de control**: Rango `[\x00-\x1F\x7F-\x9F]` para remover no imprimibles
6. **Normalización de espacios**: Colapso de múltiples espacios a uno solo

**Justificación**: Esta capa preserva el texto en su forma más cercana a la original, permitiendo que modelos basados en subword tokenization (DistilBERT) o character n-grams (FastText) aprovechen patrones morfológicos completos.

**Ejemplo**:
```
Original: "Managed team  of 5 engineers&#44; developing web&#45;based solutions."
text_basic: "Managed team of 5 engineers, developing web-based solutions."
```

### 2.2 Limpieza Media (text_medium)

**Objetivo**: Normalización léxica para modelos basados en word embeddings.

**Operaciones adicionales**:
1. **Conversión a minúsculas**: Elimina variabilidad por capitalización
2. **Eliminación de puntuación**: Regex `r'[^\w\s]'` remueve todos los caracteres no alfanuméricos
3. **Re-normalización de espacios**: Asegura separación consistente entre tokens

**Justificación**: Los embeddings preentrenados (Word2Vec, GloVe) típicamente son case-insensitive y no incluyen puntuación en su vocabulario. Esta normalización maximiza la tasa de cobertura (hit rate) en el vocabulario de embeddings.

**Ejemplo**:
```
text_basic: "Managed team of 5 engineers, developing web-based solutions."
text_medium: "managed team of 5 engineers developing web based solutions"
```

**Trade-off**: Se pierde información de énfasis (capitalización) y estructura sintáctica (puntuación), pero se gana en consistencia de representación.

### 2.3 Limpieza Avanzada (text_advanced)

**Objetivo**: Reducción dimensional del espacio léxico mediante lematización y eliminación de stopwords.

**Operaciones adicionales**:
1. **Eliminación de stopwords**: Filtrado usando lista NLTK de 179 stopwords en inglés
2. **Lematización de verbos**: `WordNetLemmatizer` con `pos='v'` (running → run, managed → manage)
3. **Lematización de sustantivos**: `WordNetLemmatizer` con `pos='n'` (engineers → engineer)

**Justificación**: Para modelos como XGBoost con representaciones TF-IDF de dimensionalidad fija (10,000 features), reducir la dispersión léxica permite que el modelo se enfoque en raíces semánticas en lugar de variantes morfológicas.

**Ejemplo**:
```
text_medium: "managed team of 5 engineers developing web based solutions"
text_advanced: "manage team 5 engineer develop web base solution"
```

**Consideraciones**:
- **Pérdida de matices**: "managing" vs "managed" colapsan a "manage", perdiendo información temporal
- **Beneficio en generalización**: Reduce vocabulario de 32,604 a ~18,000 palabras únicas, disminuyendo sparsity en TF-IDF
- **No aplicable a modelos contextuales**: DistilBERT usa text_advanced pero podría beneficiarse más de text_basic debido a su capacidad para capturar morfología

---

## 3. Data Augmentation mediante Back-Translation

### 3.1 Fundamento Teórico

El back-translation es una técnica de augmentation semántica que traduce texto a un idioma intermedio y luego de vuelta al original. El proceso introduce paráfrasis naturales mientras preserva el significado fundamental.

**Modelo teórico**:
```
T_en→es: Text_original → Text_spanish
T_es→en: Text_spanish → Text_augmented
```

Donde `T_en→es` y `T_es→en` son modelos de traducción neuronal MarianMT (Helsinki-NLP opus-mt).

### 3.2 Arquitectura del Sistema de Augmentation

**Componentes**:
1. **Modelos de traducción**: MarianMT preentrenados (en↔es)
2. **Modelo de similitud semántica**: SentenceTransformer 'all-MiniLM-L6-v2' para validación
3. **Sistema de chunking dinámico**: Manejo de textos largos mediante división inteligente
4. **Módulo de validación**: Filtros múltiples para asegurar calidad

### 3.3 Chunking Dinámico

**Problema**: Los modelos MarianMT tienen límite de 512 tokens. Textos largos (currículums completos) exceden este límite.

**Solución**: División en chunks con overlap.

**Parámetros**:
- `MAX_TOKENS_PER_CHUNK = 450` (margen de seguridad de 62 tokens)
- `CHUNK_OVERLAP = 50` palabras (preserva contexto en bordes)

**Algoritmo**:
```python
words_per_chunk = int(450 / 1.33)  # Estimación heurística token-to-word ratio
chunks = []
i = 0
while i < len(words):
    chunk = words[i : i + words_per_chunk]
    chunks.append(' '.join(chunk))
    i += words_per_chunk - overlap
```

**Ejemplo**:
- Texto original: 800 palabras (~1,064 tokens estimados)
- División: 3 chunks de ~267 palabras cada uno
- Overlap: 50 palabras entre chunks consecutivos

**Post-procesamiento de chunks**:
```python
merged = ' '.join(translated_chunks)
# Eliminación de duplicaciones por overlap
merged = re.sub(r'\b(\w+)(\s+\1\b)+', r'\1', merged)
```

### 3.4 Validación de Textos Augmentados

El sistema implementa un pipeline de validación multi-criterio para asegurar que los textos augmentados sean semánticamente válidos:

#### Criterio 1: Longitud Mínima
```python
MIN_TEXT_LENGTH = 50  # caracteres
```
**Justificación**: Textos muy cortos (e.g., "software engineer") no aportan contexto suficiente para clasificación.

#### Criterio 2: Ratio de Longitud
```python
MIN_LENGTH_RATIO = 0.60
MAX_LENGTH_RATIO = 1.50
```
**Justificación**: Si el texto augmentado es <60% o >150% del original, la traducción probablemente es defectuosa (pérdida de información o expansión artificial).

**Estadísticas observadas**:
- Mean length ratio (textos válidos): 1.03
- Textos rechazados por ratio: ~8%

#### Criterio 3: Similitud Semántica
```python
MIN_SEMANTIC_SIMILARITY = 0.70
```

**Método**: Cosine similarity entre embeddings de SentenceTransformer.

**Cálculo**:
```python
embeddings = model.encode([original, augmented])
similarity = cosine_similarity(embeddings[0], embeddings[1])
```

**Interpretación**:
- similarity ≥ 0.75: Preservación semántica excelente
- 0.70 ≤ similarity < 0.75: Paráfrasis aceptable
- similarity < 0.70: Desviación semántica inaceptable

**Estadísticas observadas**:
- Mean similarity (textos válidos): 0.85
- Textos rechazados por similitud: ~12%

#### Criterio 4: Preservación de Keywords Técnicas
```python
tech_terms = ['python', 'java', 'sql', 'react', 'aws', 'docker', ...]
keywords_preserved = len(original_kw ∩ augmented_kw) / len(original_kw)
threshold = 0.70
```

**Justificación**: Términos técnicos (lenguajes de programación, herramientas, metodologías) son cruciales para clasificación de perfiles profesionales. Su pérdida degrada significativamente la calidad del augmentation.

**Ejemplo**:
```
Original keywords: {python, java, sql, docker, kubernetes}
Augmented keywords: {python, java, sql, docker}
Preservation rate: 4/5 = 0.80 → PASS
```

### 3.5 Estrategia de Sampling para Augmentation

**Objetivo**: Llevar clases minoritarias al 80% de la clase mayoritaria.

**Algoritmo**:
```python
threshold = 0.80 * max_class_count
for class in classes:
    if count(class) < threshold:
        needed = threshold - count(class)
        max_augment = count(class) * 2.0  # MAX_AUGMENTATION_RATIO
        needed = min(needed, max_augment)
        
        # Sampling con reemplazo si needed > count(class)
        samples = sample(class, n=needed, replace=(needed > count(class)))
        augment(samples)
```

**Justificación del límite 2.0x**:
- Evita oversampling excesivo que pueda causar overfitting a las augmentations
- Mantiene balance entre diversidad real (ejemplos originales) y sintética (augmentations)

**Resultados**:
- Clase mayoritaria: 120 muestras
- Threshold: 0.80 × 120 = 96 muestras
- Clases bajo threshold: BPO (66), AGRICULTURE (96), APPAREL (97), etc.
- Total agregado: 137 muestras (2,484 → 2,621)

### 3.6 Estadísticas de Augmentation

**Resultados del proceso completo**:

| Métrica | Valor |
|---------|-------|
| Textos procesados | 137 |
| Textos válidos generados | ~120 (87.6%) |
| Textos procesados directamente | ~85 (62%) |
| Textos divididos en chunks | ~52 (38%) |
| Total de chunks procesados | ~180 |
| Promedio chunks por texto largo | 3.46 |
| Tasa de rechazo por similitud | ~12% |
| Tasa de rechazo por keywords | ~8% |
| Similitud semántica promedio | 0.85 ± 0.07 |

**Interpretación**: El sistema logra una alta tasa de éxito (87.6%) manteniendo estándares estrictos de calidad semántica. El chunking dinámico permite procesar textos largos sin degradación de calidad.

---

## 4. División del Dataset: Stratified Group Split

### 4.1 Problema: Data Leakage Prevention

**Escenario problemático**:
```
Original text → Augmented text_1, Augmented text_2
```

Si `Original text` está en train y `Augmented text_1` está en test, el modelo puede memorizar patrones del texto original y "trapear" en test (data leakage).

### 4.2 Solución: Agrupación por original_id

**Estrategia**:
1. Asignar `original_id` a cada texto original
2. Propagar `original_id` a todos sus augmentations
3. Realizar split sobre `original_id` (no sobre muestras individuales)
4. Mantener stratification por clase en el nivel de grupos

**Implementación**:
```python
# Split train-val-test sobre grupos únicos
unique_groups = data['original_id'].unique()
train_groups, test_groups = stratified_split(unique_groups, labels)

# Asignar todas las muestras de cada grupo a su split correspondiente
train_data = data[data['original_id'].isin(train_groups)]
test_data = data[data['original_id'].isin(test_groups)]
```

**Resultado**:
- Train: 2,104 muestras (80%)
- Validation: 262 muestras (10%)
- Test: 255 muestras (10%)

**Garantía**: Ningún texto original aparece en múltiples splits, incluso si tiene augmentations.

### 4.3 Stratification

**Objetivo**: Mantener distribución de clases consistente en train/val/test.

**Método**: `StratifiedShuffleSplit` aplicado a los grupos únicos.

**Verificación**:
```
Train distribution: INFORMATION-TECHNOLOGY (96), ACCOUNTANT (94), ...
Test distribution: INFORMATION-TECHNOLOGY (12), ACCOUNTANT (12), ...

Proportions preserved: ✓
```

---

## 5. Cálculo de Pesos de Clase

### 5.1 Fundamento

A pesar del augmentation, persiste desbalance residual (ratio 1.82x). Los pesos de clase compensan esto durante entrenamiento.

**Fórmula**:
```
weight_class_i = n_samples / (n_classes × n_samples_class_i)
```

**Ejemplo**:
- Total samples: 2,104
- n_classes: 24
- Class ACCOUNTANT: 94 samples

```
weight = 2104 / (24 × 94) = 0.933
```

### 5.2 Distribución de Pesos

```
BPO (n=57)                 → peso: 1.538 (más penalizado)
TEACHER (n=82)             → peso: 1.069
AGRICULTURE (n=76)         → peso: 1.154
...
ACCOUNTANT (n=94)          → peso: 0.933
INFORMATION-TECHNOLOGY (n=96) → peso: 0.913 (menos penalizado)
```

**Interpretación**: 
- Clases pequeñas (BPO) tienen peso 1.68x mayor que clases grandes (IT)
- Durante entrenamiento, errores en BPO contribuyen más a la loss function
- Esto fuerza al modelo a prestar atención a clases minoritarias

### 5.3 Aplicación en Diferentes Modelos

**Modelos neuronales (BiLSTM, CNN-1D, DistilBERT)**:
```python
loss = CrossEntropyLoss(weight=class_weights_tensor)
```
Pesos aplicados directamente en la función de pérdida.

**XGBoost**:
```python
dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weights)
```
Pesos aplicados a nivel de muestra (cada muestra recibe el peso de su clase).

**FastText**:
No soporta pesos directamente, pero el augmentation compensa parcialmente.

---

## 6. Cobertura de Embeddings: Análisis Estadístico

### 6.1 Vocabulario del Dataset

**Estadísticas**:
- Vocabulario único (text_medium): 32,604 palabras
- Top 10 palabras más frecuentes: state (14,209), company (13,303), city (12,803), management (10,691), name (10,005), customer (9,812), service (8,198), work (7,472), sale (7,038), business (6,875)

**Distribución de frecuencias**:
- ~15% del vocabulario: palabras comunes (>1000 ocurrencias)
- ~45% del vocabulario: palabras moderadas (100-1000 ocurrencias)
- ~40% del vocabulario: palabras raras (<100 ocurrencias)

### 6.2 Cobertura de Word2Vec Local

**Modelo**: Word2Vec entrenado localmente en el corpus de 2,621 muestras.

**Configuración**:
- Dimensión: 300
- Window size: 5
- Min count: 2
- Algoritmo: Skip-gram

**Resultados**:
- Palabras en embeddings: 17,758 / 32,604 (54.5%)
- Palabras OOV: 14,846 (45.5%)

**Análisis de OOV**:
Las palabras OOV tienden a ser:
- Nombres propios (empresas, ciudades, personas)
- Términos muy específicos con <2 ocurrencias
- Typos o variantes ortográficas

**Implicaciones**:
- BiLSTM y CNN-1D deben aprender representaciones para ~45% del vocabulario desde cero
- Palabras OOV se inicializan con vectores aleatorios, requiriendo más épocas de entrenamiento para convergencia
- A pesar de esto, ambos modelos logran ROC AUC >0.94, indicando que las palabras cubiertas capturan la mayoría de la información semántica

### 6.3 Comparación con DistilBERT

**Tokenización BPE (Byte Pair Encoding)**:
- DistilBERT divide palabras OOV en subwords conocidos
- Cobertura efectiva: ~100%
- Ejemplo: "kubernetes" → "ku", "ber", "net", "es" (todos en vocabulario)

**Ventaja cuantificable**:
- DistilBERT (+14.1 pp F1-macro sobre CNN-1D)
- Atribuible parcialmente a la cobertura total del vocabulario

---

## 7. Conclusiones del Pipeline

### 7.1 Fortalezas del Diseño

1. **Modularidad**: Tres niveles de limpieza permiten adaptar el preprocesamiento a diferentes arquitecturas sin re-ejecutar todo el pipeline

2. **Robustez del augmentation**: Validación multi-criterio asegura que solo augmentations de alta calidad se incorporan al dataset

3. **Prevención de data leakage**: Stratified group split garantiza evaluación honesta del modelo

4. **Manejo de textos largos**: Chunking dinámico permite procesar currículums completos sin truncamiento

### 7.2 Limitaciones Inherentes

1. **Augmentation limitado**: Solo se agregan 137 muestras (+5.5%), insuficiente para resolver completamente el desbalance

2. **Cobertura de embeddings**: Word2Vec local cubre solo 54.5% del vocabulario, limitando modelos basados en word embeddings

3. **Pérdida de información en text_advanced**: Lematización y remoción de stopwords pueden eliminar matices semánticos relevantes

4. **Sesgo del idioma intermedio**: Back-translation vía español introduce sesgos lingüísticos específicos (e.g., morfología verbal española influye en el resultado)

### 7.3 Impacto Medible en Modelos

El pipeline proporciona una base sólida que permite a todos los modelos superar significativamente baselines simples:

- FastText (51.8% accuracy): Beneficiado por text_basic que preserva n-grams morfológicos
- XGBoost (79.2% accuracy): Maximizado por text_advanced que reduce dimensionalidad de TF-IDF
- DistilBERT (87.5% accuracy): Aprovecha text_advanced pero su arquitectura BPE hace el preprocesamiento menos crítico

El sistema demuestra que un pipeline de preprocesamiento bien diseñado es fundamental para el éxito de modelos de clasificación textual, especialmente en escenarios de datos limitados.
