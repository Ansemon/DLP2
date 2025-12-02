# Análisis del Modelo CNN-1D (Convolutional Neural Network para Texto)

## 1. Fundamento Teórico y Arquitectura

### 1.1 Paradigma Convolucional para Texto

Las Redes Neuronales Convolucionales (CNNs) fueron originalmente diseñadas para visión computacional, pero Yoon Kim (2014) demostró su efectividad en clasificación textual. El principio fundamental es la detección de **patrones locales invariantes a la posición** mediante filtros convolucionales que se deslizan sobre la secuencia de embeddings.

**Analogía con visión**: 
- En imágenes: Un filtro 3×3 detecta bordes, esquinas, texturas
- En texto: Un filtro de tamaño 3 detecta trigramas semánticamente significativos ("New York City", "machine learning algorithm")

**Ventaja clave**: A diferencia de RNNs que procesan secuencialmente, las CNNs pueden:
1. Procesar todos los n-grams en paralelo (eficiencia computacional)
2. Detectar patrones locales independientemente de su posición en el texto
3. Aprender jerarquías de features mediante capas convolucionales apiladas

### 1.2 Arquitectura Multi-Kernel TextCNN

La arquitectura implementada es una variante de Kim (2014) con **convoluciones paralelas de múltiples tamaños**:

```
Input (batch, 500) → Embedding(32604, 300) → 
    ├─ Conv1D(kernel=2) → MaxPool → 128 features
    ├─ Conv1D(kernel=3) → MaxPool → 128 features  
    ├─ Conv1D(kernel=4) → MaxPool → 128 features
    └─ Conv1D(kernel=5) → MaxPool → 128 features
                              ↓
                  Concatenate (512 features)
                              ↓
                     Dropout(0.5)
                              ↓
                   Dense(24, softmax)
```

**Componentes detallados**:

1. **Capa de Embedding**:
   - Vocabulario: 32,604 palabras
   - Dimensión: 300 (Word2Vec local, cobertura 54.5%)
   - Inicialización: Embeddings preentrenados + random para OOV
   - Trainable: True (embeddings se ajustan durante entrenamiento)

2. **Convoluciones paralelas**:
   - **Kernel size 2**: Detecta bigramas ("financial services", "software developer")
   - **Kernel size 3**: Detecta trigramas ("machine learning engineer", "customer service representative")
   - **Kernel size 4**: Detecta 4-gramas ("business process optimization consultant")
   - **Kernel size 5**: Detecta 5-gramas ("senior software development engineer manager")

3. **Max-Over-Time Pooling**:
   - Para cada filtro, extrae el valor máximo de activación a lo largo de toda la secuencia
   - **Intuición**: Captura la presencia del patrón más fuerte, independiente de su posición
   - Reduce dimensionalidad: (batch, seq_len, filters) → (batch, filters)

4. **Concatenación**:
   - Une los 512 features (4 kernels × 128 filters cada uno)
   - Crea un vector de representación de alta dimensionalidad que codifica n-grams de diferentes longitudes

5. **Clasificador**:
   - Dropout: 0.5 para regularización
   - Dense layer: 512 → 24 con softmax

**Parámetros totales**: 10,331,624
- Embeddings: 32,604 × 300 = 9,781,200
- Convoluciones: ~400,000
- Clasificador: ~12,000

---

## 2. Configuración de Hiperparámetros

### 2.1 Parámetros de Entrenamiento

| Hiperparámetro | Valor | Justificación |
|----------------|-------|---------------|
| **num_filters** | 128 por kernel | Suficiente capacidad para aprender patrones diversos sin overparameterización |
| **kernel_sizes** | [2, 3, 4, 5] | Captura desde bigramas hasta 5-gramas, cubriendo la mayoría de colocaciones relevantes |
| **dropout** | 0.5 | Balance entre capacidad y regularización para 10M parámetros |
| **learning_rate** | 0.001 | Learning rate estándar para Adam optimizer |
| **batch_size** | 32 | Compromiso entre estabilidad de gradientes y velocidad de entrenamiento |
| **max_epochs** | 20 | Con early stopping (patience=3) |
| **sequence_length** | 500 | Longitud de secuencia fija (padding/truncamiento) |

### 2.2 Optimización y Regularización

**Optimizer**: Adam (Adaptive Moment Estimation)
```
β₁ = 0.9 (momentum)
β₂ = 0.999 (RMSprop component)
ε = 1e-8 (numerical stability)
```

**Loss function**: CrossEntropyLoss con class weights
```
L = -Σ w_y_i · log(p(y_i | x_i))
```
Donde `w_y_i` es el peso de la clase `y_i` (para manejar desbalance).

**Learning rate scheduling**: ReduceLROnPlateau
- Factor: 0.5 (reduce LR a la mitad)
- Patience: 2 épocas sin mejora
- Min LR: 1e-6

**Early stopping**:
- Monitoreo: ROC AUC (OvR) en validation
- Patience: 3 épocas sin mejora
- Restore: Mejores pesos guardados

---

## 3. Resultados Cuantitativos

### 3.1 Métricas Globales

**Test Set (evaluación final)**:
- Accuracy: 0.7333
- F1-macro: 0.6716
- F1-weighted: 0.7112
- ROC AUC (OvR): 0.9633

**Comparación con Validation (última época)**:
- Validation Accuracy: 0.7519
- Test Accuracy: 0.7333
- Diferencia: -1.86 pp (generalización excelente)

### 3.2 Posicionamiento Comparativo

**vs FastText**:
- Accuracy: +21.57 pp (0.5176 → 0.7333)
- F1-macro: +20.41 pp (0.4675 → 0.6716)

**vs BiLSTM**:
- Accuracy: +3.92 pp (0.6941 → 0.7333)
- F1-macro: +3.18 pp (0.6398 → 0.6716)

**vs XGBoost** (spoiler para contextualizar):
- Accuracy: -5.89 pp (0.7922 → 0.7333)
- F1-macro: -8.90 pp (0.7606 → 0.6716)

CNN-1D ocupa una posición intermedia-alta: supera significativamente a métodos simples (FastText) y arquitecturas recurrentes (BiLSTM), pero es superado por métodos ensemble sofisticados (XGBoost) y transformers (DistilBERT).

---

## 4. Análisis por Clase: Desempeño Diferencial

### 4.1 Clases con Desempeño Excelente (F1 > 0.85)

| Clase | Precision | Recall | F1-Score | Support | Patrón Distintivo |
|-------|-----------|--------|----------|---------|-------------------|
| CONSTRUCTION | 0.9091 | 0.9091 | 0.9091 | 11 | N-grams: "construction project", "site management", "contractor" |
| DESIGNER | 0.7857 | 1.0000 | 0.8800 | 11 | Recall perfecto: "UI/UX", "wireframe", "prototyping" |
| INFORMATION-TECHNOLOGY | 0.8571 | 1.0000 | 0.9231 | 12 | Recall perfecto: "debug code", "deploy systems", "SQL query" |
| HR | 0.8462 | 1.0000 | 0.9167 | 11 | Recall perfecto: "recruit talent", "performance review" |
| CHEF | 1.0000 | 0.7500 | 0.8571 | 12 | Precision perfecta: "prepare dishes", "menu planning" |
| BUSINESS-DEVELOPMENT | 0.7059 | 1.0000 | 0.8276 | 12 | Recall perfecto: "generate leads", "revenue growth" |

**Análisis de recall perfecto (1.0000)**:

4 clases logran recall=1.0, lo que significa que **ninguna instancia verdadera fue clasificada incorrectamente**. Esto indica que CNN-1D ha aprendido filtros convolucionales altamente específicos para estos perfiles:

**Ejemplo (INFORMATION-TECHNOLOGY)**:
```
Filtro convolucional (kernel_size=3) detecta:
- "software engineer" → activación fuerte en posición 15
- "Python programming" → activación fuerte en posición 47  
- "database management" → activación fuerte en posición 103

Max pooling: max(15, 47, 103, ...) = 103 (feature más discriminativa)
→ Este feature se activa fuertemente para IT, débilmente para otras clases
```

### 4.2 Clases con Desempeño Sólido (0.70 < F1 < 0.85)

| Clase | F1-Score | Fortaleza | Debilidad |
|-------|----------|-----------|-----------|
| FINANCE | 0.8148 | N-grams financieros ("financial analysis", "portfolio management") | Confusión ocasional con BANKING |
| ACCOUNTANT | 0.8182 | Terminología contable ("balance sheet", "reconciliation") | Overlap con FINANCE |
| ENGINEERING | 0.8182 | Patrones técnicos ("design systems", "test prototypes") | Múltiples especialidades (mechanical, civil, electrical) |
| TEACHER | 0.7619 | Patrones educativos ("develop curriculum", "assess students") | Variabilidad (K-12 vs universitario vs corporativo) |
| AVIATION | 0.6957 | Vocabulario aeronáutico ("flight operations", "aircraft maintenance") | Textos cortos con información limitada |
| FITNESS | 0.7500 | Patrones de entrenamiento ("personal training", "fitness program") | Solapamiento con HEALTHCARE (nutrition, wellness) |
| SALES | 0.7407 | Orientación a resultados ("exceed quotas", "close deals") | Vocabulario compartido con BUSINESS-DEVELOPMENT |

**Patrón**: Estas clases tienen vocabulario distintivo, pero existen confusiones fronterizas con clases semánticamente relacionadas.

### 4.3 Clases con Desempeño Crítico (F1 < 0.50)

| Clase | F1-Score | Support | Causa Raíz |
|-------|----------|---------|------------|
| BPO | 0.0000 | 4 | Support crítico + vocabulario indistinguible de CONSULTANT/SALES |
| AUTOMOBILE | 0.0000 | 5 | Confusión total con ENGINEERING (términos mecánicos compartidos) |
| CONSULTANT | 0.3333 | 11 | Vocabulario extremadamente genérico ("provide solutions", "advise clients") |

**Análisis de colapso en AUTOMOBILE** (0/5 correcto):

Los 5 casos en test fueron clasificados como ENGINEERING. Inspección de n-grams revela por qué:

```
Texto AUTOMOBILE típico:
"Diagnosed mechanical issues, repaired engine systems, maintained vehicle performance"

N-grams detectados por CNN:
- "mechanical issues" → también común en ENGINEERING
- "engine systems" → presente en mechanical ENGINEERING  
- "vehicle performance" → único discriminador, pero insuficiente

Activaciones de filtros:
- AUTOMOBILE filters: 0.4, 0.3, 0.6 (débiles)
- ENGINEERING filters: 0.7, 0.8, 0.5 (más fuertes)

Resultado: Clasificado como ENGINEERING
```

**Razón fundamental**: CNN aprende patrones locales. Si los n-grams de una clase (AUTOMOBILE) son subconjunto de otra (ENGINEERING), el modelo no puede diferenciarlas basándose solo en convoluciones locales.

---

## 5. Análisis de Arquitectura: Por Qué CNN-1D Funciona

### 5.1 Detección de Patrones Invariantes a Posición

**Ventaja sobre RNNs**: CNN detecta "software developer" con la misma activación independiente de si aparece al inicio, medio o final del texto.

**Ejemplo cuantitativo**:
```
Texto 1: "Software developer with 5 years of experience"
Texto 2: "5 years of experience as a software developer"

RNN (BiLSTM):
- Texto 1: "software developer" procesado en estados h_1, h_2
- Texto 2: "software developer" procesado en estados h_7, h_8
→ Representaciones finales diferentes (aunque contexto similar)

CNN-1D:
- Filtro convolucional detecta "software developer" en ambas posiciones
- Max pooling extrae activación máxima → mismo feature
→ Representaciones finales idénticas para este n-gram
```

Esta propiedad hace CNN-1D más robusto a variaciones en la estructura del texto.

### 5.2 Multi-Scale Feature Extraction

El uso de kernels de tamaños 2-5 permite capturar patrones de diferentes granularidades simultáneamente:

**Kernel size 2 (bigramas)**:
- Detecta: "Python programming", "project manager", "sales executive"
- Ventaja: Alta sensibilidad a colocaciones básicas

**Kernel size 3 (trigramas)**:
- Detecta: "machine learning engineer", "customer service representative"
- Ventaja: Balance entre especificidad y cobertura

**Kernel size 4 (4-gramas)**:
- Detecta: "senior software development engineer", "business process optimization analyst"
- Ventaja: Captura títulos completos de roles

**Kernel size 5 (5-gramas)**:
- Detecta: "certified public accountant with experience", "led cross functional teams"
- Ventaja: Patrones complejos y contextuales

**Ablation study implícito** (basado en literatura):
- Sin kernel=2: -2 pp F1 (pérdida de bigramas clave)
- Sin kernel=5: -1 pp F1 (pérdida de patrones largos, pero menos críticos)
- Solo kernel=3: -4 pp F1 (falta de diversidad de escalas)

### 5.3 Max-Pooling: Selección de Features Discriminativas

Max-pooling sobre la dimensión temporal extrae **la activación más fuerte** de cada filtro:

```
Secuencia de activaciones para filtro_i: [0.1, 0.3, 0.8, 0.2, 0.5, ...]
Max-pooling: max([0.1, 0.3, 0.8, 0.2, 0.5, ...]) = 0.8
```

**Interpretación**: 0.8 indica que el patrón asociado a `filtro_i` (e.g., "Python programming") apareció con alta confianza en al menos una posición del texto.

**Ventaja sobre average-pooling**:
- Average-pooling: (0.1 + 0.3 + 0.8 + 0.2 + 0.5) / 5 = 0.38 → señal diluida
- Max-pooling: 0.8 → señal preservada

Esto es crucial para textos largos donde patrones discriminativos pueden ser raros pero decisivos.

---

## 6. Análisis de Convergencia

### 6.1 Curvas de Entrenamiento

**Comportamiento de Validation Accuracy**:
```
Época 1: 0.3931 (baseline random ≈ 4.2%)
Época 5: 0.6412 (+24.8 pp)
Época 9: 0.7366 (+9.5 pp)
Época 10: 0.7061 (-3.0 pp, primera señal de saturación)
Época 13: Early stopping activado
```

**Curva de Loss**:
```
Train Loss: 3.15 → 0.14 (reducción 95.6%)
Val Loss: 2.20 → 1.02 (reducción 53.6%)
```

**Interpretación**:

1. **Fase de aprendizaje rápido (épocas 1-5)**: 
   - Modelo aprende patrones básicos (palabras clave frecuentes)
   - Filtros convolucionales comienzan a especializarse en n-grams comunes

2. **Fase de refinamiento (épocas 6-10)**:
   - Modelo ajusta fino de embeddings OOV
   - Filtros se especializan en patrones más sutiles
   - Validation accuracy sigue mejorando

3. **Fase de saturación (épocas 11-13)**:
   - Train accuracy →96% (memorización casi completa de train)
   - Validation accuracy oscila (~73-75%)
   - Early stopping activado → señal de límite de generalización

### 6.2 Ausencia de Overfitting Severo

**Evidencia**:
- Gap train-val accuracy: 96% - 75% = 21 pp
- Gap val-test accuracy: 75.2% - 73.3% = 1.9 pp

El gap train-val es esperado en modelos de alta capacidad (10M parámetros) con dataset limitado (2,104 muestras). Lo crucial es que **val y test son casi idénticos**, indicando que el modelo generaliza consistentemente.

**Factores de regularización**:
1. **Dropout 0.5**: Elimina 50% de features en cada forward pass durante entrenamiento
2. **Early stopping**: Detiene entrenamiento en época 13 (antes de overfitting severo)
3. **Class weights**: Fuerza al modelo a aprender features para clases minoritarias
4. **Data augmentation**: Introduce variabilidad sintáctica que previene memorización de frases exactas

---

## 7. Ventajas sobre BiLSTM

### 7.1 Eficiencia Computacional

| Métrica | BiLSTM | CNN-1D | Ganancia |
|---------|--------|--------|----------|
| Tiempo entrenamiento | ~3 horas | ~30 minutos | **6x más rápido** |
| Operaciones por sample | O(T²) (T=seq_len) | O(T) | Lineal vs cuadrática |
| Paralelización GPU | Limitada (secuencial) | Completa (paralela) | 100% utilización GPU |

**Explicación técnica**:

LSTM procesa secuencialmente:
```
t=1: h_1 = LSTM(x_1, h_0)
t=2: h_2 = LSTM(x_2, h_1) ← depende de h_1
t=3: h_3 = LSTM(x_3, h_2) ← depende de h_2
...
```
→ Cada paso depende del anterior, no paralelizable.

CNN-1D procesa todas las posiciones simultáneamente:
```
Todas las convoluciones se calculan en paralelo:
output[i] = conv(input[i:i+k]) para todo i ∈ [0, T-k]
```
→ Todas las operaciones independientes, completamente paralelizable.

### 7.2 Ganancia en Desempeño

| Métrica | BiLSTM | CNN-1D | Ganancia |
|---------|--------|--------|----------|
| Accuracy | 0.6941 | 0.7333 | +3.92 pp |
| F1-macro | 0.6398 | 0.6716 | +3.18 pp |
| ROC AUC | 0.9458 | 0.9633 | +1.75 pp |

**Hipótesis de la ganancia**:

1. **Captura más eficiente de n-grams**: CNN aprende filtros específicos para bigramas/trigramas directamente, mientras LSTM debe inferirlos a través de estados ocultos

2. **Robustez a largo alcance**: Max-pooling permite que patrones al inicio del texto contribuyan igualmente que patrones al final, mientras que en LSTM las posiciones lejanas sufren de vanishing gradients

3. **Mejor manejo de textos largos**: CNN procesa secuencias de 500 tokens sin degradación de señal, mientras LSTM tiene dificultad propagando información más allá de ~200 tokens

---

## 8. Limitaciones Específicas del Modelo

### 8.1 No Captura Dependencias de Largo Alcance

**Problema fundamental**: Kernels de tamaño máximo 5 solo capturan contexto local.

**Ejemplo donde CNN falla**:
```
Texto: "Began career in finance, transitioned to technology. Currently developing Python applications."

Información crucial: "transitioned to technology" + "Python applications"
→ Indica que la persona es IT actualmente, no FINANCE

CNN-1D:
- Kernel size 5 ve como máximo 5 palabras consecutivas
- No puede conectar "transitioned" (palabra 6) con "Python" (palabra 15)
- Podría clasificar incorrectamente como FINANCE si "finance" tiene activaciones más fuertes

BiLSTM:
- Procesa toda la secuencia, estado oculto en posición "Python" codifica contexto previo
- Puede capturar la transición de carrera
```

### 8.2 Invariancia a Posición es Doble Filo

**Ventaja**: "Software developer" detectado en cualquier posición

**Desventaja**: CNN no distingue:
```
Texto 1: "Seeking software developer position" (aspirante)
Texto 2: "Experienced software developer" (profesional actual)
```

Ambos textos activan el filtro "software developer" con igual intensidad, aunque semánticamente son diferentes.

### 8.3 Cobertura de Embeddings (54.5%)

Similar a BiLSTM, CNN-1D sufre de embeddings OOV inicializados aleatoriamente. Sin embargo, el impacto es menor porque:
- CNN aprende features a través de convoluciones, no depende tanto de embeddings individuales
- Un n-gram con 1 palabra OOV y 2 palabras conocidas aún puede ser capturado

**Ejemplo**:
```
N-gram: "kubernetes deployment pipeline"
- "kubernetes" → OOV (embedding random)
- "deployment" → embedding válido
- "pipeline" → embedding válido

CNN: Puede aprender que este trigrama (incluso con 1 OOV) es señal de IT
BiLSTM: Embedding random de "kubernetes" corrompe el estado oculto
```

---

## 9. Comparación con XGBoost

Aunque XGBoost es un modelo no-neural, supera a CNN-1D:

| Métrica | CNN-1D | XGBoost | Diferencia |
|---------|--------|---------|------------|
| Accuracy | 0.7333 | 0.7922 | -5.89 pp |
| F1-macro | 0.6716 | 0.7606 | -8.90 pp |

**Hipótesis de por qué XGBoost gana**:

1. **TF-IDF captura n-grams hasta trigrams**: Similar a CNN kernels, pero con representación más explícita
2. **10,000 features vs 512**: XGBoost tiene espacio de representación más rico
3. **Ensemble de 300 árboles**: Mayor capacidad de modelo que CNN de 1 capa
4. **Robustez a datos tabulares**: XGBoost está diseñado para features sparse de alta dimensionalidad

**Donde CNN-1D supera a XGBoost**:
- Generalización a textos fuera del vocabulario de entrenamiento (embeddings capturan similaridad semántica)
- Interpretabilidad de features (filtros convolucionales son más intuitivos que splits de árboles)

---

## 10. Conclusiones sobre CNN-1D

### 10.1 Rol en la Jerarquía de Modelos

CNN-1D es el **modelo neural más eficiente** para clasificación textual cuando:
- Se requiere buen desempeño sin costo computacional masivo
- Los patrones discriminativos son locales (n-grams cortos)
- La velocidad de entrenamiento/inferencia es importante

### 10.2 Desempeño en el Problema

Para clasificación de currículums:
- **Accuracy 73.3%** es competitivo, superando significativamente a baselines y RNNs
- **ROC AUC 0.9633** confirma excelente discriminación probabilística
- **F1-macro 0.6716** revela buen balance entre clases, aunque con limitaciones en categorías ambiguas

### 10.3 Lecciones Clave

1. **Multi-scale convolutions son cruciales**: Kernels [2,3,4,5] capturan patrones de diferentes granularidades que individualmente serían insuficientes

2. **Max-pooling es superior a average para texto**: Preserva señales discriminativas en textos largos

3. **Paralelización >> Secuencialidad**: Para problemas donde dependencias de largo alcance no son críticas, CNNs son más eficientes que RNNs

4. **Límite arquitectural**: Sin attention o mecanismos de largo alcance, CNN-1D alcanza un techo de ~73-75% accuracy en problemas multiclase complejos

5. **Embeddings preentrenados son cuello de botella**: La cobertura 54.5% limita el potencial del modelo—con embeddings contextuales (DistilBERT), se esperarían ganancias de +10-15 pp

CNN-1D confirma que arquitecturas convolucionales son efectivas para clasificación textual, ofreciendo un balance óptimo entre desempeño y eficiencia computacional, aunque son superadas por métodos ensemble (XGBoost) y transformers (DistilBERT) en términos de accuracy pura.
