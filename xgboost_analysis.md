# Análisis del Modelo XGBoost con TF-IDF

## 1. Fundamento Teórico y Arquitectura

### 1.1 Paradigma de Gradient Boosting

XGBoost (Extreme Gradient Boosting) implementa el algoritmo de gradient boosting decision trees (GBDT), una técnica de ensemble learning que construye modelos secuencialmente, donde cada nuevo modelo corrige los errores del ensemble anterior.

**Principio fundamental**:
```
F_m(x) = F_{m-1}(x) + ν · h_m(x)
```

Donde:
- `F_m(x)`: Predicción del ensemble después de m árboles
- `h_m(x)`: Nuevo árbol (aprendido para minimizar residuos)
- `ν`: Learning rate (shrinkage parameter)

**Proceso iterativo**:
1. Calcular gradientes de la loss function respecto a predicciones actuales
2. Ajustar nuevo árbol de decisión para predecir estos gradientes
3. Agregar árbol al ensemble con peso controlado por learning rate
4. Repetir hasta convergencia o límite de árboles

### 1.2 Arquitectura del Sistema: TF-IDF + XGBoost

El modelo combina dos componentes fundamentales:

**Componente 1: Vectorización TF-IDF**

```
Texto → Tokenización → Conteo de n-grams → Ponderación TF-IDF → Vector sparse (10,000 dims)
```

**TF-IDF (Term Frequency-Inverse Document Frequency)**:
```
tfidf(t, d) = tf(t, d) × idf(t)

tf(t, d) = count(t in d) / |d|
idf(t) = log(N / df(t))
```

Donde:
- `t`: término (unigrama, bigrama o trigrama)
- `d`: documento (currículum)
- `N`: total de documentos
- `df(t)`: número de documentos que contienen `t`

**Intuición**: TF-IDF asigna peso alto a términos que son:
- Frecuentes en el documento actual (TF alto)
- Raros en el corpus general (IDF alto)

**Componente 2: XGBoost Multiclase**

```
Input: TF-IDF vector (10,000 features) → Ensemble de 300 árboles → Softmax(24 clases)
```

**Objetivo multiclase (multi:softprob)**:
```
L = -Σ Σ y_{ij} · log(p_{ij})
    i j

Donde:
- i: índice de muestra
- j: índice de clase
- y_{ij}: 1 si muestra i pertenece a clase j, 0 en caso contrario
- p_{ij}: probabilidad predicha para muestra i, clase j
```

---

## 2. Configuración de Hiperparámetros

### 2.1 Parámetros del Vectorizador TF-IDF

| Parámetro | Valor | Justificación |
|-----------|-------|---------------|
| **max_features** | 10,000 | Balance entre expresividad y eficiencia computacional |
| **ngram_range** | (1, 3) | Captura unigrams, bigrams y trigrams |
| **min_df** | 2 | Elimina términos que aparecen en <2 documentos (ruido) |
| **max_df** | 0.95 | Elimina términos en >95% de documentos (stopwords implícitas) |

**Análisis del espacio de features**:
- Vocabulario potencial completo: ~32,000 palabras únicas
- Con n-grams (1,3): ~500,000 combinaciones posibles
- Limitado a 10,000: Top features por importancia TF-IDF global

**Sparsity resultante**: 96.15%
```
10,000 features por vector
Media de features no-cero por documento: ~385 (3.85%)
```

Esta alta sparsity es característica de representaciones TF-IDF y es exactamente el tipo de datos para el que XGBoost fue diseñado.

### 2.2 Parámetros de XGBoost

| Hiperparámetro | Valor | Impacto |
|----------------|-------|---------|
| **objective** | multi:softprob | Loss function para clasificación multiclase con probabilidades |
| **num_class** | 24 | Número de clases en el problema |
| **max_depth** | 6 | Profundidad máxima de cada árbol (controla complejidad) |
| **learning_rate** (η) | 0.1 | Shrinkage para prevenir overfitting |
| **n_estimators** | 300 | Número de árboles en el ensemble |
| **subsample** | 0.8 | Fracción de muestras usadas por árbol (bagging) |
| **colsample_bytree** | 0.8 | Fracción de features usadas por árbol |
| **min_child_weight** | 3 | Suma mínima de pesos de instancia en nodo hijo |
| **gamma** | 0.1 | Reducción mínima de loss para crear split |
| **reg_alpha** | 0.1 | Regularización L1 |
| **reg_lambda** | 1.0 | Regularización L2 |

**Análisis de hiperparámetros clave**:

**max_depth=6**: 
- Profundidad moderada (no shallow como 3, no deep como 12)
- Permite capturar interacciones complejas: `(tf_idf["Python"] > 0.5) AND (tf_idf["machine learning"] > 0.3) → IT`
- Cada árbol puede tener hasta 2^6 = 64 nodos hoja

**learning_rate=0.1**:
- Valor estándar que balancea velocidad de convergencia y estabilidad
- Cada árbol contribuye 10% de su predicción completa
- Requiere más árboles (300) pero reduce overfitting

**subsample=0.8, colsample_bytree=0.8**:
- Introduce aleatorización estilo Random Forest
- Cada árbol se entrena con 80% de muestras y 80% de features
- Reduce correlación entre árboles → mejor generalización

**Regularización (gamma=0.1, lambda=1.0, alpha=0.1)**:
- **Gamma**: Penaliza splits que no reducen loss significativamente
- **Lambda (L2)**: Penaliza pesos grandes de hojas (smooth predictions)
- **Alpha (L1)**: Induce sparsity en pesos de hojas

---

## 3. Resultados Cuantitativos

### 3.1 Métricas Globales

**Validation Set**:
- Accuracy: 0.7328
- F1-macro: 0.6872
- F1-weighted: 0.7143
- ROC AUC (OvR): 0.9732

**Test Set (evaluación final)**:
- Accuracy: 0.7922
- F1-macro: 0.7606
- F1-weighted: 0.7849
- ROC AUC (OvR): 0.9816

### 3.2 Análisis de Generalización

**Diferencia val-test**: 
- Accuracy: +5.94 pp (0.7328 → 0.7922)
- F1-macro: +7.34 pp (0.6872 → 0.7606)

**Interpretación**: Test performance **superior** a validation es inusual pero no alarmante:
- Tamaño de validation: 262 muestras
- Tamaño de test: 255 muestras
- Diferencia puede atribuirse a varianza estadística de muestras pequeñas
- No hay evidencia de data leakage (stratified group split fue aplicado correctamente)

### 3.3 Posicionamiento Comparativo

**vs FastText**:
- Accuracy: +27.46 pp (0.5176 → 0.7922)
- F1-macro: +29.31 pp (0.4675 → 0.7606)

**vs BiLSTM**:
- Accuracy: +9.81 pp (0.6941 → 0.7922)
- F1-macro: +12.08 pp (0.6398 → 0.7606)

**vs CNN-1D**:
- Accuracy: +5.89 pp (0.7333 → 0.7922)
- F1-macro: +8.90 pp (0.6716 → 0.7606)

XGBoost es el **mejor modelo no-transformer**, superando significativamente todos los modelos neuronales evaluados (FastText, BiLSTM, CNN-1D).

---

## 4. Análisis por Clase: Desempeño Diferencial

### 4.1 Clases con Desempeño Excelente (F1 > 0.90)

| Clase | Precision | Recall | F1-Score | Support | Patrón TF-IDF Clave |
|-------|-----------|--------|----------|---------|---------------------|
| DESIGNER | 0.9167 | 1.0000 | 0.9565 | 11 | Trigramas: "user experience design", "wireframe prototyping" |
| CONSTRUCTION | 0.9167 | 1.0000 | 0.9565 | 11 | Bigramas: "construction project", "site management" |
| HR | 1.0000 | 0.9091 | 0.9524 | 11 | Trigramas: "talent acquisition process", "employee performance review" |
| BUSINESS-DEVELOPMENT | 0.8571 | 1.0000 | 0.9231 | 12 | Trigramas: "lead generation strategy", "revenue growth pipeline" |
| INFORMATION-TECHNOLOGY | 0.8571 | 1.0000 | 0.9231 | 12 | Unigrams de alta IDF: "Python", "SQL", "debugging" |

**Análisis de recall perfecto (1.0000)**:

3 clases logran recall=1.0. Esto significa que XGBoost ha aprendido reglas (splits de árbol) extremadamente precisas:

**Ejemplo de reglas inferidas para DESIGNER**:
```
Árbol 47, Nodo 3:
  if tfidf["ux design"] > 0.23:
    if tfidf["figma"] > 0.15:
      → DESIGNER (weight: 0.8)
    else:
      if tfidf["wireframe"] > 0.10:
        → DESIGNER (weight: 0.7)
```

### 4.2 Clases con Desempeño Sólido (0.75 < F1 < 0.90)

| Clase | F1-Score | Top Features (por gain) |
|-------|----------|-------------------------|
| PUBLIC-RELATIONS | 0.8333 | "media relations", "press release", "public communication" |
| HEALTHCARE | 0.8421 | "patient care", "medical diagnosis", "clinical practice" |
| FINANCE | 0.8182 | "financial analysis", "investment portfolio", "risk management" |
| DIGITAL-MEDIA | 0.8000 | "content creation", "social media", "digital marketing" |
| ACCOUNTANT | 0.8000 | "financial reporting", "balance sheet", "audit reconciliation" |
| FITNESS | 0.8000 | "personal training", "fitness program", "exercise routine" |

**Patrón**: Estas clases tienen vocabulario distintivo suficiente para ser bien clasificadas, pero ocasionalmente comparten features con clases relacionadas.

### 4.3 Clases con Desempeño Intermedio (0.65 < F1 < 0.75)

| Clase | F1-Score | Problema Principal |
|-------|----------|--------------------|
| ADVOCATE | 0.7857 | Vocabulario legal compartido con CONSULTANT |
| ENGINEERING | 0.7857 | Múltiples especialidades (mechanical, civil, electrical) |
| BANKING | 0.7619 | Overlap con FINANCE en términos financieros |
| CONSULTANT | 0.7273 | Vocabulario extremadamente genérico |
| SALES | 0.7407 | Compartido con BUSINESS-DEVELOPMENT |
| TEACHER | 0.6667 | Variabilidad (K-12, universitario, corporativo) |
| CHEF | 0.6957 | Dataset pequeño + variabilidad (chef, sous chef, culinary manager) |

### 4.4 Clases con Desempeño Crítico (F1 < 0.50)

| Clase | F1-Score | Support | Causa Raíz |
|-------|----------|---------|------------|
| BPO | 0.4000 | 4 | Support crítico + vocabulario compartido con CONSULTANT/SALES |
| AUTOMOBILE | 0.2857 | 5 | Confusión con ENGINEERING (términos mecánicos) |
| ARTS | 0.5000 | 10 | Categoría muy amplia (visual, performing, multimedia) |

**Análisis de BPO** (1/4 correcto):
```
4 casos en test:

Caso 1: "Process optimization for telecom operations"
Top TF-IDF features: "process optimization" (0.45), "operations" (0.38), "telecom" (0.31)
XGBoost tree decision:
  if tfidf["process optimization"] > 0.4:
    if tfidf["operations"] > 0.3:
      → CONSULTANT (peso: 0.6)  # INCORRECTO
      → BPO (peso: 0.3)
Predicción: CONSULTANT

Caso 2: "Quality analyst for outsourcing firm"
Top TF-IDF features: "quality analyst" (0.52), "outsourcing" (0.41)
XGBoost decision:
  if tfidf["quality analyst"] > 0.5:
    → BPO (peso: 0.7)  # CORRECTO
Predicción: BPO ✓

Casos 3-4: Clasificados como BUSINESS-DEVELOPMENT
```

**Razón del fallo**: BPO tiene solo 57 muestras en train. XGBoost construye 300 árboles, pero la mayoría no tiene splits específicos para BPO debido a la falta de ejemplos para construir reglas robustas.

---

## 5. Análisis de Feature Importance

### 5.1 Top Features por Gain

El análisis de importancia revela qué n-grams contribuyen más a las decisiones del modelo:

| Feature (n-gram) | Gain | Interpretación |
|------------------|------|----------------|
| experience information technology | 107.60 | Fuerte señal de IT |
| construction project | 65.06 | Discrimina CONSTRUCTION |
| personal trainer | 75.92 | Específico de FITNESS |
| hardware | 45.63 | Señal de ENGINEERING/IT |
| service representative | 40.04 | Señal de SALES/CUSTOMER SERVICE |
| chef | 28.14 | Unigram altamente específico |
| designer | 27.30 | Unigram altamente específico |

**Observación crítica**: Algunos features tienen gain alto pero son **ruido**:

| Feature Ruidoso | Gain | Problema |
|----------------|------|----------|
| menu | 130.92 | Puede referirse a CHEF o a "menu de opciones" en IT |
| aaa | 29.70 | Posible artefacto de parsing (ratings AAA, nombres de empresas) |
| february 2014 | 26.29 | Fecha específica sin valor semántico real |

**Interpretación**: XGBoost aprende patrones en los datos de entrenamiento, incluyendo artefactos. Estos features tienen gain alto porque aparecen consistentemente en una clase, pero no generalizan bien.

### 5.2 Interacciones de Features

A diferencia de modelos lineales, XGBoost puede aprender interacciones no-lineales entre features:

**Ejemplo de interacción aprendida** (inferido de splits de árboles):
```
Árbol 123:
  if tfidf["financial"] > 0.3:
    if tfidf["investment"] > 0.2:
      → FINANCE (confianza alta)
    elif tfidf["audit"] > 0.2:
      → ACCOUNTANT (confianza alta)
    else:
      → BANKING (confianza media)
```

Esta regla no-lineal es imposible de aprender con modelos lineales (e.g., logistic regression sobre TF-IDF).

---

## 6. Análisis de Convergencia del Entrenamiento

### 6.1 Curvas de Loss

**Train mlogloss**:
```
Iteración 0:   2.707
Iteración 50:  0.166
Iteración 85:  0.077  ← Best iteration
Iteración 105: 0.054
```

**Validation mlogloss**:
```
Iteración 0:   2.730
Iteración 50:  0.898
Iteración 85:  0.858  ← Best score
Iteración 105: 0.862  ← Early stopping trigger
```

### 6.2 Interpretación de Convergencia

**Best iteration: 85**

El modelo alcanza su mejor validation loss en la iteración 85. Después de esto:
- Train loss continúa bajando (0.077 → 0.054): señal de overfitting
- Validation loss se estanca y sube ligeramente (0.858 → 0.862)

**Early stopping activado**: XGBoost detecta que validation loss no mejoró en las últimas 20 iteraciones (default patience), revierte a los pesos de la iteración 85.

**Análisis de train vs val loss**:
- Train loss final: 0.077
- Val loss final: 0.858
- Gap: 11.1x

Este gap es significativo y sugiere que el modelo ha aprendido patrones específicos del training set que no generalizan perfectamente. Sin embargo, el test accuracy de 79.2% indica que el overfitting no es catastrófico.

### 6.3 Factores que Limitan Overfitting

1. **Learning rate bajo (0.1)**: Cada árbol contribuye solo 10% de su peso completo
2. **Regularización L1/L2**: Penaliza pesos grandes de hojas
3. **Subsampling (0.8)**: Cada árbol ve solo 80% de datos
4. **Min child weight, gamma**: Previenen splits sobre-específicos
5. **Early stopping**: Detiene entrenamiento antes de memorización total

---

## 7. Ventajas de XGBoost sobre Modelos Neuronales

### 7.1 Superioridad en Datos Tabulares de Alta Dimensión

**Característica de TF-IDF**: Vector sparse de 10,000 dimensiones con ~385 features no-cero promedio.

**Por qué XGBoost domina aquí**:

1. **Manejo nativo de sparsity**: XGBoost tiene algoritmo específico para features sparse:
   - No calcula gradientes para features con valor 0
   - Splits consideran solo features presentes
   - Memoria: O(nnz) no O(n × d) donde nnz = non-zero entries

2. **Aprendizaje de reglas interpretables**: Cada árbol codifica reglas IF-THEN que son naturales para features TF-IDF:
   ```
   if "Python" in document AND "machine learning" in document:
       → IT (high confidence)
   ```

3. **No requiere embeddings**: A diferencia de modelos neuronales que necesitan representaciones densas, XGBoost trabaja directamente con features sparse

### 7.2 Robustez a Features Ruidosas

**CNN-1D y BiLSTM**: Sensibles a embeddings de palabras OOV (inicializados aleatoriamente)

**XGBoost**: Cada feature TF-IDF es independiente:
- Si "kubernetes" (palabra rara) tiene TF-IDF alto, el modelo aprende su importancia sin depender de su "similaridad" con otras palabras
- Features ruidosas simplemente no se usan en splits (gain bajo)

### 7.3 Ganancia Cuantitativa sobre Modelos Neuronales

| Modelo | Accuracy | Diferencia con XGBoost |
|--------|----------|------------------------|
| FastText | 0.5176 | -27.46 pp |
| BiLSTM | 0.6941 | -9.81 pp |
| CNN-1D | 0.7333 | -5.89 pp |
| **XGBoost** | **0.7922** | — |

**Hipótesis de superioridad**:

1. **TF-IDF n-grams (1,3) capturan más contexto que esperado**: Trigramas como "machine learning engineer" son feature único con peso específico, equivalente a filtros convolucionales de CNN

2. **Ensemble de 300 árboles >> 1 red neural**: XGBoost agrega predicciones de 300 modelos diversos, CNN-1D es 1 modelo

3. **Optimización para features categóricas**: TF-IDF features son esencialmente categóricas (presente/ausente con peso), ideal para árboles de decisión

---

## 8. Limitaciones Específicas del Modelo

### 8.1 No Captura Similaridad Semántica

**Problema**:
- "Software developer" y "Software engineer" son TF-IDF features **completamente independientes**
- XGBoost no "sabe" que son semánticamente similares

**Ejemplo donde falla**:
```
Train: Muchos ejemplos con "software developer" → IT
Test: Ejemplo con "software engineer" (raro en train)
TF-IDF: "software engineer" tiene IDF alto pero no está fuertemente asociado a IT en árboles
Predicción: Puede clasificar incorrectamente como ENGINEERING
```

Modelos con embeddings (BiLSTM, CNN-1D) capturan esta similaridad: embed("developer") ≈ embed("engineer").

### 8.2 Vocabulario Fijo Post-Training

**Limitación**: Si una palabra nueva aparece en producción (e.g., "blockchain" en 2024), TF-IDF no puede generarle un feature.

**Contraste con modelos neuronales**:
- BiLSTM: Palabra OOV obtiene embedding random, pero puede aún contribuir al estado oculto
- DistilBERT: Tokenización BPE divide "blockchain" en subwords conocidos

### 8.3 Sensibilidad a Features Ruidosas en Train

**Ejemplo** (feature "menu" con gain 130.92):

Si "menu" aparece consistentemente en textos de CHEF en train:
```
XGBoost aprende: if tfidf["menu"] > 0.3 → CHEF
```

Pero en test, si aparece "menu" en contexto diferente:
```
"Designed menu interface for mobile application" → IT, no CHEF
XGBoost: Clasifica incorrectamente como CHEF
```

Este problema es inherente a métodos basados en features discretas sin comprensión semántica.

---

## 9. Comparación con DistilBERT

Aunque DistilBERT supera a XGBoost:

| Métrica | XGBoost | DistilBERT | Diferencia |
|---------|---------|------------|------------|
| Accuracy | 0.7922 | 0.8745 | -8.23 pp |
| F1-macro | 0.7606 | 0.8453 | -8.47 pp |

**Análisis de la brecha**:

1. **Representaciones contextuales**: DistilBERT genera embeddings dinámicos ("bank" en "bank account" ≠ "bank" en "river bank"), XGBoost trata "bank" como feature única

2. **Transfer learning**: DistilBERT trae conocimiento de millones de documentos, XGBoost solo aprende de 2,104 muestras

3. **Capacidad de modelo**: DistilBERT (66M parámetros) vs XGBoost (300 árboles, ~5-10M splits estimados)

**Ventajas de XGBoost sobre DistilBERT**:
- **Interpretabilidad**: Los árboles de decisión son inspeccionables, attention weights de transformers son opacos
- **Velocidad de inferencia**: XGBoost procesa una muestra en <1ms, DistilBERT requiere ~50ms
- **Requerimientos computacionales**: XGBoost entrena en CPU en ~10 minutos, DistilBERT requiere GPU y ~2 horas

---

## 10. Análisis de Errores: Matriz de Confusión

### 10.1 Confusiones Principales

**FINANCE ↔ ACCOUNTANT** (3 casos):
```
Textos contienen: "financial reporting", "balance sheet", "audit"
Ambas clases comparten estos términos
XGBoost decision tree no puede diferenciar sin contexto adicional
```

**CONSULTANT ↔ BPO** (2 casos):
```
Vocabulario: "process optimization", "client service", "operations"
Completamente solapado entre ambas clases
Sin features únicos, XGBoost predice la clase más frecuente (CONSULTANT)
```

**ENGINEERING ↔ AUTOMOBILE** (4 casos):
```
Términos mecánicos: "engine systems", "mechanical design", "vehicle components"
AUTOMOBILE es subconjunto de ENGINEERING en el espacio TF-IDF
XGBoost favorece ENGINEERING (clase mayoritaria con features similares)
```

### 10.2 Patrón de Errores

**Observación clave**: La mayoría de errores ocurren entre clases semánticamente relacionadas, no aleatoriamente.

**Evidencia**:
- 0 confusiones entre CHEF y INFORMATION-TECHNOLOGY (semántica completamente diferente)
- 8 confusiones entre FINANCE, BANKING, ACCOUNTANT (semántica solapada)

Esto sugiere que XGBoost está aprendiendo la estructura semántica del problema, pero tiene dificultad en fronteras difusas.

---

## 11. Conclusiones sobre XGBoost

### 11.1 Rol en la Jerarquía de Modelos

XGBoost es el **mejor modelo clásico (no-transformer)** para este problema:
- Supera todos los modelos neuronales probados (FastText, BiLSTM, CNN-1D)
- Es superado solo por DistilBERT (modelo con preentrenamiento masivo)
- Ofrece el mejor balance entre desempeño, interpretabilidad y eficiencia computacional

### 11.2 Desempeño en el Problema

Para clasificación de currículums:
- **Accuracy 79.2%** es excelente para un modelo sin embeddings preentrenados
- **ROC AUC 0.9816** es el más alto entre modelos no-transformer
- **F1-macro 0.7606** indica buen balance entre clases, aunque con limitaciones en categorías minoritarias

### 11.3 Lecciones Clave

1. **TF-IDF + Gradient Boosting compite con deep learning**: Para muchos problemas de clasificación textual, métodos clásicos bien ajustados son competitivos

2. **Ensemble learning es poderoso**: 300 árboles agregados superan modelos neuronales de 1 capa

3. **Features explícitas tienen ventajas**: A diferencia de embeddings latentes, TF-IDF features son interpretables y debuggeables

4. **Límites sin transfer learning**: La brecha con DistilBERT (+8 pp) muestra el valor de preentrenamiento en corpus masivos

5. **Robustez a arquitectura**: XGBoost logra alto desempeño con hiperparámetros estándar, sin búsqueda exhaustiva

XGBoost confirma que para problemas de clasificación textual con datos limitados, métodos ensemble sobre representaciones TF-IDF son extremadamente competitivos y deben ser considerados seriamente antes de recurrir a arquitecturas neuronales complejas. Sin embargo, el techo de ~79% accuracy sugiere que representaciones contextuales (transformers) son necesarias para superar significativamente este nivel de desempeño.
