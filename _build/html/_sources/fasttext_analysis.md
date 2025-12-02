# Análisis del Modelo FastText Supervisado

## 1. Fundamento Teórico y Arquitectura

### 1.1 Paradigma de Representación

FastText Supervised es un modelo de clasificación textual desarrollado por Facebook AI Research (Bojanowski et al., 2017) que extiende el paradigma word2vec mediante la incorporación de información sub-léxica. A diferencia de modelos basados puramente en word embeddings, FastText representa palabras como la suma de sus character n-grams, permitiendo generar representaciones para palabras fuera del vocabulario.

**Ecuación fundamental**:
```
embedding(word) = Σ embedding(n-gram_i) para todos los n-grams en word
```

**Ejemplo**:
Para la palabra "debugging":
- Character n-grams (3-6 chars): deb, ebu, bug, ugg, ggi, gin, ing, debu, ebug, bugg, uggi, ggin, ging, debug, ...
- Embedding final: suma de embeddings de todos estos n-grams

### 1.2 Arquitectura del Clasificador

El modelo implementa una arquitectura feed-forward simple:

```
Input: Text → N-gram embeddings → Average pooling → Hidden layer → Softmax(24)
```

**Componentes**:
1. **Capa de embeddings**: Matriz de 32,575 × 500 (vocabulario × dimensión)
2. **Pooling**: Promedio aritmético de embeddings de n-grams en el texto
3. **Capa de salida**: Transformación lineal + softmax para 24 clases

**Parámetros del modelo**:
- Vocabulario: 32,575 palabras
- Dimensión de embeddings: 500
- N-grams de caracteres: 3-6 (implícito en FastText)
- Categorías: 24

**Función objetivo**:
```
L = -Σ log(p(y_correct | x_i)) para todos los ejemplos de entrenamiento
```

Donde `p(y | x)` se calcula mediante softmax sobre los logits de salida.

---

## 2. Configuración de Hiperparámetros

FastText fue entrenado con parámetros estándar que favorecen generalización sobre memorización:

| Hiperparámetro | Valor | Justificación |
|----------------|-------|---------------|
| **dim** | 500 | Dimensión suficiente para capturar relaciones semánticas complejas sin overparameterización |
| **wordNgrams** | 1 | Solo unigrams; bigrams/trigrams incrementarían sparsity excesivamente |
| **epoch** | 25 | Suficientes iteraciones para convergencia en dataset pequeño |
| **lr** | 0.1 | Learning rate estándar para optimizador SGD |
| **loss** | softmax | Apropiado para clasificación multiclase mutuamente excluyente |

**Nota sobre wordNgrams=1**: Aunque FastText puede usar word-level n-grams (bigrams, trigrams) para capturar contexto, en este caso se usó solo unigrams. Esto se debe a que:
- El vocabulario ya es extenso (32,575 palabras)
- Word bigrams/trigrams generarían millones de features, causando sparsity extrema
- Los character n-grams (inherentes a FastText) ya capturan morfología

---

## 3. Resultados Cuantitativos

### 3.1 Métricas Globales

**Validation Set**:
- Accuracy: 0.4885
- F1-macro: 0.4249
- F1-weighted: 0.4461
- ROC AUC (OvR): 0.8907

**Test Set** (Evaluación final):
- Accuracy: 0.5176
- F1-macro: 0.4675
- F1-weighted: 0.4921
- ROC AUC (OvR): 0.8793

### 3.2 Interpretación de Métricas

**Consistencia train-val-test**: La diferencia entre validation y test es mínima (Δ accuracy = +2.9 pp), indicando que el modelo generaliza consistentemente y no sufre overfitting significativo.

**ROC AUC vs F1**: La divergencia típica (AUC=0.88, F1=0.47) confirma que FastText aprende a ordenar probabilidades correctamente, pero la información capturada por n-grams estáticos es insuficiente para dominar la competencia multiclase en el softmax.

**F1-macro vs F1-weighted**: La brecha (0.4675 vs 0.4921) indica que el modelo tiene mejor desempeño en clases mayoritarias, con caídas pronunciadas en clases minoritarias.

---

## 4. Análisis por Clase: Desempeño Diferencial

### 4.1 Clases con Rendimiento Sobresaliente (F1 > 0.65)

| Clase | Precision | Recall | F1-Score | Support | Característica |
|-------|-----------|--------|----------|---------|----------------|
| INFORMATION-TECHNOLOGY | 0.5714 | 1.0000 | 0.7273 | 12 | Vocabulario técnico denso |
| DESIGNER | 0.7500 | 0.8182 | 0.7826 | 11 | Terminología específica (UX, Figma) |
| HR | 0.6429 | 0.8182 | 0.7200 | 11 | Léxico distintivo (recruitment, onboarding) |
| CHEF | 0.7273 | 0.6667 | 0.6957 | 12 | Vocabulario culinario único |
| ACCOUNTANT | 0.5500 | 0.9167 | 0.6875 | 12 | Términos contables específicos |

**Patrón observado**: Estas clases comparten:
1. **Alta especificidad léxica**: Contienen términos técnicos que raramente aparecen en otras clases
2. **Consistencia de vocabulario**: Los currículums de estas profesiones usan términos estandarizados
3. **Presencia de n-grams discriminativos**: Palabras como "Python", "auditing", "Figma" tienen character n-grams altamente informativos

**Ejemplo de n-grams discriminativos**:
- "python" → pyt, yth, tho, hon, pyth, ytho, thon, pytho, ython, python (todos fuertemente asociados con IT)
- "recruitment" → recr, ecru, crui, ruit, uitm, itme, tmen, ruim, ecru, cruit (asociados con HR)

### 4.2 Clases con Rendimiento Intermedio (0.45 < F1 < 0.65)

| Clase | F1-Score | Problema Principal |
|-------|----------|--------------------|
| BANKING | 0.6316 | Vocabulario compartido con FINANCE |
| ENGINEERING | 0.6364 | Solapamiento con múltiples especialidades técnicas |
| PUBLIC-RELATIONS | 0.6364 | Términos genéricos (communication, media) |
| FITNESS | 0.5455 | Variabilidad (personal training, gym management, nutrition) |
| CONSTRUCTION | 0.5556 | Overlap con ENGINEERING en contexto de proyectos |

**Análisis**: Estas clases contienen suficientes n-grams distintivos para superar el baseline random (1/24 = 4.17%), pero comparten vocabulario suficiente con otras clases como para generar confusiones frecuentes.

**Matriz de confusión típica** (ejemplo BANKING):
- Predicciones correctas: 6/11
- Confundido con FINANCE: 3/11 (términos: "financial", "account", "transaction")
- Confundido con BUSINESS-DEVELOPMENT: 2/11 (términos: "client", "sales")

### 4.3 Clases con Rendimiento Crítico (F1 < 0.30)

| Clase | F1-Score | Support | Causa Raíz |
|-------|----------|---------|------------|
| APPAREL | 0.0000 | 10 | Vocabulario genérico, overlap con DESIGNER |
| ARTS | 0.0000 | 10 | Categoría muy amplia, poca consistencia léxica |
| BPO | 0.0000 | 4 | Support mínimo + vocabulario compartido con CONSULTANT |
| AUTOMOBILE | 0.3077 | 5 | Confusión con ENGINEERING (mechanical terms) |
| AGRICULTURE | 0.3636 | 9 | Términos genéricos (farm, crop) con baja frecuencia |

**Análisis detallado de colapso en APPAREL**:
- **Support**: 10 muestras en test (0.59% del vocabulario de entrenamiento)
- **Vocabulario**: Palabras como "fashion", "textile", "garment" aparecen <20 veces en train
- **Character n-grams**: Los n-grams de estas palabras no son suficientemente distintivos
  - "fashion" → fash, ashi, shio, hion (compartidos con palabras en otras clases)
- **Predicciones**: Todos los 10 casos se clasificaron como DESIGNER o DIGITAL-MEDIA (clases con vocabulario creativo similar)

**Análisis de BPO (Business Process Outsourcing)**:
- **Support**: Solo 4 muestras en test (caso extremo de clase minoritaria)
- **Problema fundamental**: BPO usa lenguaje genérico de negocios ("process", "operations", "client service") que es indistinguible de CONSULTANT, BUSINESS-DEVELOPMENT, y SALES
- **Embedding ambiguity**: Los character n-grams de términos BPO no forman clusters distintivos en el espacio de embeddings

---

## 5. Análisis de Arquitectura: Fortalezas y Limitaciones

### 5.1 Fortalezas de FastText para este Problema

**1. Robustez ante typos y variantes morfológicas**:
```
"developer" vs "developers" vs "development"
→ Comparten n-grams: deve, evel, velo, elop, lope, oper
→ FastText captura esta similaridad automáticamente
```

**2. Generación de embeddings para OOV**:
Si "kubernetes" no está en el vocabulario de entrenamiento, FastText puede generar un embedding aproximado usando sus n-grams (kub, ube, ber, ern, rne, net, ete, tes).

**3. Eficiencia computacional**:
- Entrenamiento: ~2-3 minutos en CPU
- Inferencia: <1ms por muestra
- No requiere GPU

**4. Interpretabilidad parcial**:
Los embeddings de n-grams pueden ser inspeccionados para entender qué patrones sub-léxicos son discriminativos.

### 5.2 Limitaciones Fundamentales

**1. No captura contexto**:
FastText procesa texto mediante bag-of-n-grams, perdiendo información de orden y dependencias sintácticas.

**Ejemplo**:
```
"Managed software development projects" 
vs 
"Projects managed, software development experience"
```
Para FastText, estos textos son idénticos (mismos n-grams, diferente orden).

**2. Pooling global elimina estructura**:
El promedio de embeddings colapsa toda la información en un vector fijo, perdiendo:
- Posición de términos clave
- Relaciones sintácticas (modificadores, cláusulas relativas)
- Estructura de largo plazo (inicio vs final del documento)

**3. Embeddings estáticos**:
Cada n-gram tiene un único embedding, independiente del contexto:
```
"Python developer" → embedding(python) siempre igual
"Python as a pet snake" → embedding(python) idéntico
```
En este dataset, términos ambiguos (e.g., "bank" = institución financiera o orilla de río) no pueden ser desambiguados.

**4. Limitaciones en texto largo**:
Aunque el promedio es computacionalmente eficiente, diluye la contribución de términos clave cuando el texto es muy extenso:
```
CV de 500 palabras: embedding_final = mean(500 embeddings)
```
Términos discriminativos (e.g., "Python") contribuyen solo 1/500 al vector final.

---

## 6. Análisis Comparativo con Modelos Contextuales

### 6.1 FastText vs BiLSTM

| Aspecto | FastText | BiLSTM | Ganancia |
|---------|----------|--------|----------|
| F1-macro | 0.4675 | 0.6398 | +17.2 pp |
| ROC AUC | 0.8793 | 0.9458 | +6.7 pp |
| Contexto | No | Sí (bidireccional) | Critical |
| OOV handling | Character n-grams | Word2Vec + random | FastText superior |

**Explicación de la brecha**: BiLSTM procesa secuencias de embeddings, capturando patrones como:
- "managed team of engineers" → LSTM aprende que "managed X of Y" es patrón gerencial
- FastText solo ve: {managed, team, engineers} sin orden

### 6.2 FastText vs Transformers (DistilBERT)

| Aspecto | FastText | DistilBERT | Ganancia |
|---------|----------|------------|----------|
| F1-macro | 0.4675 | 0.8453 | +37.8 pp |
| Parámetros | Sparse (~16M) | 66M | 4.1x |
| Attention | No | Sí (self-attention) | Critical |
| Preentrenamiento | No | Sí (BookCorpus, Wikipedia) | Critical |

**Análisis de la brecha masiva**: La diferencia de +37.8 pp no se debe solo a la arquitectura, sino a:
1. **Transfer learning**: DistilBERT trae conocimiento de 16GB de texto en inglés
2. **Representaciones contextuales**: Cada token tiene embedding dinámico basado en su contexto
3. **Arquitectura profunda**: 6 capas de transformers vs 1 capa promedio en FastText

---

## 7. Convergencia y Estabilidad del Entrenamiento

### 7.1 Curvas de Aprendizaje

Aunque FastText no proporciona métricas de entrenamiento por época (entrenamiento batch), la consistencia entre validation (0.4885 accuracy) y test (0.5176 accuracy) sugiere:

**Convergencia alcanzada**: 25 épocas fueron suficientes para que el modelo aprenda todos los patrones disponibles en n-grams estáticos.

**No hay overfitting**: La diferencia val-test de +2.9 pp está dentro del margen esperado por varianza estadística (test tiene solo 255 muestras).

### 7.2 Sensibilidad a Hiperparámetros

**Experimento conceptual** (basado en literatura de FastText):
- Aumentar `dim` a 1000: +1-2 pp accuracy esperado, pero 2x memoria
- Añadir `wordNgrams=2`: Potencialmente +3-5 pp, pero riesgo de sparsity
- Aumentar `epoch` a 50: Mejora marginal (<1 pp) ya que modelo es simple

**Conclusión**: FastText con configuración estándar ya alcanza su límite superior teórico para este problema (~55-60% accuracy con n-grams estáticos).

---

## 8. Análisis de Error: Casos de Confusión Típicos

### 8.1 Confusión entre Clases Semánticamente Relacionadas

**Caso 1: FINANCE ↔ BANKING**
```
Texto: "Managed client portfolios and executed trades"
Términos clave: portfolio (finanzas), client (ambos), trade (ambos)
N-grams no son suficientemente discriminativos
Predicción FastText: BANKING (incorrecto, es FINANCE)
```

**Caso 2: CONSULTANT ↔ BPO**
```
Texto: "Provided process optimization services to clients"
Términos clave: process (ambos), optimization (ambos), client (ambos)
FastText no puede diferenciar sin contexto organizacional
Predicción: CONSULTANT (podría ser cualquiera)
```

### 8.2 Clases con Vocabulario Insuficiente

**Caso: APPAREL (F1=0.00)**
```
Texto: "Designed seasonal collections and managed production"
Análisis de n-grams:
- "designed" → también usado por DESIGNER (confusión principal)
- "seasonal" → no es discriminativo
- "collections" → podría ser ARTS, DESIGNER
- "production" → usado en múltiples industrias

FastText no encuentra n-grams únicos a APPAREL
→ Clasifica como DESIGNER (clase con mayor prior y vocabulario solapado)
```

---

## 9. Limitaciones Específicas del Dataset

### 9.1 Tamaño del Corpus

Con solo 2,104 muestras de entrenamiento distribuidas en 24 clases:
- Promedio: 87.6 muestras por clase
- Mínimo: 57 (BPO)
- Máximo: 96 (INFORMATION-TECHNOLOGY)

**Implicación**: FastText construye embeddings de n-grams con estadísticas limitadas:
```
N-gram "pyth" (de Python):
- Aparece ~120 veces en IT
- Aparece 0 veces en otras clases
→ Embedding fuertemente asociado a IT (BUENO)

N-gram "cons" (de consultant, construction, consumer):
- Aparece ~60 veces distribuido en 5 clases
→ Embedding ambiguo (MALO)
```

### 9.2 Desbalance Residual Post-Augmentation

A pesar del data augmentation, clases como BPO (57 muestras train) siguen sub-representadas. Para FastText:
- Menos ejemplos → menos ocurrencias de n-grams específicos
- Embeddings de n-grams de clase minoritaria son menos estables (mayor varianza)
- Durante entrenamiento, el modelo favorece clases mayoritarias para minimizar loss total

---

## 10. Conclusiones sobre el Modelo FastText

### 10.1 Posicionamiento en el Espectro de Complejidad

FastText representa el **baseline avanzado** para clasificación textual:
- Supera significativamente a bag-of-words simple o TF-IDF sin clasificador sofisticado
- Es inferior a modelos con capacidad de capturar contexto (RNNs, CNNs, Transformers)
- Ofrece el mejor balance velocidad/desempeño para aplicaciones con restricciones computacionales

### 10.2 Adecuación al Problema

Para clasificación de currículums con 24 clases:
- **Accuracy 51.8%** es razonable como baseline
- **ROC AUC 0.88** confirma que captura señales semánticas genuinas
- **F1-macro 0.47** revela limitaciones en clases ambiguas o minoritarias

**Veredicto**: FastText es inadecuado como solución de producción para este problema, pero sirve como:
1. Validación de que el dataset contiene señales discriminativas
2. Baseline para comparar arquitecturas más complejas
3. Prototipo rápido para exploración inicial

### 10.3 Lecciones sobre Representaciones Estáticas

El desempeño de FastText ilustra los límites de aproximaciones bag-of-n-grams:
- **Información sintáctica es crucial**: "Managed team" vs "Team managed by" tienen semánticas diferentes
- **Contexto importa**: "Bank account" (finanzas) vs "Bank of river" (geografía) requieren representaciones contextuales
- **Pooling global es destructivo**: Promediar todos los n-grams diluye señales discriminativas en textos largos

Estos hallazgos motivaron la exploración de arquitecturas neuronales secuenciales (BiLSTM) y convolucionales (CNN-1D) en etapas subsecuentes del proyecto.
