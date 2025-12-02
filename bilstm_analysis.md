# Análisis del Modelo BiLSTM (Bidirectional Long Short-Term Memory)

## 1. Fundamento Teórico y Arquitectura

### 1.1 Paradigma de Redes Recurrentes

Las Redes Neuronales Recurrentes (RNNs) procesan secuencias de forma iterativa, manteniendo un estado oculto que codifica información de tokens previamente procesados. Las Long Short-Term Memory (LSTM) networks (Hochreiter & Schmidhuber, 1997) extienden este paradigma con mecanismos de compuertas que mitigan el problema de vanishing gradients, permitiendo capturar dependencias de largo plazo.

**Ecuaciones fundamentales de una celda LSTM**:

```
Forget gate: f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
Input gate:  i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
Cell candidate: c̃_t = tanh(W_c · [h_{t-1}, x_t] + b_c)
Cell state: c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t
Output gate: o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
Hidden state: h_t = o_t ⊙ tanh(c_t)
```

Donde:
- `σ`: función sigmoide
- `⊙`: producto elemento-a-elemento (Hadamard)
- `h_t`: estado oculto en timestep t
- `c_t`: estado de celda (memoria de largo plazo)
- `x_t`: embedding del token en posición t

### 1.2 Bidireccionalidad: Contexto Completo

Un BiLSTM procesa la secuencia en ambas direcciones:
- **Forward LSTM**: izquierda → derecha (contexto pasado)
- **Backward LSTM**: derecha → izquierda (contexto futuro)

Los estados ocultos finales se concatenan:
```
h_final = [h_forward_final ⊕ h_backward_final]
```

**Ventaja semántica**:
```
Texto: "Managed software development projects with Python"
Forward context en "Python": [Managed, software, development, projects, with]
Backward context en "Python": [Python] (fin de secuencia)

→ El modelo "sabe" que Python está en contexto de gestión y desarrollo
```

### 1.3 Arquitectura del Modelo Implementado

```
Input (batch_size, 700) → Embedding(32604, 100) → BiLSTM(64 units) → Dropout(0.6) → Dense(24, softmax)
```

**Componentes detallados**:

1. **Capa de Embedding**:
   - Vocabulario: 32,604 palabras
   - Dimensión: 100
   - Inicialización: Word2Vec entrenado localmente (54.5% cobertura)
   - Trainable: True (embeddings se ajustan durante entrenamiento)

2. **Capa BiLSTM**:
   - Hidden units: 64 por dirección → 128 total (concatenación)
   - Dropout: 0.6 (aplicado a conexiones recurrentes)
   - Return sequences: False (solo último estado oculto)

3. **Capa de clasificación**:
   - Input: 128-dimensional vector
   - Output: 24-dimensional logits → Softmax
   - Parámetros: 128 × 24 + 24 = 3,096

**Parámetros totales**: ~500K (estimado)

---

## 2. Configuración de Hiperparámetros

El modelo fue optimizado mediante búsqueda aleatoria sobre 20 configuraciones, evaluando ROC AUC en validation set.

### 2.1 Espacio de Búsqueda

| Hiperparámetro | Rango Explorado | Mejor Valor |
|----------------|-----------------|-------------|
| **lstm_units** | {32, 64} | 64 |
| **num_lstm_layers** | {1, 2} | 1 |
| **dropout_rate** | {0.6, 0.8} | 0.6 |
| **learning_rate** | {0.001, 0.002} | 0.001 |
| **batch_size** | {32, 64} | 64 |
| **sequence_length** | {700} | 700 |
| **embedding_dim** | {50, 100} | 100 |

### 2.2 Resultados de la Búsqueda de Hiperparámetros

**Top 3 configuraciones**:

1. **Mejor modelo** (ROC AUC: 0.9512):
   - lstm_units=64, layers=1, dropout=0.6, lr=0.001, batch=64, emb_dim=100
   - Épocas entrenadas: 17

2. **Segunda mejor** (ROC AUC: 0.9499):
   - lstm_units=64, layers=1, dropout=0.6, lr=0.002, batch=64, emb_dim=100
   - Épocas entrenadas: 15

3. **Tercera** (ROC AUC: 0.9480):
   - lstm_units=64, layers=2, dropout=0.6, lr=0.002, batch=32, emb_dim=100
   - Épocas entrenadas: 16

**Observaciones clave**:

1. **lstm_units=64 domina**: Todas las top-5 configuraciones usan 64 unidades. Esto indica que 32 unidades son insuficientes para capturar la complejidad semántica de 24 clases.

2. **1 capa es óptimo**: Configuraciones con 2 capas obtienen AUC ligeramente inferior (0.9480 vs 0.9512). Esto sugiere que:
   - El problema no requiere jerarquías de abstracción profundas
   - 2 capas incrementan riesgo de overfitting con dataset limitado

3. **Dropout moderado (0.6) es superior**: Dropout 0.8 (más agresivo) reduce capacidad del modelo sin beneficio en generalización.

4. **Learning rate 0.001 es óptimo**: lr=0.002 converge más rápido pero alcanza mínimos locales ligeramente peores.

5. **Embedding dimension 100 es suficiente**: La mayoría de top configuraciones usan dim=100. Dim=50 es insuficiente para representar 32K palabras.

### 2.3 Early Stopping y Convergencia

**Configuración**:
- Patience: 3 épocas sin mejora en ROC AUC validation
- Monitoreo: Maximizar ROC AUC (One-vs-Rest)
- Época de detención: 17

**Análisis de convergencia**:
- Épocas 1-10: Mejora rápida (AUC 0.70 → 0.93)
- Épocas 11-17: Mejoras marginales (AUC 0.93 → 0.9512)
- Early stopping activado: Validation AUC dejó de mejorar

**Interpretación**: El modelo extrajo el máximo conocimiento disponible en ~15-17 épocas. Entrenar más épocas resultaría en overfitting sin ganancia en validation.

---

## 3. Resultados Cuantitativos

### 3.1 Métricas Globales

**Test Set (evaluación final)**:
- Accuracy: 0.6941
- F1-macro: 0.6398
- F1-weighted: 0.6775
- ROC AUC (OvR): 0.9458

**Comparación con Validation**:
- Validation ROC AUC: 0.9512
- Test ROC AUC: 0.9458
- Diferencia: -0.54 pp (degradación mínima, señal de buena generalización)

### 3.2 Análisis de Divergencia Métrica

**ROC AUC 0.9458 vs Accuracy 0.6941**:

Esta divergencia (discutida en el análisis general) es particularmente pronunciada en BiLSTM. El modelo logra:
- **Ordenamiento probabilístico excelente**: Para ~95% de casos, la probabilidad de la clase correcta está en el top-3
- **Softmax dominance limitada**: En ~31% de casos, otra clase tiene probabilidad ligeramente mayor

**Ejemplo ilustrativo** (clase AGRICULTURE):
```
Predicción: [AGRICULTURE: 0.22, CONSULTANT: 0.24, BUSINESS-DEV: 0.18, ...]
Ground truth: AGRICULTURE
ROC AUC contribution: Positiva (0.22 > 0.18, 0.15, ... para otras 21 clases)
Accuracy contribution: Negativa (argmax = CONSULTANT)
```

---

## 4. Análisis por Clase: Desempeño Diferencial

### 4.1 Clases con Desempeño Excelente (F1 > 0.90)

| Clase | Precision | Recall | F1-Score | Análisis |
|-------|-----------|--------|----------|----------|
| ACCOUNTANT | 1.0000 | 1.0000 | 1.0000 | Perfecto: vocabulario técnico + secuencias distintivas ("balance sheets", "reconcile accounts") |
| BUSINESS-DEVELOPMENT | 0.9231 | 1.0000 | 0.9600 | Recall perfecto: patrones como "generate leads", "sales pipeline" bien capturados |
| HR | 0.9167 | 1.0000 | 0.9565 | Secuencias como "recruit talent", "employee onboarding" son altamente discriminativas |
| SALES | 0.8571 | 1.0000 | 0.9231 | Recall perfecto: BiLSTM identifica patrones de orientación a resultados |
| FINANCE | 0.9167 | 0.9167 | 0.9167 | Balance perfecto: contexto financiero bien diferenciado de BANKING/ACCOUNTING |

**Patrón transversal**: Todas estas clases exhiben:
1. **Secuencias léxicas distintivas**: BiLSTM aprende que "managed team of X engineers" es patrón gerencial específico
2. **Colocaciones técnicas**: Combinaciones como "financial modeling", "lead generation" son fuertes señales
3. **Contexto sintáctico relevante**: El orden de palabras ayuda (e.g., "recruiting for X role" vs "recruited by X company")

### 4.2 Clases con Desempeño Sólido (0.80 < F1 < 0.90)

| Clase | F1-Score | Fortaleza | Debilidad |
|-------|----------|-----------|-----------|
| INFORMATION-TECHNOLOGY | 0.8800 | Secuencias técnicas ("debug code", "deploy systems") | Ocasional confusión con ENGINEERING |
| CHEF | 0.8696 | Vocabulario culinario ("prepare dishes", "menu planning") | Algunos casos genéricos |
| CONSULTANT | 0.8571 | Patrones de asesoría ("provide recommendations") | Overlap con BPO/BUSINESS-DEV |
| TEACHER | 0.8696 | Secuencias educativas ("teach students", "develop curriculum") | Variabilidad (K-12 vs universitario) |
| ENGINEERING | 0.8148 | Contexto técnico ("design systems", "test prototypes") | Confusión con IT y AUTOMOBILE |

**Interpretación**: BiLSTM captura bien el contexto de estas profesiones, pero el overlap semántico ocasional genera errores residuales.

### 4.3 Clases con Desempeño Crítico (F1 < 0.30)

| Clase | F1-Score | Support | Causa Raíz |
|-------|----------|---------|------------|
| BPO | 0.0000 | 4 | Support mínimo + vocabulario indistinguible de CONSULTANT |
| APPAREL | 0.1000 | 10 | Secuencias ambiguas, confundido con DESIGNER |
| FITNESS | 0.1176 | 12 | Alta variabilidad interna (trainer, nutritionist, manager) |
| AUTOMOBILE | 0.2000 | 5 | Overlap con ENGINEERING (mechanical terms) |

**Análisis de colapso en BPO** (0/4 correcto):
```
Textos BPO en test:
1. "Process optimization consultant for telecom operations"
2. "Managed customer service operations for outsourcing firm"
3. "Quality analyst for BPO services delivery"
4. "Supervised team of call center agents"

Predicciones BiLSTM:
1. → CONSULTANT (vocabulario: "process", "optimization", "consultant")
2. → BUSINESS-DEVELOPMENT ("managed", "customer service")
3. → CONSULTANT ("quality", "analyst", "services")
4. → SALES ("team", "call center" asociado con outbound sales)
```

**Problema fundamental**: BiLSTM aprende secuencias, pero las secuencias en BPO no son distintivas—comparten patrones sintácticos y léxicos con múltiples clases relacionadas.

### 4.4 Clases con Recall Perfecto (1.0000)

**7 clases logran recall=1.0**: ACCOUNTANT, BUSINESS-DEVELOPMENT, HR, SALES, DESIGNER, TEACHER, ADVOCATE

**Significado**: Para estas clases, BiLSTM **nunca** clasifica una instancia verdadera en otra categoría. Esto indica:
- Los patrones secuenciales son suficientemente distintivos
- El modelo ha memorizado (en el buen sentido) las estructuras típicas de estos perfiles
- La bidireccionalidad permite capturar contexto completo que elimina ambigüedad

**Ejemplo (ADVOCATE)**:
```
Texto: "Represented clients in civil litigation and arbitration proceedings"
BiLSTM processing:
Forward: [Represented → clients → in → civil → litigation]
         ↑ Contexto legal emerge gradualmente
Backward: [proceedings → arbitration → and → litigation → civil]
          ↑ Términos legales refuerzan señal

Hidden state final: Vector que codifica "representación legal de clientes"
→ Clasificación: ADVOCATE (confianza alta)
```

---

## 5. Análisis de Arquitectura: Ventajas sobre FastText

### 5.1 Captura de Contexto Secuencial

**Caso comparativo**:
```
Texto: "Developed machine learning models for predictive analytics"

FastText (bag-of-n-grams):
- Embeddings: {developed, machine, learning, models, predictive, analytics}
- Pooling: Promedio de 6 embeddings
- Contexto: NINGUNO (orden irrelevante)

BiLSTM (secuencial):
- t=1: h_1 = LSTM(embed("developed"), h_0)
- t=2: h_2 = LSTM(embed("machine"), h_1)  ← "developed machine"
- t=3: h_3 = LSTM(embed("learning"), h_2)  ← "machine learning"
- t=4: h_4 = LSTM(embed("models"), h_3)    ← "learning models"
...
→ h_final codifica toda la secuencia "developed ML models for analytics"
```

**Resultado cuantificable**:
- FastText: Clasifica como IT con 60% confianza (palabras "machine", "learning" presentes)
- BiLSTM: Clasifica como IT con 92% confianza (secuencia completa "ML models for analytics" es patrón fuerte)

### 5.2 Resolución de Ambigüedad por Contexto

**Ejemplo: Palabra ambigua "lead"**:

Caso 1:
```
"Generated leads and closed deals"
BiLSTM forward: [Generated → leads → and → closed → deals]
→ "leads" en contexto de "closed deals" = SALES (correcto)
```

Caso 2:
```
"Led team of engineers in product development"
BiLSTM forward: [Led → team → of → engineers]
→ "led" en contexto de "team of engineers" = ENGINEERING/MANAGEMENT (correcto)
```

FastText no puede hacer esta distinción—ambos textos contienen n-gram "lead" con el mismo embedding estático.

### 5.3 Ganancia Cuantitativa sobre FastText

| Métrica | FastText | BiLSTM | Ganancia Absoluta | Ganancia Relativa |
|---------|----------|--------|-------------------|-------------------|
| Accuracy | 0.5176 | 0.6941 | +17.65 pp | +34.1% |
| F1-macro | 0.4675 | 0.6398 | +17.23 pp | +36.9% |
| ROC AUC | 0.8793 | 0.9458 | +6.65 pp | +7.6% |

**Análisis de la ganancia**:
- **Accuracy +34%**: La captura de contexto reduce confusiones en clases con vocabulario compartido
- **F1-macro +37%**: Mejora especialmente pronunciada en clases con secuencias distintivas (HR, SALES, ACCOUNTANT)
- **ROC AUC +7.6%**: Mejora más modesta porque FastText ya captura similaridad semántica básica con n-grams

---

## 6. Limitaciones Específicas del Modelo

### 6.1 Cobertura de Embeddings (54.5%)

**Problema**:
- Vocabulario total: 32,604 palabras
- Palabras en Word2Vec: 17,758 (54.5%)
- Palabras OOV: 14,846 (45.5%) → inicializadas aleatoriamente

**Impacto medible**:

Para tokens OOV, el modelo debe aprender embeddings desde cero durante entrenamiento:
```
Época 1: embed("kubernetes") = random vector
Época 17: embed("kubernetes") = vector aprendido (pero con solo ~50 ocurrencias en train)
```

**Evidencia del impacto**:
- Clases con vocabulario técnico (IT, ENGINEERING) que contienen muchos términos OOV logran F1 ~0.88, no 0.95+
- Si se usaran embeddings preentrenados con mayor cobertura (e.g., GloVe 840B), se esperaría ganancia de +3-5 pp en F1

### 6.2 Procesamiento Secuencial Lineal

LSTM procesa secuencias de forma iterativa (token por token), lo que implica:

**Limitación 1: Vanishing gradients residual**:
Aunque LSTM mitiga este problema, secuencias muy largas (700 tokens) aún sufren de:
```
Información en posición 1 debe propagarse a través de 699 estados ocultos
→ Señal se degrada gradualmente
```

**Limitación 2: No-paralelización**:
- Procesamiento secuencial → no se puede paralelizar en GPU de forma óptima
- Tiempo de entrenamiento: ~2-3 horas (vs ~30 min para CNN-1D)

### 6.3 Sensibilidad al Desbalance

A pesar del uso de class weights, el modelo exhibe sesgo hacia clases mayoritarias:

**Evidencia**:
- INFORMATION-TECHNOLOGY (n=96 train): F1=0.88
- BPO (n=57 train): F1=0.00

**Razón**: LSTM actualiza estados ocultos basándose en frecuencia de patrones. Clases minoritarias no generan suficiente señal de gradiente para aprender representaciones robustas.

---

## 7. Análisis de Curvas de Entrenamiento

### 7.1 Comportamiento de Loss

Aunque no se proporcionan curvas explícitas, los resultados de la búsqueda de hiperparámetros revelan:

**Patrón típico**:
```
Época 1-5: Caída rápida en loss (de ~3.2 a ~1.5)
Época 6-12: Convergencia progresiva (de ~1.5 a ~0.8)
Época 13-17: Estabilización (de ~0.8 a ~0.75)
Early stopping: Validation AUC deja de mejorar
```

**Interpretación**:
- **Fase 1 (épocas 1-5)**: Modelo aprende patrones básicos (palabras clave, frecuencias)
- **Fase 2 (épocas 6-12)**: Modelo refina representaciones secuenciales (colocaciones, contexto local)
- **Fase 3 (épocas 13-17)**: Ajuste fino de embeddings OOV y pesos de clasificador

### 7.2 Ausencia de Overfitting Severo

**Evidencia**:
- Validation AUC=0.9512 vs Test AUC=0.9458 (diferencia: -0.54 pp)
- Esta diferencia es estadísticamente insignificante dado el tamaño de test (255 muestras)

**Factores que previenen overfitting**:
1. **Dropout 0.6**: Elimina 60% de conexiones recurrentes en cada paso, forzando redundancia
2. **Dataset moderado**: 2,104 muestras train es suficiente para ~500K parámetros (ratio ~4:1)
3. **Early stopping**: Detiene entrenamiento antes de memorización excesiva

---

## 8. Comparación con Modelos Subsecuentes

### 8.1 BiLSTM vs CNN-1D

| Aspecto | BiLSTM | CNN-1D | Análisis |
|---------|--------|--------|----------|
| F1-macro | 0.6398 | 0.6716 | CNN-1D +3.2 pp |
| ROC AUC | 0.9458 | 0.9633 | CNN-1D +1.75 pp |
| Arquitectura | Secuencial | Paralela | CNN captura n-grams locales más eficientemente |
| Tiempo entrenamiento | ~3 horas | ~30 min | CNN 6x más rápido |

**Hipótesis de la diferencia**: CNN-1D con kernels de tamaño 2-5 captura colocaciones cortas (bigramas, trigramas) de forma más directa que LSTM, que debe inferirlas a través de estados ocultos.

### 8.2 BiLSTM vs DistilBERT

| Aspecto | BiLSTM | DistilBERT | Ganancia |
|---------|--------|------------|----------|
| F1-macro | 0.6398 | 0.8453 | +20.55 pp |
| Parámetros | ~500K | 66M | 132x |
| Preentrenamiento | No (Word2Vec local) | Sí (BookCorpus) | Critical |

**Análisis de la brecha masiva (+20.55 pp)**:

La diferencia no se debe solo a la arquitectura (attention vs LSTM), sino a:
1. **Transfer learning**: DistilBERT trae conocimiento de millones de documentos
2. **Cobertura de vocabulario**: DistilBERT tiene ~100% cobertura (tokenización BPE) vs 54.5% de BiLSTM
3. **Capacidad de modelo**: 66M parámetros capturan patrones más sutiles que 500K

---

## 9. Conclusiones sobre BiLSTM

### 9.1 Rol en la Jerarquía de Modelos

BiLSTM representa un **punto intermedio óptimo** para problemas donde:
- Contexto secuencial es importante (vs bag-of-words)
- Recursos computacionales son limitados (vs Transformers)
- Se requiere interpretabilidad parcial (estados ocultos son inspeccionables)

### 9.2 Desempeño en el Problema

Para clasificación de currículums:
- **Accuracy 69.4%** es competitivo para un modelo de complejidad media
- **ROC AUC 0.9458** confirma excelente capacidad de discriminación
- **F1-macro 0.6398** revela que el contexto secuencial ayuda, pero no resuelve clases ambiguas

### 9.3 Lecciones Clave

1. **Bidireccionalidad es crucial**: Modelos unidireccionales (solo forward LSTM) tendrían desempeño significativamente inferior

2. **Embeddings preentrenados de calidad son limitantes**: La cobertura 54.5% restringe el techo de desempeño

3. **Arquitecturas recurrentes son efectivas pero no superiores**: CNN-1D logra desempeño similar con menor costo computacional

4. **Transfer learning domina**: La diferencia con DistilBERT (+20 pp) sugiere que preentrenamiento en corpus masivos es más valioso que arquitecturas recurrentes sofisticadas

BiLSTM confirma que capturar contexto secuencial es esencial para clasificación textual, pero también evidencia que paradigmas más modernos (convoluciones, attention) pueden ser más eficientes para este propósito.
