# Análisis General del Sistema de Clasificación Multiclase de Currículums

## 1. Contexto del Problema

El presente trabajo aborda un problema de clasificación multiclase en el dominio del procesamiento de lenguaje natural (NLP), específicamente la categorización automática de currículums vitae en 24 categorías profesionales distintas. El dataset original contiene 2,484 muestras textuales correspondientes a descripciones profesionales, con una distribución moderadamente desbalanceada entre clases.

### 1.1 Características del Problema

- **Dominio**: Clasificación de texto en lenguaje natural
- **Número de clases**: 24 categorías profesionales
- **Tamaño del corpus**: 2,484 muestras originales, expandidas a 2,621 mediante data augmentation
- **Tipo de tarea**: Clasificación multiclase (no multilabel)
- **Desbalance**: Ratio máximo de 1.82x entre clase mayoritaria y minoritaria post-augmentation

Las categorías incluyen perfiles profesionales diversos como INFORMATION-TECHNOLOGY, HEALTHCARE, BPO, AUTOMOBILE, DESIGNER, entre otros, lo que representa un espacio semántico heterogéneo con diferentes niveles de especificidad léxica.

---

## 2. Divergencia entre ROC AUC y F1-Score: Fundamento Teórico

Uno de los hallazgos consistentes a través de todos los modelos evaluados es la notable divergencia entre el ROC AUC (típicamente >0.88) y el F1-Score macro (típicamente 0.46-0.87). Este fenómeno no representa una contradicción, sino que refleja diferencias fundamentales en lo que cada métrica mide.

### 2.1 ROC AUC: Discriminación Probabilística

El ROC AUC (Area Under the Receiver Operating Characteristic Curve) en configuración One-vs-Rest (OvR) para problemas multiclase evalúa la capacidad del modelo para **ordenar correctamente las probabilidades**. Específicamente:

- Mide si la probabilidad asignada a la clase correcta es, en promedio, mayor que las probabilidades asignadas a las clases incorrectas
- **No requiere** que la probabilidad correcta sea la máxima absoluta, solo que sea consistentemente mayor
- Es robusta ante desbalances de clase y diferencias en la confianza del modelo

**Ejemplo ilustrativo**:

Para una instancia de la clase AGRICULTURE:
- Probabilidad verdadera: 0.18
- Probabilidades falsas: 0.16, 0.14, 0.12, 0.10, ...

Este patrón contribuye positivamente al ROC AUC (la clase correcta tiene mayor probabilidad que las incorrectas), pero resulta en una predicción incorrecta bajo `argmax`, ya que otra clase podría tener probabilidad 0.19.

### 2.2 F1-Score: Decisiones Discretas

El F1-Score evalúa el rendimiento después de la **decisión discreta** (típicamente vía `argmax` en el vector de probabilidades softmax). Esta métrica:

- Requiere que la clase correcta tenga la probabilidad más alta para considerarse un acierto
- Es sensible a la calibración del modelo
- Penaliza fuertemente los errores en clases minoritarias (en su variante macro)

### 2.3 Implicaciones para Problemas Multiclase Desbalanceados

En contextos de 24 clases con desbalance moderado, esta divergencia se manifiesta especialmente en:

1. **Clases con vocabulario ambiguo**: El modelo aprende patrones semánticos correctos (alto AUC) pero no logra la suficiente confianza para dominar el softmax (bajo F1)

2. **Clases minoritarias**: Incluso con buenos embeddings, la falta de ejemplos impide que el modelo alcance probabilidades suficientemente altas para ganar la competencia multiclase

3. **Solapamiento semántico**: Clases como CONSULTANT vs BPO, o ARTS vs DESIGNER comparten vocabulario, resultando en distribuciones de probabilidad difusas

### 2.4 Interpretación para Este Trabajo

Los valores de ROC AUC consistentemente altos (0.88-0.98) a través de todos los modelos indican que:

- Las arquitecturas neuronales y estadísticas empleadas **sí capturan la estructura subyacente** del problema
- Los embeddings (Word2Vec, fastText, DistilBERT) codifican información semántica relevante
- El problema radica en la **conversión de conocimiento probabilístico a decisiones discretas**

Por tanto, modelos con AUC >0.90 pero F1 <0.70 no están "fallando", sino que enfrentan limitaciones inherentes al espacio de decisión multiclase con información ambigua.

---

## 3. Comparativa Global de Arquitecturas

La siguiente tabla resume el desempeño de los cinco modelos evaluados:

| Modelo | Accuracy | F1-Macro | F1-Weighted | ROC AUC (OvR) | Parámetros |
|--------|----------|----------|-------------|---------------|------------|
| **FastText** | 0.5176 | 0.4675 | 0.4921 | 0.8793 | N/A (n-grams) |
| **BiLSTM** | 0.6941 | 0.6398 | 0.6775 | 0.9458 | ~500K |
| **CNN-1D** | 0.7333 | 0.6716 | 0.7112 | 0.9633 | 10.3M |
| **XGBoost** | 0.7922 | 0.7606 | 0.7849 | 0.9816 | N/A (300 trees) |
| **DistilBERT** | **0.8745** | **0.8453** | **0.8718** | **0.9756** | 66M |

### 3.1 Observaciones Clave

**Jerarquía de complejidad vs desempeño**: Se observa una correlación clara entre la sofisticación arquitectural y el rendimiento, aunque con rendimientos decrecientes:

- FastText (baseline) → BiLSTM: +17.6 pp en accuracy
- BiLSTM → CNN-1D: +3.9 pp en accuracy  
- CNN-1D → XGBoost: +5.9 pp en accuracy
- XGBoost → DistilBERT: +8.2 pp en accuracy

**Brecha de representación contextual**: Los modelos basados en arquitecturas que capturan contexto (BiLSTM, CNN-1D, DistilBERT) superan significativamente a métodos basados en n-grams estáticos (FastText). La excepción notable es XGBoost, cuyo rendimiento superior se atribuye a:
- Representaciones TF-IDF de alta dimensionalidad (10,000 features, n-grams 1-3)
- Capacidad de modelar interacciones no lineales complejas entre features
- Robustez ante datos tabulares de alta dimensión

**Transferencia de conocimiento**: DistilBERT, al aprovechar preentrenamiento en corpus masivos, logra la mayor ventaja absoluta (+8.2 pp sobre XGBoost), confirmando la hipótesis de que representaciones preentrenadas son superiores para tareas de clasificación textual con datos limitados.

---

## 4. Patrones Transversales en el Comportamiento por Clase

### 4.1 Clases Consistentemente Robustas

Las siguientes categorías exhiben F1-Score >0.80 en al menos 4 de los 5 modelos:

- **ACCOUNTANT**: Vocabulario altamente técnico (audit, ledger, payroll, reconciliation)
- **HR**: Terminología específica (recruitment, onboarding, HRIS, benefits)
- **INFORMATION-TECHNOLOGY**: Densidad de términos técnicos (Java, SQL, debugging, deployment)
- **DESIGNER**: Léxico distintivo (UX, Figma, wireframe, prototyping)
- **BUSINESS-DEVELOPMENT**: Patrones semánticos claros (pipeline, lead generation, revenue)

**Hipótesis**: Estas clases combinan alta especificidad léxica con suficiente representación en el dataset, permitiendo que incluso modelos simples (FastText) logren F1 >0.60.

### 4.2 Clases Consistentemente Problemáticas

Las siguientes categorías exhiben F1-Score <0.50 en al menos 3 de los 5 modelos:

- **BPO** (Business Process Outsourcing): n=57 en train, vocabulario genérico solapado con CONSULTANT y SALES
- **AUTOMOBILE**: n=83 en train, solapamiento semántico con ENGINEERING (mechanical systems)
- **APPAREL**: n=77 en train, lenguaje compartido con DESIGNER y ARTS
- **ARTS**: n=83 en train, alta variabilidad interna (visual arts, performing arts, etc.)

**Análisis**: El bajo desempeño en estas clases no se debe únicamente al tamaño muestral (el augmentation llevó la mayoría al ~80% del máximo), sino a:

1. **Ambigüedad semántica estructural**: El vocabulario es inherentemente polisémico
2. **Granularidad subóptima**: Algunas categorías podrían beneficiarse de sub-categorización (e.g., ARTS → VISUAL-ARTS, PERFORMING-ARTS)
3. **Límites difusos**: Las fronteras entre estas clases y otras no están bien definidas en el espacio semántico

### 4.3 Impacto del Data Augmentation

El proceso de back-translation logró expandir el dataset de 2,484 a 2,621 muestras (+5.5%), llevando las clases minoritarias a un máximo del 80% de la clase mayoritaria. Sin embargo, el análisis por clase revela que:

- **Clases con vocabulario técnico específico**: Beneficio marginal del augmentation, ya que la semántica distintiva se preserva incluso con pocas muestras
- **Clases con lenguaje genérico**: El augmentation no resuelve la ambigüedad fundamental, solo aumenta la representación de patrones ambiguos

Esto explica por qué BPO, a pesar de recibir augmentation significativo, mantiene F1 <0.40 en la mayoría de modelos.

---

## 5. Análisis de Convergencia y Generalización

### 5.1 Evidencia de Overfitting Controlado

Todos los modelos neuronales (BiLSTM, CNN-1D, DistilBERT) exhiben diferencias esperables entre train y validation loss, sin colapso de generalización:

- **BiLSTM**: 17 épocas hasta early stopping, ROC AUC validation estabilizado en 0.9512
- **CNN-1D**: 13 épocas hasta early stopping, validation metrics mejoraron consistentemente hasta época 10
- **DistilBERT**: 4 épocas con mejoras continuas en validation (0.64 → 0.83 F1-macro)

La ausencia de overfitting severo sugiere que:
- Los mecanismos de regularización (dropout 0.5-0.8, weight decay) son efectivos
- El tamaño del dataset, aunque limitado, es suficiente para estas arquitecturas
- El augmentation introduce variabilidad genuina que previene memorización

### 5.2 Curvas de Aprendizaje

**CNN-1D**: Exhibe la curva de aprendizaje más ilustrativa:
- Épocas 1-5: Mejora rápida en train y validation (accuracy 0.14 → 0.64)
- Épocas 6-10: Convergencia progresiva (validation accuracy 0.64 → 0.75)
- Épocas 11-13: Estabilización, mejoras marginales (<1 pp)

Este patrón confirma que el modelo extrae el máximo conocimiento disponible en los datos antes de que early stopping termine el entrenamiento.

---

## 6. Cobertura de Embeddings y su Impacto

Un factor técnico relevante es la cobertura de vocabulario en embeddings preentrenados:

- **Word2Vec local (BiLSTM, CNN-1D)**: 54.5% de cobertura (17,758/32,604 palabras)
- **FastText**: Cobertura implícitamente mayor por embeddings basados en subword
- **DistilBERT**: Cobertura ~100% gracias a tokenización BPE (Byte Pair Encoding)

### 6.1 Implicaciones

La cobertura del 54.5% en Word2Vec significa que **casi la mitad de las palabras** se representan como vectores aleatorios inicializados. A pesar de esto:

- BiLSTM y CNN-1D logran ROC AUC >0.94, indicando que las palabras cubiertas son suficientemente informativas
- DistilBERT, con cobertura total, obtiene una ventaja de +14 pp en F1-macro sobre CNN-1D

Esto sugiere que el vocabulario OOV (Out Of Vocabulary) contiene información semántica relevante que solo modelos con tokenización subword (DistilBERT) pueden aprovechar completamente.

---

## 7. Conclusiones Generales

El sistema de clasificación multiclase de currículums presenta un caso de estudio representativo de los desafíos en NLP con datos limitados:

1. **Arquitecturas modernas son superiores pero no milagrosas**: DistilBERT (87.5% accuracy) supera significativamente a FastText (51.8%), pero ningún modelo resuelve completamente las clases ambiguas

2. **El problema tiene límites inherentes**: La divergencia ROC AUC vs F1 indica que el desafío no es solo arquitectural, sino semántico—algunas clases son genuinamente difíciles de distinguir

3. **Data augmentation es útil pero no suficiente**: El back-translation preserva semántica (0.75 similarity threshold) pero no puede crear información que no existe en los textos originales

4. **XGBoost como alternativa competitiva**: Con representaciones TF-IDF adecuadas, métodos clásicos pueden competir con redes neuronales más complejas (79.2% accuracy), ofreciendo ventajas en interpretabilidad y velocidad

5. **Transferencia de conocimiento es crucial**: El salto de XGBoost (79.2%) a DistilBERT (87.5%) confirma que preentrenamiento en corpus masivos aporta conocimiento semántico que datasets pequeños no pueden proporcionar por sí solos

El trabajo evidencia que la selección de modelo debe balancear complejidad arquitectural, recursos computacionales, interpretabilidad y los límites inherentes de los datos disponibles.
