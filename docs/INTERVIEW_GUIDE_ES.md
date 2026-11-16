# Defender EvidenceForge en una entrevista senior

La idea del proyecto es separar dos preguntas: **¿recuperé evidencia relevante?** y **¿la respuesta está sustentada por esa evidencia?** EvidenceForge permite estudiar la primera con un dataset público y comparar estrategias de recuperación. La ruta opcional de respuesta añade otro problema de evaluación; sus capacidades no se deducen del score de retrieval.

Presenta únicamente funciones que existan en la versión publicada y resultados de artefactos ejecutados. El adaptador opcional de OpenAI no implica que se haya realizado una evaluación live. Las extensiones de este documento son propuestas, no experiencia implementada.

## Explicación de 45 segundos

> Construí y evalué [mi contribución verificable] sobre EvidenceForge, un laboratorio local de recuperación de evidencia científica. Comparé BM25, embeddings y fusión de rankings sobre el mismo corpus de SciFact. Separé desarrollo y evaluación, conservé métricas por consulta y analicé el costo en latencia. Además, distinguí la integridad de una cita de su capacidad para sustentar una afirmación. Mi hallazgo fue [resultado medido], y su límite principal es [limitación observada].

Sustituye los campos con trabajo propio. Si todavía estás aprendiendo del código, di que lo reprodujiste y analizaste. No atribuyas resultados a clientes ni inventes una mejora cuantitativa. Un resultado en el que BM25 gana también puede demostrar buen criterio experimental.

## Decisiones que debes poder explicar

| Pregunta de entrevista | Respuesta que conviene desarrollar con evidencia |
| --- | --- |
| ¿Por qué BM25 como baseline? | Ofrece una referencia lexical interpretable. Enseña casos donde los términos exactos importan y permite medir qué aporta realmente el componente neural. Muestra tokenización y configuración, no solo el nombre del algoritmo. |
| ¿Por qué dense retrieval? | Busca relaciones que el solapamiento de términos puede perder. Examina ejemplos de mejora y regresión: la similitud temática también puede recuperar un abstract que no responde al claim. |
| ¿Por qué RRF? | BM25 y similitud vectorial no comparten escala. RRF combina posiciones sin fingir que sus scores son comparables. La profundidad de candidatos y la constante siguen siendo decisiones del experimento. |
| ¿Por qué búsqueda exacta y no ANN? | Con 5.183 documentos, permite aislar calidad de representaciones sin sumar errores de aproximación. Para otra escala, compararía latencia, memoria y recall de ANN contra este resultado exacto. |
| ¿Cómo evitas filtrar etiquetas? | El retriever recibe corpus y consulta; las etiquetas quedan en el evaluador. Los parámetros se fijan antes de mirar test. Demuestra dónde se separan estas entradas en el código. |
| ¿Qué significa el intervalo bootstrap? | Variación sobre consultas de este conjunto. Para comparar métodos, se remuestrean las mismas consultas en ambos. No representa automáticamente incertidumbre sobre todos los dominios o sobre contaminación de pretraining. |
| ¿Qué verifica una cita exacta? | Que un fragmento está en un documento permitido del evidence pack. No verifica que ese fragmento implique la conclusión ni que una negación se haya interpretado correctamente. |
| ¿Por qué separar retrieval de answer generation? | Un error de recuperación y un error de síntesis requieren diagnósticos distintos. No necesitas pagar una llamada generativa para medir el ranking con qrels. |

## Tres límites que distinguen un análisis serio

**Relevancia no significa acuerdo.** En SciFact, un abstract puede ser evidencia relevante porque contradice la afirmación. Si una interfaz presenta todos los resultados como “pruebas de que el claim es cierto”, cambia indebidamente el significado de la tarea. Una similitud alta tampoco es una probabilidad de verdad.

**El test público no demuestra novedad para el modelo.** El test de BEIR usa el dev público original de SciFact. Mantenerlo intacto durante tu desarrollo evita una forma de tuning oportunista, pero no demuestra ausencia de ese material en pretraining. Describe el resultado como evaluación sobre ese split público, no como generalización garantizada a datos inéditos.

**Una cita literal puede acompañar una respuesta falsa.** El modelo puede seleccionar una frase real pero irrelevante, ignorar una condición o interpretar al revés una conclusión. La validación de substring es útil para integridad; evaluar soporte semántico exige otra rúbrica y datos adicionales.

## Demo de cinco minutos

Usa los comandos de la versión actual del [README](../README.md). Prepara una ejecución reproducible y evita editar datos durante la demo.

1. **0:00–0:40:** plantea la pregunta experimental y muestra el corpus/split/configuración.
2. **0:40–1:40:** recupera una consulta con BM25, dense y RRF; explica una diferencia concreta entre rankings.
3. **1:40–2:40:** abre el reporte real, el conteo de consultas y las métricas por consulta. Explica una regresión además de una mejora.
4. **2:40–3:35:** muestra qué mide la latencia y qué queda fuera: descarga, carga de modelo, indexación o caché de queries, según la corrida.
5. **3:35–4:20:** inspecciona un evidence pack y la integridad de sus referencias. Si no ejecutaste generación live, dilo explícitamente.
6. **4:20–5:00:** presenta tu contribución, una decisión descartada y el siguiente experimento que resolvería una incertidumbre real.

Una captura de un reporte es ilustración. Para una afirmación cuantitativa, enlaza también configuración y resultados que otra persona pueda reproducir.

## Ablations útiles para una contribución propia

No es necesario implementar todas. Escoge una, define la hipótesis antes de medir y publica también los resultados negativos.

- **Prefix de consulta:** con/sin la instrucción recomendada por el modelo, manteniendo el resto igual; decidir con train/dev, no test.
- **Título y abstract:** representación conjunta frente a abstract solo, registrando truncación y longitud.
- **Profundidad de candidatos:** medir qué gana RRF al recibir más resultados y cuánto cuesta.
- **Reranking:** si se implementa, comparar el mismo pool antes y después del cross-encoder, incluyendo tiempo de inferencia.
- **Cuantización:** comparar artefactos identificados con hashes, sin asumir equivalencia con pesos de otra precisión.

Si cambias varias variables a la vez, el resultado compara sistemas completos; no permite atribuir causalmente la diferencia a una sola elección.

## Extensiones avanzadas — roadmap, no funciones incluidas

**Pruebas contrafactuales de grounding.** Quita el documento relevante o reemplázalo por uno que contradiga el claim. Comprueba si una futura respuesta cambia, matiza o se abstiene. Evalúa además si el juez distingue soporte de simple coincidencia literal.

**Recuperación multi-hop.** Registra subconsultas, evidencia obtenida por paso y criterio de parada. Necesitarás tareas que requieran combinar documentos; no basta con llamar dos veces al mismo retriever.

**Contextual retrieval.** Enriquece documentos con contexto antes de indexar y compáralos con texto original. Registra el origen y costo del enriquecimiento; impedir que use la consulta evaluada o su respuesta esperada es parte del diseño.

**Evaluación semántica calibrada.** Construye una rúbrica de afirmaciones/citas, etiqueta una muestra humana y mide desacuerdos antes de confiar en un LLM-as-judge. Mantén separados errores del sistema y errores del juez.

**Operación a mayor escala.** Añade ANN, actualizaciones de índice y control de acceso solo cuando el problema lo requiera. Conserva el índice exacto pequeño como referencia para comprobar pérdida de recall y errores de aislamiento.

## Checklist para tus afirmaciones públicas

- Cada cifra apunta a un reporte real y dice corpus, split, métodos, configuración y fecha.
- El número de consultas evaluadas coincide con el alcance anunciado; una prueba corta no se presenta como el benchmark completo.
- Las comparaciones usan el mismo corpus y las mismas consultas, y los errores no se excluyen silenciosamente.
- Distingues modelo original, variante ONNX, parámetros del índice y artefactos descargados.
- No afirmas haber evaluado respuestas live si solo ejecutaste retrieval local.
- No presentas recuperación biomédica como verificación clínica, consejo médico o evidencia científica definitiva.
- Puedes mostrar una modificación propia, una prueba y un hallazgo que cambió una decisión de ingeniería.

Para profundizar: [protocolo experimental](EXPERIMENT.md) y [datos, modelos y fuentes](DATA_AND_MODELS.md).
