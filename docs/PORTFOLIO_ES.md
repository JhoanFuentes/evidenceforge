# Presentar EvidenceForge con evidencia

EvidenceForge demuestra decisiones de AI Engineering mediante un experimento auditable: mismo corpus, mismas consultas, tres métodos y resultados por consulta. El trabajo local incluye recuperación con inferencia real de embeddings; no implica que se hayan evaluado respuestas generativas live. El repositorio está preparado para publicarse, pero su creación local no significa que ya exista en GitHub. Los workflows de CI están preparados; su ejecución remota debe comprobarse después de subirlo.

## Cómo complementa HarnessLab

HarnessLab estudia contratos del runtime: persistencia, recuperación, presupuestos y aprobaciones ante decisiones controladas. EvidenceForge estudia calidad de recuperación con datos públicos y un modelo local. Juntos permiten explicar dos preguntas diferentes: **¿el sistema conserva sus garantías?** y **¿recupera evidencia útil?** Ninguno establece por sí solo que una respuesta sea verdadera o que el sistema esté listo para producción.

## Demo de tres minutos

| Tiempo | Evidencia que mostrar |
| --- | --- |
| 0:00–0:30 | Problema, corpus y configuración congelada. Aclara que BEIR test corresponde al dev público original de SciFact. |
| 0:30–1:10 | Una consulta con BM25, dense y RRF. Explica una diferencia concreta entre documentos recuperados. |
| 1:10–1:55 | Artefacto del benchmark: métricas por consulta, comparación pareada y latencia. Incluye una regresión, no solo el mejor ejemplo. |
| 1:55–2:30 | Verificación del artefacto y un evidence pack. Explica por qué una cita literal no prueba soporte semántico. |
| 2:30–3:00 | Tu contribución verificable, una decisión descartada y la siguiente hipótesis. |

Usa los comandos del [README](../README.md). Si muestras una captura, conserva también el artefacto que permite reproducirla.

## CV: frases que puedes adaptar

Úsalas después de comprender y verificar el trabajo; distingue tu contribución de la asistencia de IA recibida.

- “Proyecto personal: evaluación local de BM25, embeddings BGE y RRF sobre 5.183 abstracts y 300 consultas de BEIR SciFact, con resultados por consulta y configuración reproducible.”
- “Implementé [contribución propia] y validé [comportamiento concreto] mediante pruebas de regresión y revisión de artefactos.”
- “Analicé diferencias de recuperación con bootstrap pareado y mediciones de latencia; documenté límites por contaminación potencial, juicios incompletos y truncación.”

Añade mejoras porcentuales únicamente cuando el reporte las sustente. No conviertas tests locales en “CI aprobado”, recuperación en “precisión de respuestas”, ni un proyecto personal en experiencia productiva con clientes.

Resultado ya medido para explicar: dense obtuvo nDCG@10 **0,7127** frente a **0,6634** de BM25; diferencia **+0,0493**, intervalo bootstrap pareado del 95% **[+0,0128; +0,0850]**. RRF obtuvo **0,7021**, con más latencia y sin mejora concluyente frente a dense en esta métrica. Presenta también sus regresiones: dense empeoró frente a BM25 en 52 de las 300 consultas. Estas cifras describen recuperación en este benchmark público, no exactitud de respuestas generadas ni resultados de negocio.

## Cuatro semanas de aprendizaje

| Semana | Ejercicio y entregable |
| --- | --- |
| 1 | Reproduce la corrida; sigue cinco consultas desde qrels hasta métricas. Entrega una explicación de un acierto y una falla por método. |
| 2 | En train, compara título+abstract frente a abstract solo. Congela parámetros antes de evaluar; identifica nuevos resultados sobre el test ya visto como exploratorios. Entrega hipótesis, configuración y análisis de regresiones. |
| 3 | Altera un índice, duplica un resultado y crea una cita inexistente. Comprueba rechazos y deduplicación. Entrega un PR con una prueba de regresión útil. |
| 4 | Investiga una limitación y prepara la demo. Entrega un ADR, un reporte reproducible y una propuesta de experimento siguiente. |

Multi-hop retrieval, pruebas contrafactuales de grounding, contextual retrieval y evaluación semántica calibrada son **extensiones pendientes**, no capacidades que debas anunciar como terminadas. Elige una y demuestra su aporte antes de ampliar el alcance.

## Borrador opcional para LinkedIn

> Estoy trabajando en EvidenceForge, un laboratorio de recuperación de evidencia científica con BM25, embeddings locales y fusión de rankings. Me interesaba separar un resultado convincente de uno verificable: conservar configuración, inspeccionar cada consulta y comparar métodos sin ajustar parámetros sobre test. Una distinción clave es que recuperar evidencia relevante no significa confirmar una afirmación; un documento puede contradecirla. También separé integridad de citas y soporte semántico. El siguiente paso es publicar el código, los artefactos y una explicación de los casos donde cada método falla. Mi contribución específica fue [trabajo propio], y el hallazgo que cambió una decisión fue [hallazgo observado].

Reemplaza los campos antes de publicar. El valor del portafolio está en poder defender decisiones y fallas con datos, sin prometer un resultado laboral.
