# Presentar EvidenceForge con evidencia

EvidenceForge demuestra decisiones de AI Engineering mediante un experimento auditable: mismo corpus, mismas consultas, tres métodos y resultados por consulta. Es un proyecto personal desarrollado con asistencia de IA. El trabajo local incluye recuperación con inferencia real de embeddings; no implica que se hayan evaluado respuestas generativas live.

[Repositorio](https://github.com/JhoanFuentes/evidenceforge) · [Reporte interactivo](https://jhoanfuentes.github.io/evidenceforge/) · [Estado de CI](https://github.com/JhoanFuentes/evidenceforge/actions/workflows/ci.yml) · [Guía de revisión técnica](REVIEW_GUIDE.md)

La validación local registra 105 pruebas aprobadas con el extra dense. Para hablar de CI, consulta la ejecución remota del commit correspondiente; una prueba local y un workflow alojado son evidencias distintas.

## Cómo complementa HarnessLab

[HarnessLab](https://github.com/JhoanFuentes/harnesslab) estudia contratos del runtime: persistencia, recuperación, presupuestos y aprobaciones ante decisiones controladas. EvidenceForge estudia calidad de recuperación con datos públicos y un modelo local. Juntos permiten explicar dos preguntas diferentes: **¿el sistema conserva sus garantías?** y **¿recupera evidencia útil?** Ninguno establece por sí solo que una respuesta sea verdadera o que el sistema esté listo para producción.

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

Resultado ya medido para explicar: dense obtuvo nDCG@10 **0,7127** frente a **0,6634** de BM25; diferencia **+0,0493**, intervalo bootstrap pareado del 95% **[+0,0128; +0,0850]**. RRF obtuvo **0,7021**; su diferencia frente a dense fue **−0,0106**, con intervalo **[−0,0377; +0,0164]**, que cruza cero. RRF sí tuvo el mayor Recall@5 (**0,7712**). La decisión depende del objetivo de recuperación y del presupuesto de latencia: no corresponde declarar a un método superior en todos los criterios. Presenta también sus regresiones: dense empeoró frente a BM25 en 52 de las 300 consultas. Estas cifras describen recuperación en este benchmark público, no exactitud de respuestas generadas ni resultados de negocio.

Una entrada concisa y verificable para CV, como proyecto personal:

> **EvidenceForge — evaluación reproducible de recuperación de evidencia.** Proyecto desarrollado con asistencia de IA: comparación de BM25, embeddings BGE locales y RRF sobre 5.183 documentos y 300 consultas. Artefactos auditables con rankings por consulta, intervalos bootstrap pareados, mediciones de latencia y controles de integridad de citas. Validación local: 105 pruebas automatizadas aprobadas con el extra dense.

En entrevista, concreta qué decisiones revisaste personalmente, qué parte extendiste y qué experimento puedes reproducir. El objetivo Senior AI Engineer / Applied AI no convierte este proyecto en experiencia laboral de esa senioridad.

## Cuatro semanas de aprendizaje

| Semana | Ejercicio y entregable |
| --- | --- |
| 1 | Reproduce la corrida; sigue cinco consultas desde qrels hasta métricas. Entrega una explicación de un acierto y una falla por método. |
| 2 | En train, compara título+abstract frente a abstract solo. Congela parámetros antes de evaluar; identifica nuevos resultados sobre el test ya visto como exploratorios. Entrega hipótesis, configuración y análisis de regresiones. |
| 3 | Altera un índice, duplica un resultado y crea una cita inexistente. Comprueba rechazos y deduplicación. Entrega un PR con una prueba de regresión útil. |
| 4 | Investiga una limitación y prepara la demo. Entrega un ADR, un reporte reproducible y una propuesta de experimento siguiente. |

Multi-hop retrieval, pruebas contrafactuales de grounding, contextual retrieval y evaluación semántica calibrada son **extensiones pendientes**, no capacidades que debas anunciar como terminadas. Elige una y demuestra su aporte antes de ampliar el alcance.

## Borrador opcional para LinkedIn

> EvidenceForge es mi proyecto personal, desarrollado con asistencia de IA, para comparar recuperación de evidencia científica con BM25, embeddings locales y fusión de rankings. En una corrida CPU sobre 5.183 documentos y 300 consultas, dense obtuvo nDCG@10 de 0,7127 frente a 0,6634 de BM25. La fusión RRF no mostró una mejora concluyente sobre dense en esa métrica, aunque consiguió el mayor Recall@5. El resultado útil es la decisión que permiten tomar esos datos: explicitar la métrica objetivo, inspeccionar regresiones y considerar latencia antes de añadir complejidad. El repositorio conserva configuraciones, rankings e intervalos pareados para auditar el experimento. También distingue recuperación relevante, integridad de citas y soporte semántico: una cita literal no demuestra que una respuesta sea verdadera. Código y reporte: https://github.com/JhoanFuentes/evidenceforge

Adapta la voz del borrador a tu experiencia y comprensión real antes de publicarlo. El valor del portafolio está en poder defender decisiones y fallas con datos, sin prometer un resultado laboral.
