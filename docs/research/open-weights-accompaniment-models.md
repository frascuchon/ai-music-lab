# Modelos de pesos abiertos para acompañar o armonizar una melodía MIDI

Investigación del 2026-10-27. Fuentes: README y páginas de arXiv de cada proyecto, consultados ese día.
Marcado como **no verificado** lo que no pude confirmar. MIDI-GPT se probó después (ver «Prueba de MIDI-GPT»); el resto no se ha ejecutado.

## Resumen

| Modelo | Qué hace con una melodía | Pesos y licencia | Validación | Coste de integrarlo |
|---|---|---|---|---|
| **Anticipatory Music Transformer** (el actual) | Acompañamiento condicionado por la melodía | En HF (`stanford-crfm/music-*-800k`); Apache 2.0 según nuestro script, no reverificado | TMLR; evaluadores humanos juzgaron la musicalidad de sus acompañamientos de 20 s «similar» a la de música compuesta por humanos (según el resumen consultado; protocolo y cifras no verificados) | Ya integrado |
| **MIDI-GPT** (Metacreation, AAAI 2025) | Genera una pista nueva o rellena compases teniendo las otras pistas como contexto; controles de densidad, polifonía y duración | HF `Metacreation/MIDI-GPT` (acceso con aceptación de licencia). Código MIT, **pesos de uso no comercial** (investigación y educación) | AAAI 2025; el resumen solo dice que evita copiar el entrenamiento, se parece al corpus y respeta los controles (detalles de la evaluación no verificados) | Bajo: `pip install "midigpt[inference]"`, API Python, wheels macOS/Linux/Windows. Existe un plugin de REAPER del mismo laboratorio |
| **Structured Accompaniment Arrangement / AccoMontage-3** (NeurIPS 2024) | Partitura de acordes (lead sheet) → acompañamiento de piano → arreglo multipista | Checkpoints en Google Drive; MIT | NeurIPS 2024 | Alto: Python 3.8, PyTorch 1.10 con CUDA 11.1, se usa mediante notebooks, entrada en formato lead sheet |
| **GETMusic** (Microsoft, IJCAI 2025) | Genera pistas (bajo, batería, guitarra, piano, cuerdas) a partir de pistas de condición, p. ej. melodía + acordes | **El README no enlaza checkpoints**; el repositorio está archivado y no publican los datos | IJCAI 2025 | Inviable si no hay pesos (**no verificado** si existen en otro sitio) |
| **Multitrack Music Transformer** (ICASSP 2023) | Generación multipista; el modo de condicionar por una melodía no está documentado en el README | Modelos en una carpeta SharePoint de UCSD; MIT | ICASSP 2023 | Medio-alto: entorno conda, datos preprocesados, repo antiguo |
| Moonbeam, Function Alignment (ISMIR 2025) | Infilling, generación de melodía a partir de acordes, batería, reconocimiento de acordes | Pesos abiertos según los autores; licencia **no verificada** | ISMIR 2025 / arXiv 2025 | No cubren acompañamiento armónico de una melodía |

## Recomendación

1. **Mantener Anticipatory** para la demo. Con el arranque automático ya produce un acompañamiento razonable (confirmado por el usuario), es de pesos permisivos y es el único con validación humana explícita para esta tarea.
2. **Probar MIDI-GPT como segunda opinión.** Es el que mejor combina tarea (nueva pista condicionada por las demás), instalación sencilla y relación con REAPER. Limitaciones: los pesos son solo para uso no comercial (válido para una charla, no para distribuir el plugin comercialmente) y requiere aceptar la licencia en Hugging Face con la cuenta del usuario. No se ha probado con la melodía del usuario.
3. **AccoMontage-3** solo si interesa un acompañamiento de piano de más calidad a partir de una progresión de acordes: exige lead sheet y un entorno antiguo.
4. **Descartados por ahora:** GETMusic (sin pesos localizables), MMT (condicionamiento no documentado), Moonbeam y Function Alignment (otras tareas).

## Prueba de MIDI-GPT (2026-10-27)

Script: `plugins-and-extensions/MidiGenerator/research/research_midigpt_modal.py` (Modal, usa el secret `huggingface-secret`).
Modelo `yellow_medium`, melodía de guitarra del usuario (49 notas, 8 compases, 100,4 BPM), dos pistas nuevas (bajo, teclado) generadas con la melodía como contexto.

- **La API sí permite el caso de uso**: `TrackPrompt(bars=[...], autoregressive=True)` genera una pista nueva y las demás cuentan como contexto. Admite además restringir las notas a un conjunto de clases de altura o a un registro (`controls["pitch_mask"]`) y fijar densidad y polifonía.
- **Sin restricciones de registro** el bajo bajaba hasta la nota MIDI 20 y el teclado hasta la 27 (inaudible). Con `pitch_mask.shape` (bajo 33–52, teclado 52–76) quedó bien.
- **Irregularidad** (6 candidatos con registro corregido): cobertura de la melodía 29–97 %, densidad 2–25 notas/s. Anticipatory con el arranque automático, en 8 candidatos: 89–99 % y 3,5–5,7 notas/s (el estilo `pad` es más escaso, 1,8 notas/s).
- Consonancia con la melodía al inicio de cada nota: 58–85 % (MIDI-GPT) frente a 64–84 % (Anticipatory). Métrica burda.
- **No se ha comparado a oído**. Audios de comparación generados en el scratchpad de la sesión.
- Sin controles de densidad MIDI-GPT es menos predecible; la API ofrece `attributes` (densidad, polifonía, duración) para acotarlo, sin probar.

**Conclusión provisional:** MIDI-GPT funciona y es viable técnicamente, pero hoy da resultados más irregulares que Anticipatory con el arranque automático y sus pesos son solo no comerciales. Merece la pena como segunda opción si el oído del usuario prefiere sus candidatos; hacerlo fiable exige ajustar densidad/polifonía.

## Estado: integrado en el plugin (2026-10-27)

MIDI-GPT se añadió al panel de MidiGenerator como tercer modelo, junto a Anticipatory, con los mismos controles (partes, estilo, densidad, instrumentos, creatividad). Resultados con presets (2 semillas cada uno): 0 partes duplicadas en las 10 configuraciones probadas; densidad 2,7 → 5,2 → 6,5 notas/s; hasta 5 partes. Atributos del checkpoint `yellow_medium`: `note_density` (solo pistas de batería), `min/max_polyphony`, `min/max_note_duration` (niveles 0–5). La duración larga no discriminó en las pruebas; el control efectivo es la **máscara rítmica por parte** (posiciones dentro del compás, 12 ticks por negra).

## Fuentes
- MIDI-GPT: <https://github.com/Metacreation-Lab/MIDI-GPT>, <https://arxiv.org/abs/2501.17011>
- Anticipatory Music Transformer: <https://arxiv.org/abs/2306.08620>
- Structured Accompaniment Arrangement: <https://github.com/zhaojw1998/AccoMontage-3>
- GETMusic: <https://github.com/microsoft/muzic/tree/main/getmusic>, <https://arxiv.org/abs/2305.10841>
- MMT: <https://github.com/salu133445/mmt>, <https://arxiv.org/abs/2207.06983>
- Function Alignment: <https://arxiv.org/abs/2506.15548>; Moonbeam: <https://arxiv.org/abs/2505.15559>
