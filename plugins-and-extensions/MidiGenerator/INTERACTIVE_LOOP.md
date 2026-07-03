# Generación incremental con feedback humano (Human-in-the-Loop)

## Concepto

Los modelos de generación MIDI pueden funcionar en dos modos:

- **Batch**: se da todo el input, se obtiene todo el output de una vez (batch offline).
- **Incremental**: se genera en chunks cortos (2-5s), el usuario acepta o rechaza cada chunk antes de continuar.

El modo incremental es el que describe AMT como *"non-linear, asynchronous feedback controller"*:

> "Users incrementally generate a short sequence of notes (2-5 seconds at a time) and decide whether to accept or reject proposed continuations."

El modelo toma los fragmentos **aceptados** como historial para condicionar la siguiente generación. Si el usuario rechaza un chunk, se re-genera con diferente semilla sin alterar el historial acumulado.

```
historia    [──────────────]
                           ↓
                     generate(chunk N)
                           ↓
              ┌──────────────────────┐
              │  candidatos v0/v1/v2  │
              └──────────────────────┘
                           ↓
             usuario escucha y elige
                     ↙         ↘
               acepta           rechaza
                 ↓                  ↓
        historia += vN        re-genera
                 ↓            (seed+k)
          chunk N+1
```

---

## Por qué es relevante para un plugin DAW

El flujo batch produce **un resultado de N segundos de una sola vez**, lo que obliga al usuario a aceptarlo o descartarlo entero. El flujo incremental permite **co-autoría real**: el usuario guía la composición en tiempo real, fragmento a fragmento, como si estuviera dirigiendo al modelo.

Para un plugin en REAPER esto tiene valor claro:
- El usuario mantiene control sobre la evolución armónica y rítmica.
- Los errores del modelo no contaminan más de un chunk de 3s.
- El historial de aceptados actúa como "memoria compositiva" — el modelo no diverge del estilo establecido por el usuario.
- El `top_p` puede bajarse progresivamente para consolidar el estilo a medida que avanza la pieza.

---

## Requisitos de la API del modelo para soportar el loop

Para que un modelo soporte generación incremental necesita cumplir tres condiciones:

| Requisito | Descripción |
|---|---|
| **1. Entrada de historial explícita** | El modelo debe aceptar un buffer de eventos ya generados/aceptados (`inputs`) separado del input de control. |
| **2. Ventana temporal configurable** | La generación debe poder arrancar en un `start_time > 0` y terminar en `end_time = start_time + chunk`. |
| **3. Condicionamiento externo fijo** | Un control externo (melodía, acordes, estructura) que permanece fijo durante todas las iteraciones. |

Un modelo que solo expone `generate(prompt) → output` (como Text2midi o MIDI-LLM en su modo básico) **no puede** participar en este loop directamente sin un wrapper que simule la acumulación de historial.

---

## Estado de soporte en los modelos evaluados

| Modelo | Historial explícito | Ventana configurable | Control externo | Loop nativo |
|---|---|---|---|---|
| **AMT** (Anticipatory MT) | ✅ `inputs=` | ✅ `start_time / end_time` | ✅ `controls=` (melodía) | ✅ diseñado para esto |
| **Aria** (EleutherAI) | ✅ MIDI de entrada como contexto | ✅ (continuation) | ❌ piano solo, sin control externo | Parcial — solo continuation |
| **Text2midi** | ❌ generación desde 0, no acepta historial | ❌ siempre genera desde t=0 | ❌ no acepta MIDI de guía | ❌ |
| **MIDI-LLM** | ❌ idem Text2midi | ❌ | ❌ | ❌ |
| **MuseCoco** | ❌ atributos en texto, no historial MIDI | ❌ | ❌ | ❌ |

AMT es el único modelo de los evaluados con soporte nativo completo del loop. Para los demás habría que investigar si exponen APIs internas que lo permitan (poco probable sin reentrenamiento).

---

## Implementación en AMT: cómo funciona el loop

La función `generate()` de AMT (`anticipation/sample.py`) ya soporta el patrón:

```python
generate(model, start_time, end_time, inputs=None, controls=None, top_p=1.0)
```

- `inputs` — eventos aceptados hasta `start_time` (crece cada iteración)
- `controls` — melodía o guía completa (fija desde el principio)
- `start_time` / `end_time` — ventana del chunk actual

**Pseudocódigo del loop**:
```python
controls = extract_melody(input_midi)           # fija, no cambia
accepted = clip(non_melody_events, 0, 5)        # primeros 5s como prompt inicial
t = 5
chunk = 3  # segundos por iteración

while t < total_duration:
    candidates = []
    for i in range(N_candidates):
        np.random.seed(base_seed + iteration * N_candidates + i)
        c = generate(model, t, t + chunk,
                     inputs=accepted, controls=controls, top_p=top_p)
        candidates.append(c)

    chosen = user_picks(candidates)             # acepta uno, o rechaza todos

    if chosen is not None:
        accepted = sort(accepted + clip(chosen, t, t + chunk))
        t += chunk
        top_p = max(0.85, top_p - 0.02)        # opcional: consolidar estilo gradualmente

# resultado final
output = clip(combine(accepted, controls), 0, total_duration)
mid = events_to_midi(output)
```

Cada iteración es **reanudable**: si se serializa `accepted` como MIDI y se guarda `t`, el loop puede interrumpirse y continuarse en otra sesión.

---

## Esquema de integración con REAPER (diseño, no implementado)

```
┌─────────────────────────────────┐
│          REAPER (Lua)           │
│                                 │
│  1. Exporta melodía como MIDI   │──→ melody.mid
│  2. Exporta historial aceptado  │──→ accepted.mid + t=N
│  3. Llama al backend Python     │
└────────────────┬────────────────┘
                 ↓
┌─────────────────────────────────┐
│   generate_amt.py (CLI)         │
│   --mode accompaniment          │
│   --controls melody.mid         │
│   --inputs accepted.mid         │
│   --start-time N                │
│   --chunk 3                     │
│   --multiplicity 3              │──→ chunk_v0.mid, chunk_v1.mid, chunk_v2.mid
└─────────────────────────────────┘
                 ↓
┌─────────────────────────────────┐
│          REAPER (Lua)           │
│                                 │
│  4. Importa 3 candidatos        │
│  5. Usuario audita en REAPER    │
│  6. Acepta uno (o rechaza all)  │
│  7. Actualiza accepted.mid      │
│  8. Vuelve al paso 2            │
└─────────────────────────────────┘
```

**Contratos importantes para el plugin**:
- El archivo `accepted.mid` es el **estado compartido** entre REAPER y el backend Python.
- `t` (tiempo actual del playhead de generación) se puede persistir como metadato en el nombre del archivo o en un JSON de estado.
- Si el usuario rechaza todos los candidatos en una iteración, el backend se relanza con `seed = prev_seed + N` sin modificar `accepted.mid`.
- El `top_p` puede exponerse en la UI como slider "creatividad" (alto = explorador, bajo = conservador).

---

## Extensión a otros modelos futuros

Cualquier modelo que quiera participar en este loop desde el plugin deberá exponer:

```
generate_chunk(
    model,
    history_midi_path: str,   # eventos aceptados hasta start_time
    controls_midi_path: str,  # guía/melodía fija (opcional)
    start_time: float,        # segundos desde inicio
    chunk_seconds: float,     # duración del chunk a generar
    seed: int,
    top_p: float,
) -> midi_bytes
```

Esta interfaz es suficientemente genérica para AMT, y podría adaptarse a modelos futuros que soporten condicionamiento por tokens previos (e.g., variantes de MusicGen que acepten `melody_conditioning` + `continuation_tokens`).

---

*Análisis realizado: 2026-06-11. Basado en `humaneval/accompany.py` (upstream Anticipatory Music Transformer) e issue #18 del repositorio. El loop batch (sin interactividad) está implementado en `research/research_anticipatory.py::run_accompaniment`.*
