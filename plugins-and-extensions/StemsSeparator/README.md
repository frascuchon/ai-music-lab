# Stem Separator

Separación de stems en REAPER con dos backends:

- **DEMUCS** (local) — htdemucs, htdemucs_ft, htdemucs_6s, mdx_extra
- **SAM Audio** (cloud GPU vía Modal.com) — separación por prompt de texto

## Requisitos

- REAPER con la extensión **ReaImGui** (instalar vía ReaPack)
- Python ≥ 3.10 configurado en REAPER (Preferences → Plug-ins → ReaScript)
- Para SAM Audio: cuenta en [modal.com](https://modal.com) y token de [Hugging Face](https://huggingface.co/settings/tokens)

## Instalación en REAPER

1. Copia la carpeta `StemsSeparator/` donde quieras (p.ej. `~/Documents/REAPER/Scripts/StemsSeparator/`).
2. En REAPER: **Actions** → **Show action list** → **Load ReaScript** → selecciona `StemSeparator.lua`.
3. Repite el paso 2 con `Setup.lua`.
4. Opcional: asigna atajos de teclado a ambas acciones.

## Configuración (primera vez)

Abre `Setup.lua` desde REAPER Actions. El wizard:

1. **Detecta el entorno**: Python, uv, demucs, Modal CLI, autenticación Modal, HF secret.
2. **Instala demucs** si falta (botón inline).
3. **Login Modal**: lanza `modal token new` y abre el navegador automáticamente. Espera hasta que el token quede guardado en `~/.modal.toml`.
4. **Crea el secret de Hugging Face**: pega tu token `hf_...` y lo guarda como Modal secret `huggingface-secret`.
5. **Pre-warm** (opcional): descarga los pesos del modelo SAM Audio a un Modal Volume para que la primera separación no tarde 10-20 min extra.

Si `uv` no está instalado, el wizard muestra el comando de instalación y abre la documentación.

`StemSeparator.lua` muestra un banner amarillo en la parte superior si detecta configuración incompleta, con instrucción de abrir `Setup.lua`.

## Uso

1. Selecciona un track o media item en REAPER.
2. Abre `StemSeparator.lua` → haz clic en **R** para capturar la fuente.
3. Elige la pestaña **DEMUCS** (local, gratis) o **SAM AUDIO** (cloud, ~$0.10/pista).
4. Pulsa **SEPARAR** — los stems se importan como carpeta de tracks al terminar.

## Troubleshooting

| Síntoma | Solución |
|---|---|
| Banner "Config. incompleta" en StemSeparator | Abre Setup.lua y completa los pasos que aparecen en rojo |
| Demucs falla silenciosamente | `Setup.lua` → ✗ demucs → Instalar |
| SAM falla con código de error | Abre Setup.lua y verifica Modal auth + HF secret |
| `modal token new` no abre el navegador | Comprueba que `uv` esté en PATH y ejecuta manualmente en terminal |
| Pesos del modelo tardan 10-20 min | Normal en la primera ejecución; usa Pre-warm en Setup.lua para hacerlo antes |
