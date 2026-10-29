# Stem Separator

Separación de stems en REAPER con dos backends:

- **DEMUCS** (local) — htdemucs, htdemucs_ft, htdemucs_6s, mdx_extra
- **SAM Audio** (cloud GPU vía Modal.com) — separación por prompt de texto

## Instalación

Se instala con ReaPack como parte del plugin unificado **AI Music Lab** (pestaña Stems Separator); ver la [guía de instalación](../../README.md#quick-install-about-10-minutes) del README principal. La configuración (uv, Modal, tokens) se hace desde el botón **Setup** del panel.

## Uso

1. Selecciona un track o media item en REAPER.
2. Abre **AI Music Lab** → pestaña Stems Separator → haz clic en **R** para capturar la fuente.
3. Elige la pestaña **DEMUCS** (local, gratis) o **SAM AUDIO** (cloud, ~$0.10/pista).
4. Pulsa **SEPARAR** — los stems se importan como carpeta de tracks al terminar.

## Troubleshooting

| Síntoma | Solución |
|---|---|
| Banner "Config. incompleta" en StemSeparator | Abre Setup y completa los pasos que aparecen en rojo |
| Demucs falla silenciosamente | Setup → ✗ demucs → Instalar |
| SAM falla con código de error | Abre Setup y verifica Modal auth + HF secret |
| `modal token new` no abre el navegador | Comprueba que `uv` esté en PATH y ejecuta manualmente en terminal |
| Pesos del modelo tardan 10-20 min | Normal en la primera ejecución; usa Pre-warm en Setup para hacerlo antes |
