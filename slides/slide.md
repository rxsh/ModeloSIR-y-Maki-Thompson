# Avance: Modelos SIR y Maki-Thompson — 5 minutos

**Objetivo (30 s)**

- **Qué**: Implementar y comparar modelos de propagación (SIR) y modelos de rumor (Maki‑Thompson / general).
- **Por qué**: Demostrar que el enfoque unificado permite estudiar difusión de enfermedades e información y comparar métodos numéricos.

**Métodos (60 s)**

- Modelos implementados: `sir_rhs`, `sir_rhs_full`, `rumor_general_rhs`, `maki_thompson_rhs` (`src/models.py`).
- Métodos numéricos: Euler explícito, Euler mejorado, RK4, y `solve_ivp` (RK45) como benchmark (`src/solvers.py`).
- Visualización y análisis reproducible en `run_all.py` (figuras guardadas en `outputs/`).

**Resultados clave (120 s)**

- SIR (ejemplo): con β=0.5, γ=0.1 → R0=5 (>1): se observa un pico en I(t) (ver `outputs/sir_timeseries.png`).
- Retrato de fase S–I: trayectorias para condiciones iniciales diferentes; marcamos I_max por trayectoria (`outputs/retrato_fase_SI.png`).
- Maki‑Thompson: simulaciones para varias condiciones iniciales; `Y_final` (proporción final informantes) guardada en `outputs/maki_final_proportions.txt`. Ejemplos: IC=[0.99,0.01] → Y_final≈1e-4.
- Comparación numérica: para dt=0.1, errores observados — Euler ≫ Euler mejorado ≫ RK4; gráfico `outputs/error_vs_dt.png` y `outputs/time_vs_dt.png` muestran coste/beneficio.

**Interpretación (60 s)**

- R0 (SIR) determina si habrá epidemia: R0 = β/γ. R0>1 → I(t) alcanza un pico; R0<1 → la infección se apaga.
- Diferencia conceptual entre SIR y Maki‑Thompson:
  - En SIR, la retirada (recuperación) es autónoma (γ). En Maki, la retirada depende de contactos con no‑ignorantes (α o δ) — esto cambia la dinámica de Y(t).
  - En la práctica, Y(t) puede decaer más rápidamente o persistir según α, δ y λ.
- Interpretación de resultados numéricos: elegir método depende del equilibrio entre precisión y coste. RK4 es más preciso por paso; Euler puede ser suficiente para exploraciones rápidas.

**Conclusiones y próximos pasos (30 s)**

- Estado: implementación completa de modelos, solvers y visualizaciones; análisis de métodos numéricos generado.
- Recomendación inmediata para la entrega: incluir una tabla breve con pares (β,γ) mostrando R0, I_max, t_peak (puedo generarla ahora). También exportar 2–3 figuras clave en la diapositiva.

---

Notas para el orador (guion, ~5 min total)

0:00–0:30 — Objetivo: explicar brevemente qué se modela y por qué es interesante comparar epidemias y rumores.

0:30–1:30 — Métodos: mostrar código/estructura del repo y dónde ejecutar (`python .\run_all.py`). Mencionar `requirements.txt`.

1:30–3:30 — Resultados: mostrar en pantalla `sir_timeseries.png`, `retrato_fase_SI.png` y `comparacion_metodos.png`. Para cada figura, explicar qué se observa (pico, dirección en retrato de fase, diferencias de error).

3:30–4:30 — Interpretación: explicar R0 y diferencia entre mecanismos de retirada en SIR vs Maki. Mostrar `maki_final_proportions.txt` y explicar significado de Y_final.

4:30–5:00 — Cierre: conclusiones y próximos pasos propuestos (barrido β/γ y tabla de resumen; exportar notebooks a PDF para anexar a la entrega).

Instrucciones rápidas para generar PDF/slide visual (opcional):

1. Asegúrate de tener `pandoc` o `nbconvert` si quieres un PDF del Markdown.

Ejemplo con `pandoc` (Windows PowerShell):
```powershell
pandoc slides\5min_slide.md -o slides\5min_slide.pdf --pdf-engine=pdflatex
```

2. Alternativa rápida: abrir `slides\5min_slide.md` y copiar el contenido a PowerPoint/Google Slides; insertar las imágenes desde `outputs/`.

*** Fin de la diapositiva
