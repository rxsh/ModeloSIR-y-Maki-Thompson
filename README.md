# ModeloSIR-y-Maki-Thompson

Repositorio con implementaciones y experimentos para los modelos SIR y Maki-Thompson
de propagación (rumores/enfermedades). Contiene código para simular modelos, comparar
métodos numéricos y generar visualizaciones listas para presentación.

Requisitos
- Python 3.8+
- numpy
- scipy
- matplotlib

Uso rápido
1. Instalar dependencias:

```powershell
pip install -r requirements.txt
```

2. Ejecutar el script principal para regenerar figuras (se guardan en `outputs/`):

```powershell
python .\run_all.py
```

Salida
- Figuras PNG en la carpeta `outputs/`.
- Archivo `outputs/maki_final_proportions.txt` con las proporciones finales de informantes.

