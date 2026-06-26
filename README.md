# Proyecto Final - Algoritmos Bioinspirados

## Descripción

Este proyecto implementa el algoritmo de **Recocido Simulado (Simulated Annealing)** con manejo de restricciones mediante **Stochastic Ranking**.

El algoritmo se aplica a dos problemas clásicos de optimización con restricciones:

- Diseño de un resorte de tensión/compresión.
- Diseño de una viga soldada.

El programa realiza múltiples ejecuciones independientes para distintos valores del parámetro Pf, calcula estadísticas de desempeño y genera gráficas de convergencia.

---

## Requisitos

- Python 3.10 o superior

Instalar las siguientes librerías:

```bash
pip install numpy matplotlib
```

---

## Archivos

- `RS_P4.py`: implementación completa del algoritmo.
- `README.md`: instrucciones de uso.

---

## Cómo ejecutar

1. Abrir una terminal en la carpeta del proyecto.

2. Ejecutar:

```bash
python RS_P4.py
```

---

## Salida del programa

Al ejecutarse, el programa:

- Resuelve el problema del resorte.
- Resuelve el problema de la viga soldada.
- Ejecuta 30 corridas independientes para cada valor de Pf.
- Genera gráficas de convergencia.
- Genera gráficas del comportamiento instantáneo del algoritmo.
- Imprime una tabla con:
  - Promedio de la función objetivo.
  - Desviación estándar.
  - Número de soluciones factibles.

---

## Autores
Alcantara Ascencio Leticia,
Najar Carbajal Santiago
