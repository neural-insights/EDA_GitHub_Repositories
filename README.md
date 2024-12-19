# Data Science Repository Trends on GitHub: An Insightful EDA

### Project Break - EDA
Autor: Lucas Perez Barone

--------------------------------------

### Justificación: 
La Ciencia de Datos es un área de conocimiento amplia, reciente y que está siendo constantemente revolucionada por la velocidad con la que se producen nuevas tecnologías y técnicas computacionales. En consecuencia, el Científico de Datos y las instituciones que se benefician de ella también deben estar en constante aprendizaje y transformación para mantenerse al día con el avance técnico.

### Objetivos: 
El presente proyecto tiene como objetivo principal observar la evolución del campo de la Ciencia de Datos y la emergencia de nuevas tendencias en el área, con el fin de generar insights a partir de estas.

### Objetivos específicos:
* Visión general sobre el aumento del número de técnicas y subáreas dentro de la Ciencia de Datos a lo largo de los años;
* Identificar cuáles de estas técnicas y subáreas han aumentado su relevancia en los últimos años y cuáles han perdido utilidad;
* Buscar temas específicos dentro de la Ciencia de Datos que estén relacionados con la mayor popularidad y el mayor nivel de compromiso en los repositorios;
* Generar insights sobre los resultados obtenidos.

---------------------------------------

Bibliotecas necesarias para ejecutar el notebook (importadas automáticamente por el archivo ***user_functions.py***)

```pyton
from collections import Counter
from datetime import datetime, date
from itertools import combinations
import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu, pearsonr
import os
import pandas as pd
import random
import requests
import seaborn as sns
import time
from wordcloud import WordCloud
```


Para acceder a los requisitos de version de las bibliotecas del entorno virtual utilizado para ejecutar este proyecto, acceda al archivo 'requirements.txt'
