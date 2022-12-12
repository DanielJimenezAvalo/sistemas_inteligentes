# -*- coding: latin-1 -*-
import numpy as np
import random
import warnings
import random
import copy
import pandas as pd
import time
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime
from PSO_python import *
import plotly.express as px
# Ejemplo optimizacion

def funcion_objetivo(x_0, x_1):
    '''
    Para la región acotada entre -10<=x_0<=0 y -6.5<=x_1<=0 la función tiene
    múltiples mínimos locales y un único minimo global que se encuentra en
    f(-3.1302468,-1.5821422) = -106.7645367
    '''
    g1 = -2-1*x_0-1*x_1
    g2 = -3-1*x_0-2*x_1
        
    f_1 = np.sin(x_1)*np.exp(1-np.cos(x_0))**2 \
        + np.cos(x_0)*np.exp(1-np.sin(x_1))**2 \
        + (x_0-x_1)**2
        
    if len(g1[g1>=0]):
        penalizacion_g1 = 0
    else:
        penalizacion_g1 = g1
        
    if len(g2[g2>=0]):
        penalizacion_g2 = 0
    else:
        penalizacion_g2 = g2
        
    f = f_1 + 1*penalizacion_g1**2 + 2*penalizacion_g2**2 # cambiar 1 y 2
         
    return(f)

def extraer_posicion(particula):
    posicion = particula.posicion
    return(posicion)

# Contour plot función objetivo
x_0 = np.linspace(start = -10, stop = 0, num = 100)
x_1 = np.linspace(start = -6.5, stop = 0, num = 100)
x_0, x_1 = np.meshgrid(x_0, x_1)
z = funcion_objetivo(x_0, x_1)
plt.contour(x_0, x_1, z, 35, cmap='RdGy')
plt.show()

enjambre = Enjambre(
               n_particulas = 50,
               n_variables  = 2,
               limites_inf  = [-10, -6.5],
               limites_sup  = [0, 0],
               verbose      = False
            )

enjambre.optimizar(
    funcion_objetivo = funcion_objetivo,
    optimizacion     = "minimizar",
    n_iteraciones    = 250,
    inercia          = 0.8,
    reduc_inercia    = True,
    inercia_max      = 0.9,
    inercia_min      = 0.4,
    peso_cognitivo   = 1,
    peso_social      = 2,
    parada_temprana  = True,
    rondas_parada    = 5,
    tolerancia_parada = 10**-3,
    verbose          = False
)

print(enjambre)

# EvoluciÃ³n de la optimizaciÃ³n
fig = plt.figure(figsize=(6,4))
enjambre.resultados_df['mejor_valor_enjambre'].plot()
plt.xlabel("Iteración")
plt.ylabel("Func.Objetivo")
plt.show()

# Se extrae la posición de las partículas en cada iteración del enjambre



lista_df_temp = []

for i in np.arange(len(enjambre.historico_particulas)):
    posiciones = list(map(extraer_posicion, enjambre.historico_particulas[i]))
    df_temp = pd.DataFrame({"iteracion": i, "posicion": posiciones})
    lista_df_temp.append(df_temp)

df_posiciones = pd.concat(lista_df_temp)

df_posiciones[['x_0','x_1']] = pd.DataFrame(df_posiciones["posicion"].values.tolist(),
                                            index= df_posiciones.index)

print("--------------------------------------")
print(df_posiciones.head())

px.scatter(
    df_posiciones,
    x       = "x_0",
    y       = "x_1",
    range_x = [-10, 0],
    range_y = [-6.5, 0],
    animation_frame = "iteracion"
)

import matplotlib.animation as animation
fig = plt.figure(figsize=(8,5))
plt.xlim(-10,0)
plt.ylim(-6.5,0)
def animate(i):
    p2 = fig.clear()
    plt.xlim(-10,0)
    plt.ylim(-6.5,0)
    df_posiciones_i = df_posiciones[df_posiciones["iteracion"] == i][["x_0", "x_1"]] #select data range
    p1 = plt.contour(x_0, x_1, z, 35, cmap='RdGy')
    p2 = plt.scatter(df_posiciones_i["x_0"], df_posiciones_i["x_1"])
    plt.suptitle("Iteración "+str(i))
    return 
ani = matplotlib.animation.FuncAnimation(fig, animate, range(1,enjambre.realnum_iter+1),repeat = True, blit = False)
plt.show()

''' file1 = "PSO.mp4" 
fig = plt.figure(figsize=(8,5))
plt.xlim(-10,0)
plt.ylim(-6.5,0)
matplotlib.rcParams['animation.ffmpeg_path'] = r'C:\\ffmpeg\bin\ffmpeg.exe'
writervideo = matplotlib.animation.FFMpegWriter(fps=15)
ani2 = matplotlib.animation.FuncAnimation(fig, animate, range(1,enjambre.realnum_iter+1), blit = False) 
ani2.save(file1, writer=writervideo)
'''