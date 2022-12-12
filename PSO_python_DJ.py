# -*- coding: latin-1 -*-
#############################################################################
#                 CODIGO OPTIMIZACION CON ALGORITMO PSO                     #
#                                                                           #
# Libreria hecha por Joaquin Amat Rodrigo , licencia Creative Commons       #
# Attribution 4.0 International.                                            #
#############################################################################

#%%
################################################################################
#                          LIBRERIAS NECESARIAS                                #
################################################################################
import numpy as np
import random
import warnings
import random
import copy
import pandas as pd
import time
from datetime import datetime

#%%
################################################################################
#                              CLASE PARTICULA                                 #
################################################################################

class Particula:
    """
    Esta clase representa nueva part�cula con una posici�n inicial definida por
    una combinaci�n de valores num�ricos aleatorios y velocidad de 0. El rango
    de posibles valores para cada variable (posici�n) puede estar acotado. Al
    crear una nueva part�cula, solo se dispone de informaci�n sobre su posici�n 
    inicial y velocidad, el resto de atributos est�n vac�os.
    
    Parameters
    ----------
    n_variables : `int`
        n�mero de variables que definen la posici�n de la part�cula.
        
    limites_inf : `list` or `numpy.ndarray`, optional
        l�mite inferior de cada variable. Si solo se quiere predefinir l�mites
        de alguna variable, emplear ``None``. Los ``None`` ser�n remplazados
        por el valor (-10**3). (default is ``None``)
        
    limites_sup : `list` or `numpy.ndarray`, optional
        l�mite superior de cada variable. Si solo se quiere predefinir l�mites
        de alguna variable, emplear ``None``. Los ``None`` ser�n remplazados
        por el valor (+10**3). (default is ``None``)

    verbose : `bool`, optional
        mostrar informaci�n de la part�cula creada. (default is ``False``)

    Attributes
    ----------
    n_variables : `int`
        n�mero de variables que definen la posici�n de la part�cula.

    limites_inf : `list` or `numpy.ndarray`
        l�mite inferior de cada variable. Si solo se quiere predefinir l�mites
        de alguna variable, emplear ``None``. Los ``None`` ser�n remplazados por
        el valor (-10**3).

    limites_sup : `list` or `numpy.ndarray`
        l�mite superior de cada variable. Si solo se quiere predefinir l�mites
        de alguna variable, emplear ``None``. Los``None`` ser�n remplazados por
        el valor (+10**3).

    mejor_valor : `numpy.ndarray`
        mejor valor que ha tenido la part�cula hasta el momento.

    mejor_posicion : `numpy.ndarray`
        posici�n en la que la part�cula ha tenido el mejor valor hasta el momento.

    valor : `float`
        valor actual de la part�cula. Resultado de evaluar la funci�n objetivo
        con la posici�n actual.

    velocidad : `numpy.ndarray`
        array con la velocidad actual de la part�cula.

    posicion : `numpy.ndarray`
        posici�n actual de la part�cula.

    Raises
    ------
    raise Exception
        si `limites_inf` es distinto de None y su longitud no coincide con
        `n_variables`.

    raise Exception
        si `limites_sup` es distinto de None y su longitud no coincide con
        `n_variables`.

    Examples
    --------
    Ejemplo creaci�n part�cula.

    >>> part = Particula(
                    n_variables = 3,
                    limites_inf = [-1,2,0],
                    limites_sup = [4,10,20],
                    verbose     = True
                    )

    """
    
    def __init__(self, n_variables, limites_inf=None, limites_sup=None,
                 verbose=False):

        # N�mero de variables de la part�cula
        self.n_variables = n_variables
        # L�mite inferior de cada variable
        self.limites_inf = limites_inf
        # L�mite superior de cada variable
        self.limites_sup = limites_sup
        # Posici�n de la part�cula
        self.posicion = np.repeat(None, n_variables)
        # Velocidad de la part�cula
        self.velocidad = np.repeat(None, n_variables)
        # Valor de la part�cula
        self.valor = np.repeat(None, 1)
        # Mejor valor que ha tenido la part�cula hasta el momento
        self.mejor_valor = None
        # Mejor posici�n en la que ha estado la part�cula hasta el momento
        self.mejor_posicion = None
        
        # CONVERSIONES DE TIPO INICIALES
        # ----------------------------------------------------------------------
        # Si limites_inf o limites_sup no son un array numpy, se convierten en
        # ello.
        if self.limites_inf is not None \
        and not isinstance(self.limites_inf,np.ndarray):
            self.limites_inf = np.array(self.limites_inf)

        if self.limites_sup is not None \
        and not isinstance(self.limites_sup,np.ndarray):
            self.limites_sup = np.array(self.limites_sup)
        
        # COMPROBACIONES INICIALES: EXCEPTIONS Y WARNINGS
        # ----------------------------------------------------------------------
        if self.limites_inf is not None \
        and len(self.limites_inf) != self.n_variables:
            raise Exception(
                "limites_inf debe tener un valor por cada variable. " +
                "Si para alguna variable no se quiere l�mite, emplear None. " +
                "Ejemplo: limites_inf = [10, None, 5]"
                )
        elif self.limites_sup is not None \
        and len(self.limites_sup) != self.n_variables:
            raise Exception(
                "limites_sup debe tener un valor por cada variable. " +
                "Si para alguna variable no se quiere l�mite, emplear None. " +
                "Ejemplo: limites_sup = [10, None, 5]"
                )
        elif (self.limites_inf is None) or (self.limites_sup is None):
            warnings.warn(
                "Es altamente recomendable indicar los l�mites dentro de los " + 
                "cuales debe buscarse la soluci�n de cada variable. " + 
                "Por defecto se emplea [-10^3, 10^3]."
                )
        elif any(np.concatenate((self.limites_inf, self.limites_sup)) == None):
            warnings.warn(
                "Los l�mites empleados por defecto cuando no se han definido " +
                "son: [-10^3, 10^3]."
            )

        # COMPROBACIONES INICIALES: ACCIONES
        # ----------------------------------------------------------------------

        # Si no se especifica limites_inf, el valor m�nimo que pueden tomar las 
        # variables es -10^3.
        if self.limites_inf is None:
            self.limites_inf = np.repeat(-10**3, self.n_variables)

        # Si no se especifica limites_sup, el valor m�ximo que pueden tomar las 
        # variables es 10^3.
        if self.limites_sup is None:
             self.limites_sup = np.repeat(+10**3, self.n_variables)
            
        # Si los l�mites no son nulos, se reemplazan aquellas posiciones None por
        # el valor por defecto -10^3 y 10^3.
        if self.limites_inf is not None:
            self.limites_inf[self.limites_inf == None] = -10**3
           
        if self.limites_sup is not None:
            self.limites_sup[self.limites_sup == None] = +10**3
        
        # BUCLE PARA ASIGNAR UN VALOR A CADA UNA DE LAS VARIABLES QUE DEFINEN LA
        # POSICION
        # ----------------------------------------------------------------------
        for i in np.arange(self.n_variables):
        # Para cada posici�n, se genera un valor aleatorio dentro del rango
        # permitido para esa variable.
            self.posicion[i] = random.uniform(
                                    self.limites_inf[i],
                                    self.limites_sup[i]
                                )

        # LA VELOCIDAD INICIAL DE LA PARTICULA ES 0
        # ----------------------------------------------------------------------
        self.velocidad = np.repeat(0, self.n_variables)

        # INFORMACION DEL PROCESO (VERBOSE)
        # ----------------------------------------------------------------------
        if verbose:
            print("Nueva part�cula creada")
            print("----------------------")
            print("Posici�n: " + str(self.posicion))
            print("L�mites inferiores de cada variable: " \
                  + str(self.limites_inf))
            print("L�mites superiores de cada variable: " \
                  + str(self.limites_sup))
            print("Velocidad: " + str(self.velocidad))
            print("")

    def __repr__(self):
        """
        Informaci�n que se muestra cuando se imprime un objeto part�cula.

        """

        texto = "Part�cula" \
                + "\n" \
                + "---------" \
                + "\n" \
                + "Posici�n: " + str(self.posicion) \
                + "\n" \
                + "Velocidad: " + str(self.velocidad) \
                + "\n" \
                + "Mejor posici�n: " + str(self.mejor_posicion) \
                + "\n" \
                + "Mejor valor: " + str(self.mejor_valor) \
                + "\n" \
                + "L�mites inferiores de cada variable: " \
                + str(self.limites_inf) \
                + "\n" \
                + "L�mites superiores de cada variable: " \
                + str(self.limites_sup) \
                + "\n"

        return(texto)

    def evaluar_particula(self, funcion_objetivo, optimizacion, verbose = False):
        """
        Este m�todo eval�a una part�cula calculando el valor que toma la funci�n
        objetivo en la posici�n en la que se encuentra. Adem�s, compara si la
        nueva posici�n es mejor que las anteriores. Modifica los atributos
        valor, mejor_valor y mejor_posicion de la part�cula.
        
        Parameters
        ----------
        funcion_objetivo : `function`
            funci�n que se quiere optimizar.

        optimizacion : {'maximizar', 'minimizar'}
            dependiendo de esto, el mejor valor hist�rico de la part�cula ser�
            el mayor o el menor valor que ha tenido hasta el momento.

        verbose : `bool`, optional
            mostrar informaci�n del proceso por pantalla. (default is ``False``)
          
        Raises
        ------
        raise Exception
            si el argumento `optimizacion` es distinto de 'maximizar' o
            'minimizar'.

        Examples
        --------
        Ejemplo evaluar part�cula con una funci�n objetivo.

        >>> part = Particula(
                n_variables = 3,
                limites_inf = [-1,2,0],
                limites_sup = [4,10,20],
                verbose     = True
                )

        >>> def funcion_objetivo(x_0, x_1, x_2):
                f= x_0**2 + x_1**2 + x_2**2
                return(f)

        >>> part.evaluar_particula(
                funcion_objetivo = funcion_objetivo,
                optimizacion     = "maximizar",
                verbose          = True
                )

        """

        # COMPROBACIONES INICIALES: EXCEPTIONS Y WARNINGS
        # ----------------------------------------------------------------------
        if not optimizacion in ["maximizar", "minimizar"]:
            raise Exception(
                "El argumento optimizacion debe ser: 'maximizar' o 'minimizar'"
                )

        # EVALUACION DE LA FUNCION OBJETIVO EN LA POSICION ACTUAL
        # ----------------------------------------------------------------------
        self.valor = funcion_objetivo(*self.posicion)

        # MEJOR VALOR Y POSICION
        # ----------------------------------------------------------------------
        # Se compara el valor actual con el mejor valor hist�rico. La comparaci�n
        # es distinta dependiendo de si se desea maximizar o minimizar.
        # Si no existe ning�n valor hist�rico, se almacena el actual. Si ya 
        # existe alg�n valor hist�rico se compara con el actual y, de ser mejor 
        # este �ltimo, se sobrescribe.
        
        if self.mejor_valor is None:
            self.mejor_valor    = np.copy(self.valor)
            self.mejor_posicion = np.copy(self.posicion)
        else:
            if optimizacion == "minimizar":
                if self.valor < self.mejor_valor:
                    self.mejor_valor    = np.copy(self.valor)
                    self.mejor_posicion = np.copy(self.posicion)
            else:
                if self.valor > self.mejor_valor:
                    self.mejor_valor    = np.copy(self.valor)
                    self.mejor_posicion = np.copy(self.posicion)

        # INFORMACION DEL PROCESO (VERBOSE)
        # ----------------------------------------------------------------------
        if verbose:
            print("La part�cula ha sido evaluada")
            print("-----------------------------")
            print("Valor actual: " + str(self.valor))
            print("")

    def mover_particula(self, mejor_p_enjambre, inercia=0.8, peso_cognitivo=2,
                        peso_social=2, verbose=False):
        """
        Este m�todo ejecuta el movimiento de una part�cula, lo que implica
        actualizar su velocidad y posici�n. No se permite que la part�cula
        salga de la zona de b�squeda acotada por los l�mites.
        
        Parameters
        ----------
        mejor_p_enjambre : `np.narray`
            mejor posici�n de todo el enjambre.

        inercia : `float`, optional
            coeficiente de inercia. (default is 0.8)

        peso_cognitivo : `float`, optional
            coeficiente cognitivo. (default is 2)

        peso_social : `float`, optional
            coeficiente social. (default is 2)

        verbose : `bool`, optional
            mostrar informaci�n del proceso por pantalla. (default is ``False``)
          
        Examples
        --------
        Ejemplo mover part�cula.

        >>> part = Particula(
                n_variables = 3,
                limites_inf = [-1,2,0],
                limites_sup = [4,10,20],
                verbose     = True
                )

        >>> def funcion_objetivo(x_0, x_1, x_2):
                f= x_0**2 + x_1**2 + x_2**2
                return(f)

        >>> part.evaluar_particula(
                funcion_objetivo = funcion_objetivo,
                optimizacion     = "maximizar",
                verbose          = True
                )

        >>> part.mover_particula(
                mejor_p_enjambre = np.array([-1000,-1000,+1000]),
                inercia          = 0.8,
                peso_cognitivo   = 2,
                peso_social      = 2,
                verbose          = True
                )
       
        """

        # ACTUALIZACION DE LA VELOCIDAD
        # ----------------------------------------------------------------------
        componente_velocidad = inercia * self.velocidad
        r1 = np.random.uniform(low=0.0, high=1.0, size = len(self.velocidad))
        r2 = np.random.uniform(low=0.0, high=1.0, size = len(self.velocidad))
        componente_cognitivo = peso_cognitivo * r1 * (self.mejor_posicion \
                                                      - self.posicion)
        componente_social = peso_social * r2 * (mejor_p_enjambre \
                                                - self.posicion)
        nueva_velocidad = componente_velocidad + componente_cognitivo \
                          + componente_social
        self.velocidad = np.copy(nueva_velocidad)
        
        # ACTUALIZACION DE LA POSICION
        # ----------------------------------------------------------------------
        self.posicion = self.posicion + self.velocidad

        # COMPROBAR LIMITES
        # ----------------------------------------------------------------------
        # Se comprueba si alg�n valor de la nueva posici�n supera los l�mites
        # impuestos. En tal caso, se sobrescribe con el valor del l�mite
        # correspondiente y se reinicia a 0 la velocidad de la part�cula en esa
        # componente.
        for i in np.arange(len(self.posicion)):
            if self.posicion[i] < self.limites_inf[i]:
                self.posicion[i] = self.limites_inf[i]
                self.velocidad[i] = 0

            if self.posicion[i] > self.limites_sup[i]:
                self.posicion[i] = self.limites_sup[i]
                self.velocidad[i] = 0
                
        # INFORMACION DEL PROCESO (VERBOSE)
        # ----------------------------------------------------------------------
        if verbose:
            print("La part�cula se ha desplazado")
            print("-----------------------------")
            print("Nueva posici�n: " + str(self.posicion))
            print("")

#%%
################################################################################
#                              CLASE ENJAMBRE (SWARM)                          #
################################################################################

class Enjambre:
    """
    Esta clase crea un enjambre de n part�culas.

    Parameters
    ----------
    n_particulas :`int`
        n�mero de part�culas del enjambre.

    n_variables : `int`
        n�mero de variables que definen la posici�n de las part�culas.

    limites_inf : `list` or `numpy.ndarray`
        l�mite inferior de cada variable. Si solo se quiere predefinir l�mites
        de alguna variable, emplear ``None``. Los ``None`` ser�n remplazados por
        el valor (-10**3).

    limites_sup : `list` or `numpy.ndarray`
        l�mite superior de cada variable. Si solo se quiere predefinir l�mites
        de alguna variable, emplear ``None``. Los``None`` ser�n remplazados por
        el valor (+10**3).

    verbose : `bool`, optional
        mostrar informaci�n del proceso por pantalla. (default is ``False``)

    Attributes
    ----------
    particulas : `list`
        lista con todas las partículas del enjambre.
    
    n_particulas :`int`
        n�mero de part�culas del enjambre.

    n_variables : `int`
        n�mero de variables que definen la posici�n de las part�culas.

    limites_inf : `list` or `numpy.ndarray`
        l�mite inferior de cada variable.

    limites_sup : `list` or `numpy.ndarray`
        l�mite superior de cada variable.

    mejor_particula : `object particula`
        la mejor part�cula del enjambre en estado actual.

    mejor_valor : `floar`
        el mejor valor del enjambre en su estado actual.

    historico_particulas : `list`
        lista con el estado de las part�culas en cada una de las iteraciones que
        ha tenido el enjambre.

    historico_mejor_posicion : `list`
        lista con la mejor posici�n en cada una de las iteraciones que ha tenido
        el enjambre.

    historico_mejor_valor : `list`
        lista con el mejor valor en cada una de las iteraciones que ha tenido el
        enjambre.

    diferencia_abs : `list`
        diferencia absoluta entre el mejor valor de iteraciones consecutivas.

    resultados_df : `pandas.core.frame.DataFrame`
        dataframe con la informaci�n del mejor valor y posici�n encontrado en
        cada iteraci�n, as� como la mejora respecto a la iteraci�n anterior.

    valor_optimo : `float`
        mejor valor encontrado en todas las iteraciones.

    posicion_optima : `numpy.narray`
        posici�n donde se ha encontrado el valor_optimo.

    optimizado : `bool`
        si el enjambre ha sido optimizado.

    iter_optimizacion : `int`
        n�mero de iteraciones de optimizacion.

    Examples
    --------
    Ejemplo crear enjambre

    >>> enjambre = Enjambre(
               n_particulas = 5,
               n_variables  = 3,
               limites_inf  = [-5,-5,-5],
               limites_sup  = [5,5,5],
               verbose      = True
            )

    """

    def __init__(self, n_particulas, n_variables, limites_inf = None,
                 limites_sup = None, verbose = False):

        # N�mero de part�culas del enjambre
        self.n_particulas = n_particulas
        # N�mero de variables de cada part�cula
        self.n_variables = n_variables
        # L�mite inferior de cada variable
        self.limites_inf = limites_inf
        # L�mite superior de cada variable
        self.limites_sup = limites_sup
        # Lista de las part�culas del enjambre
        self.particulas = []
        # Etiqueta para saber si el enjambre ha sido optimizado
        self.optimizado = False
        # N�mero de iteraciones de optimizaci�n llevadas a cabo
        self.iter_optimizacion = None
        # Mejor part�cula del enjambre
        self.mejor_particula = None
        # Mejor valor del enjambre
        self.mejor_valor = None
        # Posici�n del mejor valor del enjambre.
        self.mejor_posicion = None
        # Estado de todas las part�culas del enjambre en cada iteraci�n.
        self.historico_particulas = []
        # Mejor posici�n en cada iteraci�n.
        self.historico_mejor_posicion = []
        # Mejor valor en cada iteraci�n.
        self.historico_mejor_valor = []
        # Diferencia absoluta entre el mejor valor de iteraciones consecutivas.
        self.diferencia_abs = []
        # data.frame con la informaci�n del mejor valor y posici�n encontrado en
        # cada iteraci�n, as� como la mejora respecto a la iteraci�n anterior.
        self.resultados_df = None
        # Mejor valor de todas las iteraciones
        self.valor_optimo = None
        # Mejor posici�n de todas las iteraciones
        self.posicion_optima = None
        self.realnum_iter = 0
        
        # CONVERSIONES DE TIPO INICIALES
        # ----------------------------------------------------------------------
        # Si limites_inf o limites_sup no son un array numpy, se convierten en
        # ello.
        if self.limites_inf is not None \
        and not isinstance(self.limites_inf,np.ndarray):
            self.limites_inf = np.array(self.limites_inf)

        if self.limites_sup is not None \
        and not isinstance(self.limites_sup,np.ndarray):
            self.limites_sup = np.array(self.limites_sup)

        # SE CREAN LAS PARTICULAS DEL ENJAMBRE Y SE ALMACENAN
        # ----------------------------------------------------------------------
        for i in np.arange(n_particulas):
            particula_i = Particula(
                            n_variables = self.n_variables,
                            limites_inf = self.limites_inf,
                            limites_sup = self.limites_sup,
                            verbose     = verbose
                          )
            self.particulas.append(particula_i)

        # INFORMACION DEL PROCESO (VERBOSE)
        # ----------------------------------------------------------------------
        if verbose:
            print("---------------")
            print("Enjambre creado")
            print("---------------")
            print("N�mero de part�culas: " + str(self.n_particulas))
            print("L�mites inferiores de cada variable: "
                  + str(self.limites_inf))
            print("L�mites superiores de cada variable: " \
                  + str(self.limites_sup))
            print("")

    def __repr__(self):
        """
        Informaci�n que se muestra cuando se imprime un objeto enjambre.

        """

        texto = "============================" \
                + "\n" \
                + "         Enjambre" \
                + "\n" \
                + "============================" \
                + "\n" \
                + "N�mero de part�culas: " + str(self.n_particulas) \
                + "\n" \
                + "L�mites inferiores de cada variable: " + str(self.limites_inf) \
                + "\n" \
                + "L�mites superiores de cada variable: " + str(self.limites_sup) \
                + "\n" \
                + "Optimizado: " + str(self.optimizado) \
                + "\n" \
                + "Iteraciones optimizaci�n: " + str(self.iter_optimizacion) \
                + "\n" \
                + "\n" \
                + "Informaci�n mejor part�cula:" \
                + "\n" \
                + "----------------------------" \
                + "\n" \
                + "Mejor posici�n actual: " + str(self.mejor_posicion) \
                + "\n" \
                + "Mejor valor actual: " + str(self.mejor_valor) \
                + "\n" \
                + "\n" \
                + "Resultados tras optimizar:" \
                + "\n" \
                + "----------------------------" \
                + "\n" \
                + "Posici�n �ptima: " + str(self.posicion_optima) \
                + "\n" \
                + "Valor �ptimo: " + str(self.valor_optimo)
                
        return(texto)

    def mostrar_particulas(self, n=None):
        """
        Este m�todo muestra la informaci�n de cada una de las n primeras 
        part�culas del enjambre.

        Parameters
        ----------

        n : `int`
            n�mero de particulas que se muestran. Si no se indica el valor
            (por defecto ``None``), se muestran todas. Si el valor es mayor
            que `self.n_particulas` se muestran todas.
        
        Examples
        --------
        >>> enjambre = Enjambre(
               n_particulas = 5,
               n_variables  = 3,
               limites_inf  = [-5,-5,-5],
               limites_sup  = [5,5,5],
               verbose      = True
            )

        >>> enjambre.mostrar_particulas(n = 1)

        """

        if n is None:
            n = self.n_particulas
        elif n > self.n_particulas:
            n = self.n_particulas

        for i in np.arange(n):
            print(self.particulas[i])
        return(None)

    def evaluar_enjambre(self, funcion_objetivo, optimizacion, verbose = False):
        """
        Este m�todo eval�a todas las part�culas del enjambre, actualiza sus
        valores e identifica la mejor part�cula.

        Parameters
        ----------
        funcion_objetivo : `function`
            funci�n que se quiere optimizar.

        optimizacion : {maximizar o minimizar}
            Dependiendo de esto, el mejor valor hist�rico de la part�cula ser�
            el mayor o el menor valor que ha tenido hasta el momento.

        verbose : `bool`, optional
            mostrar informaci�n del proceso por pantalla. (default is ``False``)
        
        Examples
        --------
        Ejemplo evaluar enjambre

        >>> enjambre = Enjambre(
               n_particulas = 5,
               n_variables  = 3,
               limites_inf  = [-5,-5,-5],
               limites_sup  = [5,5,5],
               verbose      = True
            )

        >>> def funcion_objetivo(x_0, x_1, x_2):
                f= x_0**2 + x_1**2 + x_2**2
                return(f)

        >>> enjambre.evaluar_enjambre(
                funcion_objetivo = funcion_objetivo,
                optimizacion     = "minimizar",
                verbose          = True
                )
        
        """

        # SE EVALUA CADA PARTICULA DEL ENJAMBRE
        # ----------------------------------------------------------------------
        for i in np.arange(self.n_particulas):
            self.particulas[i].evaluar_particula(
                funcion_objetivo = funcion_objetivo,
                optimizacion     = optimizacion,
                verbose          = verbose
                )

        # MEJOR PARTICULA DEL ENJAMBRE
        # ----------------------------------------------------------------------
        # Se identifica la mejor part�cula de todo el enjambre. Si se est�
        # maximizando, la mejor part�cula es aquella con mayor valor.
        # Lo contrario si se est� minimizando.

        # Se selecciona inicialmente como mejor part�cula la primera.
        self.mejor_particula =  copy.deepcopy(self.particulas[0])
        # Se comparan todas las part�culas del enjambre.
        for i in np.arange(self.n_particulas):
            if optimizacion == "minimizar":
                if self.particulas[i].valor < self.mejor_particula.valor:
                    self.mejor_particula = copy.deepcopy(self.particulas[i])
            else:
                if self.particulas[i].valor > self.mejor_particula.valor:
                    self.mejor_particula = copy.deepcopy(self.particulas[i])

        # Se extrae la posici�n y valor de la mejor part�cula y se almacenan
        # como mejor valor y posici�n del enjambre.
        self.mejor_valor    = self.mejor_particula.valor
        self.mejor_posicion = self.mejor_particula.posicion

        # INFORMACION DEL PROCESO (VERBOSE)
        # ----------------------------------------------------------------------
        if verbose:
            print("-----------------")
            print("Enjambre evaluado")
            print("-----------------")
            print("Mejor posici�n encontrada : " + str(self.mejor_posicion))
            print("Mejor valor encontrado : " + str(self.mejor_valor))
            print("")

    def mover_enjambre(self, inercia, peso_cognitivo, peso_social,
                       verbose = False):
        """
        Este m�todo mueve todas las part�culas del enjambre.

        Parameters
        ----------
        optimizacion : {'maximizar', 'minimizar'}
            si se desea maximizar o minimizar la funci�n.

        inercia : `float` or `int`
            coeficiente de inercia.

        peso_cognitivo : `float` or `int`
            coeficiente cognitivo.

        peso_social : `float` or `int`
            coeficiente social.

        verbose : `bool`, optional
            mostrar informaci�n del proceso por pantalla. (default is ``False``)
        
        """

        # Se actualiza la posici�n de cada una de las part�culas que forman el
        # enjambre.
        for i in np.arange(self.n_particulas):
            self.particulas[i].mover_particula(
                mejor_p_enjambre = self.mejor_posicion,
                inercia          = inercia,
                peso_cognitivo   = peso_cognitivo,
                peso_social      = peso_social,
                verbose          = verbose
            )

        # Informaci�n del proceso (VERBOSE)
        # ----------------------------------------------------------------------
        if verbose:
            print("---------------------------------------------------------" \
                  "------------")
            print("La posici�n de todas las part�culas del enjambre ha sido " \
                  "actualizada.")
            print("---------------------------------------------------------" \
            "------------")
            print("")


    def optimizar(self, funcion_objetivo, optimizacion, n_iteraciones = 50,
                  inercia = 0.8, reduc_inercia = True, inercia_max = 0.9,
                  inercia_min = 0.4, peso_cognitivo = 2, peso_social = 2,
                  parada_temprana = False, rondas_parada = None,
                  tolerancia_parada  = None, verbose = False):
        """
        Este m�todo realiza el proceso de optimizaci�n de un enjambre.

        Parameters
        ----------
        funcion_objetivo : `function`
            funci�n que se quiere optimizar.

        optimizacion : {'maximizar' o 'minimizar'}
            si se desea maximizar o minimizar la funci�n.

        m_iteraciones : `int` , optional
            numero de iteraciones de optimizaci�n. (default is ``50``)

        inercia : `float` or `int`, optional
            coeficiente de inercia. (default is ``0.8``)

        peso_cognitivo : `float` or `int`, optional
            coeficiente cognitivo. (default is ``2``)

        peso_social : `float` or `int`, optional
            coeficiente social. (default is ``2``)

        reduc_inercia: `bool`, optional
           activar la reducci�n del coeficiente de inercia. En tal caso, el
           argumento `inercia` es ignorado. (default is ``True``)

        inercia_max : `float` or `int`, optional
            valor inicial del coeficiente de inercia si se activa `reduc_inercia`.
            (default is ``0.9``)

        inercia_min : `float` or `int`, optional
            valor minimo del coeficiente de inercia si se activa `reduc_min`.
            (default is ``0.4``)

        parada_temprana : `bool`, optional
            si durante las �ltimas `rondas_parada` iteraciones la diferencia
            absoluta entre mejores part�culas no es superior al valor de 
            `tolerancia_parada`, se detiene el algoritmo y no se crean nuevas
            iteraciones. (default is ``False``)

        rondas_parada : `int`, optional
            n�mero de iteraciones consecutivas sin mejora m�nima para que se
            active la parada temprana. (default is ``None``)

        tolerancia_parada : `float` or `int`, optional
            valor m�nimo que debe tener la diferencia de iteraciones consecutivas
            para considerar que hay cambio. (default is ``None``)

         verbose : `bool`, optional
            mostrar informaci�n del proceso por pantalla. (default is ``False``)
        
        Raises
        ------
        raise Exception
            si se indica `parada_temprana = True` y los argumentos `rondas_parada`
            o `tolerancia_parada` son ``None``.

        raise Exception
            si se indica `reduc_inercia = True` y los argumentos `inercia_max`
            o `inercia_min` son ``None``.

        Examples
        --------
        Ejemplo optimizaci�n

        >>> def funcion_objetivo(x_0, x_1):
                # Para la regi�n acotada entre -10<=x_0<=0 y -6.5<=x_1<=0 la 
                # funci�n tiene m�ltiples m�nimos locales y un �nico m�nimo 
                # global en f(-3.1302468,-1.5821422)= -106.7645367.
                f = np.sin(x_1)*np.exp(1-np.cos(x_0))**2 \
                    + np.cos(x_0)*np.exp(1-np.sin(x_1))**2 \
                    + (x_0-x_1)**2
                return(f)

        >>> enjambre = Enjambre(
                        n_particulas = 50,
                        n_variables  = 2,
                        limites_inf  = [-10, -6.5],
                        limites_sup  = [0, 0],
                        verbose      = False
                        )

        >>> enjambre.optimizar(
                funcion_objetivo  = funcion_objetivo,
                optimizacion      = "minimizar",
                n_iteraciones     = 250,
                inercia           = 0.8,
                reduc_inercia     = True,
                inercia_max       = 0.9,
                inercia_min       = 0.4,
                peso_cognitivo    = 1,
                peso_social       = 2,
                parada_temprana   = True,
                rondas_parada     = 5,
                tolerancia_parada = 10**-3,
                verbose           = False
            )

        """

        # COMPROBACIONES INICIALES: EXCEPTIONS Y WARNINGS
        # ----------------------------------------------------------------------
        # Si se activa la parada temprana, hay que especificar los argumentos
        # rondas_parada y tolerancia_parada.
        if parada_temprana \
        and (rondas_parada is None or tolerancia_parada is None):
            raise Exception(
                "Para activar la parada temprana es necesario indicar un " \
                + " valor de rondas_parada y de tolerancia_parada."
                )
        
        # Si se activa la reducci�n de inercia, hay que especificar los argumentos
        # inercia_max y inercia_min.
        if reduc_inercia \
        and (inercia_max is None or inercia_min is None):
            raise Exception(
            "Para activar la reducci�n de inercia es necesario indicar un " \
            + "valor de inercia_max y de inercia_min."
            )

        # ITERACIONES
        # ----------------------------------------------------------------------
        self.realnum_iter=n_iteraciones
        start = time.time()

        for i in np.arange(n_iteraciones):
            if verbose:
                print("-------------")
                print("Iteraci�n: " + str(i))
                print("-------------")
            
            # EVALUAR PARTICULAS DEL ENJAMBRE
            # ------------------------------------------------------------------
            self.evaluar_enjambre(
                funcion_objetivo = funcion_objetivo,
                optimizacion     = optimizacion,
                verbose          = verbose
                )

            # SE ALMACENA LA INFORMACION DE LA ITERACION EN LOS HISTORICOS
            # ------------------------------------------------------------------
            self.historico_particulas.append(copy.deepcopy(self.particulas))
            self.historico_mejor_posicion.append(copy.deepcopy(self.mejor_posicion))
            self.historico_mejor_valor.append(copy.deepcopy(self.mejor_valor))

            # SE CALCULA LA DIFERENCIA ABSOLUTA RESPECTO A LA ITERACION ANTERIOR
            # ------------------------------------------------------------------
            # La diferencia solo puede calcularse a partir de la segunda
            # iteraci�n.
            if i == 0:
                self.diferencia_abs.append(None)
            else:
                diferencia = abs(self.historico_mejor_valor[i] \
                                 - self.historico_mejor_valor[i-1])
                self.diferencia_abs.append(diferencia)

            # CRITERIO DE PARADA
            # ------------------------------------------------------------------
            # Si durante las �ltimas n iteraciones, la diferencia absoluta entre
            # mejores part�culas no es superior al valor de tolerancia_parada,
            # se detiene el algoritmo y no se crean nuevas iteraciones.
            if parada_temprana and i > rondas_parada:
                ultimos_n = np.array(self.diferencia_abs[-(rondas_parada): ])
                if all(ultimos_n < tolerancia_parada):
                    print("Algoritmo detenido en la iteraci�n " 
                          + str(i) \
                          + " por falta cambio absoluto m�nimo de " \
                          + str(tolerancia_parada) \
                          + " durante " \
                          + str(rondas_parada) \
                          + " iteraciones consecutivas.")
                    self.realnum_iter=i
                    break
            
            # MOVER PARTICULAS DEL ENJAMBRE
            # ------------------------------------------------------------------
            # Si se ha activado la reducci�n de inercia, se recalcula su valor 
            # para la iteraci�n actual.
            if reduc_inercia:
                inercia = ((inercia_max - inercia_min) \
                          * (n_iteraciones-i)/n_iteraciones) \
                          + inercia_min
           
            self.mover_enjambre(
               inercia        = inercia,
               peso_cognitivo = peso_cognitivo,
               peso_social    = peso_social,
               verbose        = False
            )

        end = time.time()
        self.optimizado = True
        self.iter_optimizacion = i
        
        # IDENTIFICACION DEL MEJOR PARTICULA DE TODO EL PROCESO
        # ----------------------------------------------------------------------
        if optimizacion == "minimizar":
            indice_valor_optimo=np.argmin(np.array(self.historico_mejor_valor))
        else:
            indice_valor_optimo=np.argmax(np.array(self.historico_mejor_valor))

        self.valor_optimo    = self.historico_mejor_valor[indice_valor_optimo]
        self.posicion_optima = self.historico_mejor_posicion[indice_valor_optimo]
        
        # CREACION DE UN DATAFRAME CON LOS RESULTADOS
        # ----------------------------------------------------------------------
        self.resultados_df = pd.DataFrame(
            {
            "mejor_valor_enjambre"   : self.historico_mejor_valor,
            "mejor_posicion_enjambre": self.historico_mejor_posicion,
            "diferencia_abs"         : self.diferencia_abs
            }
        )
        self.resultados_df["iteracion"] = self.resultados_df.index
        
        print("-------------------------------------------")
        print("Optimizaci�n finalizada " \
              + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        print("-------------------------------------------")
        print("Duraci�n optimizaci�n: " + str(end - start))
        print("Número de iteraciones: " + str(self.iter_optimizacion))
        print("Posici�n �ptima: " + str(self.posicion_optima))
        print("Valor �ptimo: " + str(self.valor_optimo))
        print("")