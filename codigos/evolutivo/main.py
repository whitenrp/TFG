#!pip install pgmpy
#!pip install psutil
import psutil
import tracemalloc
import time
import sys
from pgmpy.estimators import PC, CITests
from itertools import combinations
import random
import argparse
import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import pearsonr, spearmanr


def parse_args():
    """Función para obtener los argumentos introducidos por línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Programa que acepta parámetros opcionales y obligatorios."
    )

    # Argumentos obligatorios
    parser.add_argument("dataset", help="Conjunto de datos para detectar la inferencia causal.")
    parser.add_argument("tipo_variable", help="Tipo de variable \"categoric\" o \"numeric\"")

    # Argumentos opcionales con valores por defecto
    parser.add_argument("--tamano_poblacion", type=int, default=10, help="Tamaño de la población (por defecto: 10)")
    parser.add_argument("--n_elementos_p_inicial", type=int, default=3, help="Número de elementos en la población inicial (por defecto: 3)")
    parser.add_argument("--n_max_epocs", type=int, default=50, help="Número máximo de épocas (por defecto: 50)")
    parser.add_argument("--valor_suficiente_pvalue", type=float, default=0.01, help="Valor de p-value suficiente (por defecto: 0.01)")
    parser.add_argument("--probabilidad_cruce", type=float, default=0.5, help="Probabilidad de cruce (por defecto: 0.5)")
    parser.add_argument("--output", type=str, default="resultado.txt", help="Nombre del fichero de salida (por defecto: resultado.txt)")
    parser.add_argument("--estricto", type="store_true", deldault="False", help="Si se activa, fuerza valor_suficiente_pvalue a 1e-8 para pruebas estrictas")

    return parser.parse_args()


def validar_dataset(dataset_path, tipo_variable):
    """
    Comprueba que el datasete se puede leer y que sus variables son solo continuas para los conjuntos continuos
    :param dataset_path: nombre del conjunto de datos a usar
    :tipo_variable: verifica la restricción de no categóricos para conjuntos continuos
    """
    try:
        df = pd.read_csv(dataset_path)
    except Exception as e:
        print(f"El dataset no se ha cargado correctamente: {e}")
        sys.exit(1)

    if tipo_variable == "numeric":
        # Detectar columnas categóricas
        columnascategoricas = df.select_dtypes(include=["object", "category"]).columns
        if len(columnascategoricas) != 0:
            print("El dataset Contiene variables categóricas, NO se podrá proceder con la ejecución.")
            sys.exit(1)


class Individuo:
    """
    Representa los distintos individuos de la poblacion, cada individuo posee unos elementos que representan
    el nombre de las columnas que posee y una puntuacion que representa las relaciones no independientes que
    posee el individuo entre sus elementos
    :param self : el individuo al cual vamos a inicializar
    :param elementos: los distintos elementos que contendrá el individuo
    """
    def __init__(self, elementos):
        self.elementos = elementos
        self.puntuacion = 0

    def __repr__(self):
        """
        funcion para mostrar los Individuos de la poblacion
        :param self: individuo a mostrar datos
        """
        return f'Individuo(elementos={self.elementos}, puntuacion={self.puntuacion})'

    def add_elemento(self, valor):
        """
        funcion para añadir elementos a un individuo concreto de la población
        :param self: individuo al cual queremos añadir valores a elemetnos
        :param valor: valor a añadir en lso elementos del individuo
        """
        self.elementos.append(valor)

def cruce(individuo1, individuo2):
    """
    Operación de cruce entre ambos individuos de la población
    :param individuo1: primer individuo seleccionado para la operación de cruce
    :param individuo2: segundo individuo seleccionado para la operación de cruce
    """
    set1 = set(individuo1.elementos)
    set2 = set(individuo2.elementos)
    libre1 = set1-set2
    libre2 = set2-set1
    num_intercambio = min(len(libre1), len(libre2))

    if num_intercambio < 2:
        return individuo1, individuo2
    else:
        cantidad_a_cruzar = random.randint(1, num_intercambio-1)
        libre1 = random.sample(list(libre1), cantidad_a_cruzar)
        libre2 = random.sample(list(libre2), cantidad_a_cruzar)
        hijo1 = Individuo(list(set1-set(libre1)))
        hijo2 = Individuo(list(set2-set(libre2)))
        hijo1.elementos.extend(libre2)
        hijo2.elementos.extend(libre1)

    return hijo1, hijo2

def ordenar_poblacion(nombres_columnas, poblacion):
    """
    ordenar población cambiará el orden de cada uno de los elementos que
    poseen los individuos de la población para que se encuentren ordenados
    en base al orden de columnas de los datos.
    :param nombres_columnas: nombre de las columnas con el orden que aparece en el dataset
    :param poblacion: la población que esta utilizando el algoritmo y la cual queremos ordenar los elementos
        de sus individuos
    """
    for individuo in poblacion:
        individuo.elementos = ordenar_elementos(nombres_columnas, individuo.elementos)
    return poblacion

def ordenar_elementos(nombres_columnas, elementos):
    """
    Toma los elementos de los individuos y los ordena acorde al orden de nombres_columnas, despues los retorna
    :param nombres_columnas: nombre de las columnas con el orden que aparece en el dataset
    :param elementos: los elementos que posee un individuo.
    """

    var_columns = np.array(nombres_columnas)
    indices = np.array([np.where(var_columns == i)[0][0] for i in elementos if i in var_columns])
    sorted_indices = np.argsort(indices)
    return [elementos[i] for i in sorted_indices]

def dar_puntuacion(individuo,matriz,nombres_columnas,diccionario):
    """
    comprobamos cada par de elementos en un individuo y si su valor en la matriz es 0 para ese par,
    en caso de ser positivo el individuo obtendría un punto, en caso negativo no obtendría puntuación.
    :param individuo: el individuo al que vamos a dar puntuación
    :param matriz: la matriz de independencia, tiene 1 si ese par de variables es independiente y 0 si 
        no existen evidencias para considerarlos independientes
    :param nombres_columnas: nombre de las columnas con el orden que aparece en el dataset
    """
    puntos=0
    for i in range(len(individuo.elementos)-1):
        x = individuo.elementos[i]
        for j in range(i+1, len(individuo.elementos)):
            y = individuo.elementos[j]
            indices_x = np.where(np.array(nombres_columnas) == x)[0]
            indices_y = np.where(np.array(nombres_columnas) == y)[0]
            if indices_x.size > 0 and indices_y.size > 0:
                pos_x, pos_y = np.sort([indices_x[0], indices_y[0]])
                if matriz[pos_x,pos_y] != 1:
                  puntos +=1
            else:
                print("Los indices no existen")
    return puntos

def asignar_unos_vecinos(matriz_ind,independent_pairs,nombres_columnas,vecinos_dic):
    """
    para cada par de elementos en independent_pairs, se busca su posición en la matriz
    y se le asigna el valor 1, indicando que existen evidencias de que ese par es
    independiente, a continuación eliminamos los elementos del diccionario de vecinos para evitar
    procesamientos futuros.
    :param matriz_ind: la matriz de independencia, tiene 1 si ese par de variables es independiente y 0 si 
        no existen evidencias para considerarlos independientes
    :param independent_pairs: lista con cada una de las parejas de elementos detectadas como independientes
    :param nombres_columnas: nombre de las columnas con el orden que aparece en el dataset
    :param vecinos_dic: el diccionario de vecinos, que nos indica las variables o elementos con los cuales
        sigue teniendo relacion la variable.
    """
    for pair in independent_pairs:
        x, y = pair
        indices_x = np.where(np.array(nombres_columnas) == x)[0]
        indices_y = np.where(np.array(nombres_columnas) == y)[0]
        pos_x, pos_y = np.sort([indices_x[0], indices_y[0]])
        matriz_ind[pos_x,pos_y] = 1
        if y in vecinos_dic[x]:
            vecinos_dic[x].remove(y)
        if x in vecinos_dic[y]:
            vecinos_dic[y].remove(x)

def calculate_lift(dataset, columns):
    """
    Se comprueba la correlación existente entre cada una de las variables
    de nuestro problema, si detectamos correlación entre las variables 
    puede existir inferencia causal entre ellas, en caso negativo no habrá
    relaciones causales
    :param dataset: el dataset utilizado como problema
    :param columns: nombre de las columnas con el orden que aparece en el dataset
    """
    lifts_between_features = pd.DataFrame(index=columns, columns=columns)
    for index_1, var_1 in enumerate(columns):
        for index_2, var_2 in enumerate(columns[index_1 + 1:]):
            p_var_1 = dataset[var_1].value_counts(normalize=True)
            p_var_2 = dataset[var_2].value_counts(normalize=True)
            p_var_1_and_var_2 = pd.crosstab(dataset[var_1], dataset[var_2], normalize='all')
            p_var1_array = p_var_1.to_numpy()[:, None]
            p_var2_array = p_var_2.to_numpy()[None, :]
            lift = p_var_1_and_var_2 / (p_var1_array * p_var2_array)
            lifts_between_features.loc[var_1, var_2] = round(lift.mean().mean(), 2)

    return lifts_between_features

def generate_population(tamano_poblacion, nombres_columnas, n_elementos_p_inicial):
    """
    Generamos la población inical junto con sus elementos, posteriormente se devuelve
    la población
    :param tamano_poblacion: el valor de tamaño de población inicial
    :param nombres_columnas: nombre de las columnas con el orden que aparece en el dataset
    :param n_elementos_p_inicial: el numero de elementos que tendrá cada individuo de la población
    """
    poblacion = []
    nombres_columnas_disponibles = nombres_columnas.copy()
    for i in range(tamano_poblacion):
        seleccionados = random.sample(nombres_columnas_disponibles, n_elementos_p_inicial)
        individuo = Individuo(seleccionados)
        poblacion.append(individuo)

    return poblacion


def actualizar_matriz_cat_vecinos(elementos, nombres_columnas, matrix_ind, profundidad, diccionario, dataset,
                      valor_suficiente_pvalue, independent_pairs, vecinos_dic):
    """
    vamos seleccionando dos a dos cada uno de los elementos de nuestro individuo para cada par comprobamos
    si su valor en la matriz es distinto de 0 y los valores comunes entre sus vecinos del mismo individuo,
    si para ese par de variables no se ha probado un conjunto de elementos, se realiza la prueba de independencia
    y se registra el/los elementos probados para ese par en su diccionario correspondiente, en caso de que el
    valor de la prueba de independencia sea superior al valor de significancia, el par de variables es
    incluido en independent_pairs para su procesamento posterior, en caso negativo se continúa probando variables
    para ese par hasta que se hayan cubierto todas las posibilidades.
    :param elementos: conjunto de elementos que posee un individuo
    :param nombres_columnas: nombre de las columnas con el orden que aparece en el dataset
    :param matrix_ind: la matriz de independencia, tiene 1 si ese par de variables es independiente y 0 si 
        no existen evidencias para considerarlos independientes
    :param profundidad: nivel de profundidad o tamaño del conjunto mínimo separador que vamos a probar
        para intentar detectar independencia entre las variables
    :param diccionario: el diccionario que contiene las variables o conjunto de variables sobre las cuales hemos
        realizado pruebas de independencia para cada par de variables, se guarda registro para evitar repetir pruebas
    :param dataset: el dataset utilizado como problema
    :param valor_suficiente_pvalue: el valor sobre el cual consideramos la independencia entre variables
    :param independent_pairs: lista sobre la cual guardaremos todo par de variables detectados como independientes
    :param vecinos_dic: diccionario que almacena los elementos sobre los cuales no hay evidencias para considerar
        independientes, respecto al elemento seleccionado
    """
    for i, pattern_x in enumerate(elementos[:-1]):
        for j, pattern_y in enumerate(elementos[i+1:]):
            if pattern_x != pattern_y:
                indices_x = np.where(np.array(nombres_columnas) == pattern_x)[0]
                pair_elements = (pattern_x, pattern_y)
                indices_y = np.where(np.array(nombres_columnas) == pattern_y)[0]
                if indices_x.size > 0 and indices_y.size > 0:
                    pos_x, pos_y = np.sort([indices_x[0], indices_y[0]])
                    if matrix_ind[pos_x, pos_y] == 0:
                        if profundidad == 0:
                            if "nule" not in diccionario.get(pair_elements, []):
                                diccionario[pair_elements].append("nule")
                                if CITests.chi_square(X=pattern_x, Y=pattern_y, Z=[], data=dataset, boolean=True, significance_level=valor_suficiente_pvalue):
                                    independent_pairs.append(pair_elements)
                        else:
                            if profundidad == 1:
                                algo = set(elementos) - set(pair_elements)
                                listos = (set(vecinos_dic[pattern_x]) | set(vecinos_dic[pattern_y])) & algo
                                combinaciones_faltantes = set(listos) - set(diccionario[pair_elements])
                                if len(combinaciones_faltantes) != 0:
                                    for combinacion in combinaciones_faltantes:
                                        diccionario[pair_elements].append(combinacion)
                                        if CITests.chi_square(X=pattern_x, Y=pattern_y, Z=[combinacion], data=dataset, boolean=True, significance_level=valor_suficiente_pvalue):
                                            independent_pairs.append(pair_elements)
                                            break
                            else:
                                algo = set(elementos) - set(pair_elements)
                                listos = (set(vecinos_dic[pattern_x]) | set(vecinos_dic[pattern_y])) & algo
                                completo = [i for i in combinations(listos, profundidad)]
                                combinaciones_faltantes = set(completo) - set(diccionario[pair_elements])
                                if len(combinaciones_faltantes) != 0:
                                    for combinacion in combinaciones_faltantes:
                                        diccionario[pair_elements].append(combinacion)
                                        if CITests.chi_square(X=pattern_x, Y=pattern_y, Z=combinacion, data=dataset, boolean=True, significance_level=valor_suficiente_pvalue):
                                            independent_pairs.append(pair_elements)
                                            break


def actualizar_matriz_cont(elementos, nombres_columnas, matrix_ind, profundidad, diccionario, dataset,
                      valor_suficiente_pvalue, independent_pairs, vecinos_dic):
    """
    vamos seleccionando dos a dos cada uno de los elementos de nuestro individuo para cada par comprobamos
    si su valor en la matriz es distinto de 0 y los valores comunes entre sus vecinos del mismo individuo,
    si para ese par de variables no se ha probado un conjunto de elementos, se realiza la prueba de independencia
    y se registra el/los elementos probados para ese par en su diccionario correspondiente, en caso de que el
    valor de la prueba de independencia sea superior al valor de significancia, el par de variables es
    incluido en independent_pairs para su procesamento posterior, en caso negativo se continúa probando variables
    para ese par hasta que se hayan cubierto todas las posibilidades.
    :param elementos: conjunto de elementos que posee un individuo
    :param nombres_columnas: nombre de las columnas con el orden que aparece en el dataset
    :param matrix_ind: la matriz de independencia, tiene 1 si ese par de variables es independiente y 0 si 
        no existen evidencias para considerarlos independientes
    :param profundidad: nivel de profundidad o tamaño del conjunto mínimo separador que vamos a probar
        para intentar detectar independencia entre las variables
    :param diccionario: el diccionario que contiene las variables o conjunto de variables sobre las cuales hemos
        realizado pruebas de independencia para cada par de variables, se guarda registro para evitar repetir pruebas
    :param dataset: el conjunto de datos utilizado como problema
    :param valor_suficiente_pvalue: el valor sobre el cual consideramos la independencia entre variables
    :param independent_pairs: lista sobre la cual guardaremos todo par de variables detectados como independientes
    :param vecinos_dic: diccionario que almacena los elementos sobre los cuales no hay evidencias para considerar
        independientes, respecto al elemento seleccionado
    """
    for i, pattern_x in enumerate(elementos[:-1]):
        for j, pattern_y in enumerate(elementos[i+1:]):
            if pattern_x != pattern_y:
              indices_x = np.where(np.array(nombres_columnas) == pattern_x)[0]
              pair_elements = (pattern_x, pattern_y)
              indices_y = np.where(np.array(nombres_columnas) == pattern_y)[0]
              if indices_x.size > 0 and indices_y.size > 0:
                  pos_x, pos_y = np.sort([indices_x[0], indices_y[0]])
                  if matrix_ind[pos_x, pos_y] == 0:
                      if profundidad == 0:
                          if "nule" not in diccionario.get(pair_elements, []):
                              diccionario[pair_elements].append("nule")
                              corr, p_value = pearsonr(dataset[pattern_x], dataset[pattern_y])
                              if p_value > valor_suficiente_pvalue:
                                  independent_pairs.append((pattern_x, pattern_y))
                      else:
                          algo = set(elementos) - set(pair_elements)
                          listos = (set(vecinos_dic[pattern_x]) | set(vecinos_dic[pattern_y])) & algo
                          completo = [i for i in combinations(listos, profundidad)]
                          combinaciones_faltantes = set(completo) - set(diccionario[pair_elements])
                          if len(combinaciones_faltantes) != 0:
                              for combinacion in combinaciones_faltantes:
                                  diccionario[(pattern_x,pattern_y)].append(combinacion)
                                  if CITests.pearsonr(pattern_x,pattern_y,combinacion,dataset,boolean=True,significance_level=valor_suficiente_pvalue):
                                      independent_pairs.append((pattern_x, pattern_y))
                                      break


def seleccion_por_torneo(poblacion, tam_torneo=3, num_padres=None):

    if num_padres is None:
        num_padres = len(poblacion)

    padres = []
    for _ in range(num_padres):
        competidores = random.sample(poblacion, tam_torneo)
        ganador = max(competidores, key=lambda ind: ind.puntuacion)
        padres.append(ganador)

    return padres

#def run_genetic_categorico(dataset, tamano_poblacion = 10, n_elementos_p_inicial=3,n_max_epocs = 50, valor_suficiente_pvalue= 0.01, probabilidad_cruce = 0.5):
def run_genetic_categorico(dataset, tamano_poblacion, n_elementos_p_inicial, n_max_epocs, valor_suficiente_pvalue, probabilidad_cruce, output):
    """
    funcion principal para variables categóricas
    :param dataset: el conjunto de datos utilizado como problema
    :param tamano_poblacion: el número de individuos que formarán nuestra población
    :param n_elementos_p_inicial: el número de elementos que tendrá cada uno de los individuos de nuestra población
    :param n_max_epocs: el número máximo de épocas que realizará el algoritmo evolutivo
    :param valor_suficiente_pvalue: el valor de suficiencia a partir del cual consideraremos una relacion entre
        variables como independiente
    :param probabilidad_cruce: la probabilidad de cruce entre los individuos de la población
    :param output: nombre del fichero de salida donde se guardarán los resultados
    """

    contador, maximo_valor_individuos, auxiliar, contenido_elementos_individuos = 0, 0, 0, 1
    df = dataset.copy()
    df.columns = df.columns.str.strip()
    nombres_columnas = df.columns.to_list()
    set_nombres_columnas = set(nombres_columnas)
    nvariables = df.shape[1]
    matriz_ind = np.zeros((nvariables, nvariables))

    combinaciones = list(combinations(nombres_columnas, 2))
    lifts_between_variables = calculate_lift(df, nombres_columnas)
    vecinos_dic = {var1: lifts_between_variables.loc[var1][lifts_between_variables.loc[var1] != 1].index.tolist() for
                   var1 in lifts_between_variables.index}
    for i, var1 in enumerate(nombres_columnas):
        for j, var2 in enumerate(nombres_columnas[i + 1:]):
            if lifts_between_variables.loc[var1, var2] < 1.2:
                matriz_ind[i, j] = 1

    diccionario = {comb: [] for comb in combinaciones}
    poblacion = generate_population(tamano_poblacion, nombres_columnas, n_elementos_p_inicial)

    for epoch in range(n_max_epocs):
        contador += 1
        if contador == 5:
            break
        auxiliar = 0
        independent_pairs = []

        poblacion = ordenar_poblacion(nombres_columnas, poblacion)
        for profundidad in range(0, contenido_elementos_individuos):
            independent_pairs = []
            for individuo in poblacion:
                detectado = individuo.elementos
                actualizar_matriz_cat_vecinos(detectado, nombres_columnas, matriz_ind, profundidad, diccionario, dataset,
                      valor_suficiente_pvalue, independent_pairs, vecinos_dic)

            if len(independent_pairs) != 0:
                asignar_unos_vecinos(matriz_ind, independent_pairs, nombres_columnas, vecinos_dic)

        for individuo in poblacion:
            individuo.puntuacion = dar_puntuacion(individuo, matriz_ind, nombres_columnas, diccionario)
            if maximo_valor_individuos < individuo.puntuacion:
                maximo_valor_individuos = individuo.puntuacion
                contador = 0

        nueva_poblacion = []
        random.shuffle(poblacion)
        # CRUCE
        for i in range(0, len(poblacion)-1, 2):
            padre1 = poblacion[i]
            padre2 = poblacion[i + 1]
            if probabilidad_cruce > random.random():
                hijo1, hijo2 = cruce(padre1, padre2)
                nueva_poblacion.append(hijo1)
                nueva_poblacion.append(hijo2)
            else:
                nueva_poblacion.append(padre1)
                nueva_poblacion.append(padre2)
        del poblacion

        # MUTACION
        for individuo in nueva_poblacion:
            libres = set_nombres_columnas - set(individuo.elementos)
            if len(libres) > 1:
                if auxiliar == 0 & contenido_elementos_individuos < 5:
                    contenido_elementos_individuos += 1
                    auxiliar = 1
                individuo.add_elemento(random.choice(list(libres)))
                individuo.add_elemento(random.choice(list(libres)))
            elif len(libres) == 1 :
                if auxiliar == 0 & contenido_elementos_individuos < 5:
                    contenido_elementos_individuos += 1
                    auxiliar = 1
                individuo.add_elemento(random.choice(list(libres)))
        poblacion = nueva_poblacion
    contenido_elementos_individuos = 5
    for profundidad in range(contenido_elementos_individuos):
        independent_pairs = []

        detectado = nombres_columnas
        actualizar_matriz_cat_vecinos(detectado, nombres_columnas, matriz_ind, profundidad, diccionario, dataset,
                            valor_suficiente_pvalue, independent_pairs, vecinos_dic)

        if len(independent_pairs) != 0:
            asignar_unos_vecinos(matriz_ind, independent_pairs, nombres_columnas, vecinos_dic)

    relaciones_cero = []
    for i in range(len(nombres_columnas) - 1):
        for j in range(i + 1, len(nombres_columnas)):
            if matriz_ind[i, j] != 1:  # Verificar si el valor es 0
                relacion = (nombres_columnas[i], nombres_columnas[j])
                relaciones_cero.append(relacion)

    print(f"\nExisten {len(relaciones_cero)} relaciones:")
    print(relaciones_cero)

    with open(output, "w") as f:  # `output` viene de args.output
        f.write(f"Existen {len(relaciones_cero)} relaciones:\n")
        for r in relaciones_cero:
            f.write(f"{r[0]} -- {r[1]}\n")

def run_genetic_numeric(dataset, tamano_poblacion, n_elementos_p_inicial, n_max_epocs, valor_suficiente_pvalue, probabilidad_cruce, output):
    """
    funcion principal para variables continuas
    :param dataset: el conjunto de datos utilizado como problema
    :param tamano_poblacion: el número de individuos que formarán nuestra población
    :param n_elementos_p_inicial: el número de elementos que tendrá cada uno de los individuos de nuestra población
    :param n_max_epocs: el número máximo de épocas que realizará el algoritmo evolutivo
    :param valor_suficiente_pvalue: el valor de suficiencia a partir del cual consideraremos una relacion entre
        variables como independiente
    :param probabilidad_cruce: la probabilidad de cruce entre los individuos de la población
    :param output: nombre del fichero de salida donde se guardarán los resultados
    """
    contador, maximo_valor_individuos, auxiliar, contenido_elementos_individuos = 0, 0, 0, 1
    df = dataset.copy()
    df.columns = df.columns.str.strip()
    nombres_columnas = df.columns.to_list()
    set_nombres_columnas = set(nombres_columnas)
    nvariables = df.shape[1]  # shape[0] filas, shape [1] columnas
    matriz_ind = np.zeros((nvariables, nvariables))
    combinaciones = list(combinations(nombres_columnas, 2))
    diccionario = {comb: [] for comb in combinaciones}
    vecinos_dic = {columna: [c for c in nombres_columnas if c != columna] for columna in nombres_columnas}
    poblacion = generate_population(tamano_poblacion, nombres_columnas, n_elementos_p_inicial)

    for epoch in range(n_max_epocs):
        contador += 1
        if contador == 5:
            break
        auxiliar = 0
        independent_pairs = []

        poblacion = ordenar_poblacion(nombres_columnas, poblacion)

        for profundidad in range(0, contenido_elementos_individuos):
            independent_pairs = []
            for individuo in poblacion:
                detectado = individuo.elementos
                actualizar_matriz_cont(detectado, nombres_columnas, matriz_ind, profundidad, diccionario, dataset,
                      valor_suficiente_pvalue, independent_pairs, vecinos_dic)

            if len(independent_pairs) != 0:
                asignar_unos_vecinos(matriz_ind, independent_pairs, nombres_columnas, vecinos_dic)


        for individuo in poblacion:
            individuo.puntuacion = dar_puntuacion(individuo, matriz_ind, nombres_columnas, diccionario)
            if maximo_valor_individuos < individuo.puntuacion:
                maximo_valor_individuos = individuo.puntuacion
                contador = 0


        nueva_poblacion = []

        random.shuffle(poblacion)
        # CRUCE
        for i in range(0, len(poblacion)-1, 2):
            padre1 = poblacion[i]
            padre2 = poblacion[i + 1]
            if probabilidad_cruce > random.random():
                hijo1, hijo2 = cruce(padre1, padre2)
                nueva_poblacion.append(hijo1)
                nueva_poblacion.append(hijo2)
            else:
                nueva_poblacion.append(padre1)
                nueva_poblacion.append(padre2)
        del poblacion

        # MUTACION
        for individuo in nueva_poblacion:
            libres = set_nombres_columnas - set(individuo.elementos)
            if len(libres) > 1:
                if auxiliar == 0 & contenido_elementos_individuos < 5:
                    contenido_elementos_individuos += 1
                    auxiliar = 1
                individuo.add_elemento(random.choice(list(libres)))
                individuo.add_elemento(random.choice(list(libres)))
            elif len(libres) == 1:
                if auxiliar == 0 & contenido_elementos_individuos < 5:
                    contenido_elementos_individuos += 1
                    auxiliar = 1
                individuo.add_elemento(random.choice(list(libres)))

        poblacion = nueva_poblacion

    contenido_elementos_individuos = 5

    for profundidad in range(contenido_elementos_individuos):
        independent_pairs = []

        detectado = nombres_columnas
        actualizar_matriz_cont(detectado, nombres_columnas, matriz_ind, profundidad, diccionario, dataset,
                            valor_suficiente_pvalue, independent_pairs, vecinos_dic)

        if len(independent_pairs) != 0:
            asignar_unos_vecinos(matriz_ind, independent_pairs, nombres_columnas, vecinos_dic)

    relaciones_cero = []
    for i in range(len(nombres_columnas) - 1):
        for j in range(i + 1, len(nombres_columnas)):
            if matriz_ind[i, j] != 1:  # Verificar si el valor es 0
                relacion = (nombres_columnas[i], nombres_columnas[j])
                relaciones_cero.append(relacion)

    print(f"\nExisten {len(relaciones_cero)} relaciones:")
    print(relaciones_cero)

    with open(output, "w") as f:  # `output` viene de args.output
        f.write(f"Existen {len(relaciones_cero)} relaciones:\n")
        for r in relaciones_cero:
            f.write(f"{r[0]} -- {r[1]}\n")


def main():
    """
    funcion main, recibe el dataset, el tipo de variables que contiene y configuración de parámetros a través de parse_args,
     si el tipo de variable coincide con categoric o numeric se procede con el algoirtmo.
    """ 

    args = parse_args()

    df = validar_dataset(args.dataset, args.tipo_variable)

    # Aquí va tu código principal usando los argumentos
    #print("Dataset:", args.dataset)
    #print("Tipo de variable:", args.tipo_variable)
    #print("Tamaño población:", args.tamano_poblacion)
    #print("Elementos población inicial:", args.n_elementos_p_inicial)
    #print("Máximo épocas:", args.n_max_epocs)
    #print("Valor p-value suficiente:", args.valor_suficiente_pvalue)
    #print("Probabilidad cruce:", args.probabilidad_cruce)
    #print("Fichero de salida:", args.output)

    if args.estricto:
        args.valor_suficiente_pvalue = 1e-8

    if args.tipo_variable == "categoric":
        result = run_genetic_categorico(args.dataset,args.tamano_poblacion,args.n_elementos_p_inicial,args.n_max_epocs,args.valor_suficiente_pvalue,args.probabilidad_cruce,args.output)
    elif args.tipo_variable == "numeric":
        result = run_genetic_numeric(args.dataset,args.tamano_poblacion,args.n_elementos_p_inicial,args.n_max_epocs,args.valor_suficiente_pvalue,args.probabilidad_cruce,args.output)
    else:
        print("El tipo de variable indicado no es válido, opciones \"categoric\" y \"numeric\"")
    return result
if __name__ == "__main__":
    main()