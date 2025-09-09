#!pip install git+https://github.com/xunzheng/notears.git
#!pip install psutil

#import psutil
#from notears.linear import notears_linear
#import networkx as nx
#import pandas as pd
#import time
#import numpy as np
#import tracemalloc


def measure_performance(func):
    def wrapper(*args, **kwargs):
        process = psutil.Process(os.getpid())
        start_time = time.time()
        start_memory = process.memory_info().rss
        start_cpu = process.cpu_percent(interval=None)
        
        result = func(*args, **kwargs)
        
        end_time = time.time()
        end_memory = process.memory_info().rss
        end_cpu = process.cpu_percent(interval=None)
        
        print(f"Time taken: {end_time - start_time:.2f} seconds")
        print(f"Memory used: {end_memory - start_memory} bytes")
        print(f"CPU used: {end_cpu - start_cpu}%")
        
        return result
    return wrapper

@measure_performance
def experiment_notears(data, type_data="categoricos"):
    if type_data == "categoricos":
        data_one_hot = pd.get_dummies(data)
        data_matrix = data_one_hot.to_numpy()
        W_est = notears_linear(data_matrix, lambda1=0.1, loss_type='l2')

        nombres_columnas = data.columns.to_list()
        relaciones_cero = []
        # Recorremos toda la matriz
        for i in range(len(W_est)):
            for j in range(len(W_est)):
                if W_est[i, j] != 0:  # Verificar si el valor es distinto de 0
                    # Prevenir el acceso fuera de rango
                    if i < len(nombres_columnas) and j < len(nombres_columnas):
                        relacion = f"[{nombres_columnas[i]}-{nombres_columnas[j]}]"
                        relaciones_cero.append(relacion)

        print("Estructura aprendida por el algoritmo Notears:")
        print(relaciones_cero)
        print(f"Ha detectado {len(relaciones_cero)} relaciones distintas de cero.")
    else:
        W_est = notears_linear(data.values, lambda1=0.1, loss_type='l2')
        G = nx.DiGraph(W_est)
        nombres_columnas = data.columns.to_list()
        relaciones_cero = []
        for i in range(len(W_est)):
            for j in range(len(W_est)):
                if W_est[i, j] != 0:
                    if i < len(nombres_columnas) and j < len(nombres_columnas):
                        relacion = f"({nombres_columnas[i]}-{nombres_columnas[j]})"
                        relaciones_cero.append(relacion)

        print("Estructura aprendida por el algoritmo Notears:")
        print(relaciones_cero)
        print(f"Ha detectado {len(relaciones_cero)} relaciones distintas de cero.")


def main():
    datasets = { "categoricos": ["adult_3000", "adult_6000","adult_9000","adult_12000","adult_15000","adult_18000","adult_21000","adult_24000","adult_27000","adult_30000","agar_800","agar_1600","agar_2400","agar_3200","agar_4000","agar_4800","agar_5600","agar_6400","agar_7200","agar_8000","nursery_1200","nursery_2400","nursery_3600","nursery_4800","nursery_6000","nursery_7200","nursery_8400","nursery_9600","nursery_10800","nursery_12000"],
                "continuos": ["wec49_3600","wec49_3600x50","wec49_3600x100","wec49_7200","wec49_7200x50","wec49_7200x100","wec49_10800","wec49_10800x50","wec49_10800x100","wec49_14400","wec49_14400x50","wec49_14400x100","wec49_18000","wec49_18000x50","wec49_18000x100","wec49_21600","wec49_21600x50","wec49_21600x100","wec49_25200","wec49_25200x50","wec49_25200x100","wec49_28800","wec49_28800x50","wec49_28800x100","wec49_32400","wec49_32400x50","wec49_32400x100","wec49_36000","wec49_36000x50","wec49_36000x100" ] 
                }
    work_path = "completo/datasets"
    for type_dataset, list_datasets in datasets.items():
        for i in list_datasets:
            print(f"Dataset: {i}")
            experiment_notears(i, type_dataset)

if __name__ == "__main__":
    main()