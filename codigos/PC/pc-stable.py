from itertools import combinations
import pandas as pd
import numpy as np
import random
from scipy.stats import pearsonr, spearmanr
from pgmpy.estimators import CITests, PC
from scipy.stats import chi2_contingency
import psutil
import os
import time


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
def experiment_pc_estable(data, type_data="categoricos"):
    if type_data == "categoricos":
        pc = PC(data)
        model_pcstable_cate = pc.estimate(ci_test=CITests.chi_square, variant='stable',return_type='cpdag')
        print("Estructura aprendida por el algoritmo PC-stable:")
        print(model_pcstable_cate.edges)
        print(f"Ha detectado {len(model_pcstable_cate.edges)}")
    else:
        pc = PC(data)
        model_pcstable_nume = pc.estimate(ci_test=CITests.pearsonr, variant='stable',return_type='cpdag')
        print("Estructura aprendida por el algoritmo PC-stable:")
        print(model_pcstable_nume.edges)
        print(f"Ha detectado {len(model_pcstable_nume.edges)}")

def main():
    datasets = { "categoricos": ["adult_3000", "adult_6000","adult_9000","adult_12000","adult_15000","adult_18000","adult_21000","adult_24000","adult_27000","adult_30000","agar_800","agar_1600","agar_2400","agar_3200","agar_4000","agar_4800","agar_5600","agar_6400","agar_7200","agar_8000","nursery_1200","nursery_2400","nursery_3600","nursery_4800","nursery_6000","nursery_7200","nursery_8400","nursery_9600","nursery_10800","nursery_12000"],
                "continuos": ["wec49_3600","wec49_3600x50","wec49_3600x100","wec49_7200","wec49_7200x50","wec49_7200x100","wec49_10800","wec49_10800x50","wec49_10800x100","wec49_14400","wec49_14400x50","wec49_14400x100","wec49_18000","wec49_18000x50","wec49_18000x100","wec49_21600","wec49_21600x50","wec49_21600x100","wec49_25200","wec49_25200x50","wec49_25200x100","wec49_28800","wec49_28800x50","wec49_28800x100","wec49_32400","wec49_32400x50","wec49_32400x100","wec49_36000","wec49_36000x50","wec49_36000x100" ] 
                }
    work_path = "completo/datasets"
    for type_dataset, list_datasets in datasets.items():
        for i in list_datasets:
            print(f"Dataset: {i}")
            data = pd.read_csv(f"{work_path}/{i}.csv")
            experiment_pc(data, type_dataset)
            experiment_pc_estable(data, type_dataset)
            #experiment_notears(data, type_dataset)
            experiment_own(data, type_dataset)

if __name__ == "__main__":
    main()
