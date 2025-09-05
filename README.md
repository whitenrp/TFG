
# Algoritmos de Inferencia Causal (Trabajo Fin de Grado)

En este repositorio se encontrará todo el código desarrollado para la realización del Trabajo de Fin de Grado, para el Grado de Ingeniería Informática de la Universidad de Córdoba.

## Estructura del Repositorio
Este repositorio se encuentra dividido en diferentes carpetas con el fin de permitir una mejor comprensión del proyecto desarrollado, a continuación se encontrará el árbol de directorios para cada una de las carpetas principales.
### Codigos
En este directorio se encuentran todos los archivos necesarios para hacer uso de los algoritmos y la aplicación desarrollada. Se puede observar como en el se encuentra el fichero ```requirements.txt``` el cual permitirá llevar a cabo la instalación de las diferentes librerías que se requieren para un uso correcto. 


    ├── codigos
    │   ├── evolutivo
    │   │   ├── main.py
    │   │   ├── main_estricto.py
    │   │   ├── test.py
    │   │   └── test_estricto.py
    │   ├── PC
    │   │   ├── Pc.py
    │   │   ├── PC-stable.py
    │   │   ├── test_PC.py
    │   │   └── test_PC-stable.py
    │   └── Notears
    |       ├── Notears.py
    │       └── test_Notears.py

### Datasets
En este directorio se encuentran todas las bases de datos utilizadas en la Experimentación realizada en el Manual Técnico del Trabajo de Fin de Grado. Con el fin de que se puedan analizar y ser utilizadas fácilmente por aquellas personas que deseen comprobar el funcionamiento de los algoritmos desarrollados.

    ├── datasets
    │   ├── Categóricos
    │   │   ├── Alta Dimensionalidad
    │   │   │   ├── Adult
    │   │   │   ├── Agar
    │   │   │   ├── Connect-4
    │   │   │   └── nursery
    │   │   └── Baja Dimensionalidad
    │   │       ├── sintetico_categórico
    │   │       ├── breast-cancer
    │   │       ├── car
    │   │       └── house-votes-84
    │   └── Continuos
    │       ├── Alta Dimensionalidad
    │       │   ├── criteo-uplift-v2.1
    │       │   ├── dowhy
    │       │   ├── higgs
    │       │   └── nursery
    │       └── Baja Dimensionalidad
    │           ├── sintetico_continuo
    │           ├── ihpi
    │           ├── car
    │           └── house-votes-84


### Ficheros
En el se encontrarán todas los ficheros auxiliares, imagenes y resultados de las pruebas que han sido utilizados a la hora de desarrollar el TFG. 

    ├── ficheros
    │   ├── imagenes
    │   │   ├── categoricos_recursos_columnas
    │   │   ├── categoricos_recursos_filas
    │   │   ├── categoricos_tiempo_columnas
    │   │   ├── categoricos_tiempo_filas
    │   │   ├── connect4_recursos
    │   │   ├── connect4_tiempos
    │   │   ├── continuos_recursos_columnas
    │   │   ├── continuos_recursos_filas
    │   │   ├── continuos_tiempo_columnas
    │   │   ├── continuos_tiempo_filas
    │   │   ├── criteo_recursos
    │   │   ├── criteo_tiempos
    │   │   ├── dowhy_recursos
    │   │   ├── dowhy_tiempos
    │   │   ├── higgs_recursos
    │   │   ├── higgs_tiempo
    │   │   ├── nemnyi_categorico_4
    │   │   ├── nemnyi_categorico_4_recursos
    │   │   ├── nemnyi_categorico_4_tiempo
    │   │   ├── nemnyi_continuos_4_recursos
    │   │   ├── nemnyi_continuos_4_tiempo
    │   │   ├── nursery_consumo
    │   │   ├── nursery_tiempo
    │   │   ├── parkinson_recursos
    │   │   ├── parkinson_tiempo
    │   ├── prueba_tiempo_algoritmo
    │   │   └── Pruebas_de_rendimiento.csv
    │   └── resultados_experimentacion
    │       ├── Categóricos
    │       │   ├── Adult
    │       │   │   ├── Salida_Notears.txt
    │       │   │   ├── Salida_Evolutivo.txt
    │       │   │   ├── Salida_Evolutivo-estricto.txt
    │       │   │   ├── Salida_PC.txt
    │       │   │   └── Salida_PC-stable.txt
    │       │   ├── Agar
    │       │   │   ├── Salida_Notears.txt
    │       │   │   ├── Salida_Evolutivo.txt
    │       │   │   ├── Salida_Evolutivo-estricto.txt
    │       │   │   ├── Salida_PC.txt
    │       │   │   └── Salida_PC-stable.txt
    │       │   ├── Connect-4
    │       │   │   ├── Salida_Notears.txt
    │       │   │   ├── Salida_Evolutivo.txt
    │       │   │   ├── Salida_Evolutivo-estricto.txt
    │       │   │   ├── Salida_PC.txt
    │       │   │   └── Salida_PC-stable.txt
    │       │   ├── nursery
    │       │   │   ├── Salida_Notears.txt
    │       │   │   ├── Salida_Evolutivo.txt
    │       │   │   ├── Salida_Evolutivo-estricto.txt
    │       │   │   ├── Salida_PC.txt
    │       │   │   └── Salida_PC-stable.txt
    │       │   ├── sintetico_categórico
    │       │   │   ├── Salida_Notears.txt
    │       │   │   ├── Salida_Evolutivo.txt
    │       │   │   ├── Salida_Evolutivo-estricto.txt
    │       │   │   ├── Salida_PC.txt
    │       │   │   └── Salida_PC-stable.txt
    │       │   ├── breast-cancer
    │       │   │   ├── Salida_Notears.txt
    │       │   │   ├── Salida_Evolutivo.txt
    │       │   │   ├── Salida_Evolutivo-estricto.txt
    │       │   │   ├── Salida_PC.txt
    │       │   │   └── Salida_PC-stable.txt
    │       │   ├── car
    │       │   │   ├── Salida_Notears.txt
    │       │   │   ├── Salida_Evolutivo.txt
    │       │   │   ├── Salida_Evolutivo-estricto.txt
    │       │   │   ├── Salida_PC.txt
    │       │   │   └── Salida_PC-stable.txt
    │       │   └── house-votes-84
    │       │       ├── Salida_Notears.txt
    │       │       ├── Salida_Evolutivo.txt
    │       │       ├── Salida_Evolutivo-estricto.txt
    │       │       ├── Salida_PC.txt
    │       │       └── Salida_PC-stable.txt
    │       └── Categóricos
    │           ├── Adult
    │           │   ├── Salida_Notears.txt
    │           │   ├── Salida_Evolutivo.txt
    │           │   ├── Salida_Evolutivo-estricto.txt
    │           │   ├── Salida_PC.txt
    │           │   └── Salida_PC-stable.txt
    │           ├── Agar
    │           │   ├── Salida_Notears.txt
    │           │   ├── Salida_Evolutivo.txt
    │           │   ├── Salida_Evolutivo-estricto.txt
    │           │   ├── Salida_PC.txt
    │           │   └── Salida_PC-stable.txt
    │           ├── Connect-4
    │           │   ├── Salida_Notears.txt
    │           │   ├── Salida_Evolutivo.txt
    │           │   ├── Salida_Evolutivo-estricto.txt
    │           │   ├── Salida_PC.txt
    │           │   └── Salida_PC-stable.txt
    │           ├── nursery
    │           │   ├── Salida_Notears.txt
    │           │   ├── Salida_Evolutivo.txt
    │           │   ├── Salida_Evolutivo-estricto.txt
    │           │   ├── Salida_PC.txt
    │           │   └── Salida_PC-stable.txt
    │           ├── sintetico_categórico
    │           │   ├── Salida_Notears.txt
    │           │   ├── Salida_Evolutivo.txt
    │           │   ├── Salida_Evolutivo-estricto.txt
    │           │   ├── Salida_PC.txt
    │           │   └── Salida_PC-stable.txt
    │           ├── breast-cancer
    │           │   ├── Salida_Notears.txt
    │           │   ├── Salida_Evolutivo.txt
    │           │   ├── Salida_Evolutivo-estricto.txt
    │           │   ├── Salida_PC.txt
    │           │   └── Salida_PC-stable.txt
    │           ├── car
    │           │   ├── Salida_Notears.txt
    │           │   ├── Salida_Evolutivo.txt
    │           │   ├── Salida_Evolutivo-estricto.txt
    │           │   ├── Salida_PC.txt
    │           │   └── Salida_PC-stable.txt
    │           └── house-votes-84
    │               ├── Salida_Notears.txt
    │               ├── Salida_Evolutivo.txt
    │               ├── Salida_Evolutivo-estricto.txt
    │               ├── Salida_PC.txt
    │               └── Salida_PC-stable.txt

    

Finalmente en este proyecto se encuentra el ```README.md``` que pretende explicar con detalle todo el proyecto.

    └── README.md


## Autor

- [Nemesio Romero Pino](https://www.github.com/whitenrp) Alumno de 4º curso del Grado de Ingeniería Informática – Mención: Computación.

## Director
- [Jose Maria Luna Ariza](https://github.com/jmluna) Profesor Contratado Doctor del Dpto. de Informática y Análisis Numérico.

## Codirector
- [Christian Luna Escudero](https://www.github.com/ChrisLe7) Personal Investigador Pre-doctoral FPU.


## Instalación

A la hora de desear hacer uso de cualquier parte de este proyecto, se recomienda para un uso correcto utilizar Python 3.10.

### Instalación de Python
Para la instalación de Python, se pueden seguir los pasos para cada sistema operativo expuestos a continuación o hacer uso del videotutorial desarrollado, el cuál se puede encontrar en Youtube, a través del siguiente [enlace](https://www.youtube.com/watch?v=TDQdaDHtyGA).

#### Windows
1. Diríjase a la página oficial de Python y descargue la versión 3.10 del instalador de Python
para Windows.

2. Ejecute el archivo de instalación descargado y siga las instrucciones del asistente de instalación.

3. En la ventana de configuración, asegúrese de seleccionar la opción “Agregar Python 3.10 a PATH”. Esto permitirá utilizar Python desde la línea de comandos de Windows.

4. Continúe con la instalación hasta que se complete.

#### Linux 

Python suele estar incluido en casi todas las distribuciones de GNU/Linux. Si se diera el caso de que no estuviera instalado en nuestro equipo, o que a versión instalada no fuera la 3.9 se deberán de realizar los siguientes pasos para su instalación.
1. Abra un terminal y actualice el índice de paquetes apt con el comando:
```bash
sudo apt-get update
```
2. Instale Python con el comando:
```bash
sudo apt-get install python3.10
```
3. Instale el gestor de entornos de Python:
```bash
sudo apt-get install python3.10-venv
```
4. Para la creación y activación del entorno se podrá hacer uso de los siguientes comandos:
```bash
python3.10 -m venv ~/TFG
source ~/TFG/bin/activate
```
###  Instalación de las librerías requeridas
Para la instalación de las diferentes librerías se puede realizar todo fácilmente mediante el fichero `requirements.txt` que se encuentra dentro del directorio `codigos`. Se recomienda tener actualizado `pip` a la última versión disponible.

```bash
pip install -r requirements.txt
```

## Desinstalación

### Desinstalación de Python
##### Windows
Para desinstalar Python, se deberá de hacer

- Abra el menú de Inicio y busque “Agregar o quitar programas” (o “Programas y características” en versiones más recientes de Windows).
- Busca “Python” en la lista de programas instalados y selecciona la versión que se desea desinstalar.
- Haga clic en “Desinstalar” y siga las instrucciones del asistente de desinstalación.
- Una vez que se haya completado el proceso de desinstalación, asegúrese de eliminar cualquier archivo o carpeta relacionada con Python que aún pueda existir en su sistema.

#### Linux
Python no se puede desinstalar en Linux, ya que algunas partes del sistema lo necesitan para funcionar. Pero si se puede desinstalar la versión concreta realizada, para ello se pueden seguir los siguientes pasos:
1. Elimine el entorno creado:
```bash
rm -r ~/TFG
```
2. Desinstalación del gestor de entornos de python3.9 y python3.9.
```bahs
python remove python3.9-venv
python remove python3.9
```
