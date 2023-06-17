------------------------------------GUÍA PARA LA EJECUCIÓN DE LOS ALGORITMOS DEL NOTEBOOK PARA EL PROYECTO DE PREDICCIÓN DE PROBABILIDAD DE DIABETES EN PACIENTES---------------------------




--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

								INTEGRANTES: ANA ESTEFANÍA HENAO RESTREPO Y JUAN JOSÉ GIL HOYOS
									ESPECIALIZACIÓN EN ANALÍTICA Y CIENCIA DE DATOS 
										 FACULTAD DE INGENIERÍA 
									DEPARTAMENTO DE INGENIERÍA DE SISTEMAS
										UNIVERSIDAD DE ANTIOQUIA

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

PASOS A TENER EN CUENTA:

1. Por favor descargar o utilizar directamente desde el repositorio de GitHub de cualquiera de los integrantes de este equipo de trabajo, el archivo .csv que tiene por nombre "train_data.csv"
2. Este notebook está compuesto por varias secciones, las cuales se irán detallando a continuación:
	
---------------------------------------------------------------------Librerías necesarias---------------------------------------------------------------------------------------------------


Esta sección del notebook contiene aquellos paquetes que fue necesario importarlos para hacer uso de los modelos de machine learning que permitieran dar cumplimiento al objetivo inicial de este proyecto. Por mencionar alguno de ellos, es necesario hacer uso de sklearn, matplotlib, numpy, pandas, seaborn, scipy, entre otras.


---------------------------------------------------------------------Importación de datos---------------------------------------------------------------------------------------------------


Aquí se utiliza el comando pd.read_csv para hacer una lectura de los datos que se encuentran contenidos dentro del archivo de extensión .csv que tiene por nombre "train_data.csv"

NOTA: ES IMPORTANTE QUE SE MODIFIQUE LA RUTA DEL ARCHIVO "C:/Users/juanj/Universidad de Antioquia/Monografia[] - General/Predicción de Diabetes Nov 2022/train_data.csv", YA QUE LA QUE 
EXISTE ACTUALMENTE NO NECESARIAMENTE SERÁ IGUAL A UNA DE LAS RUTAS DENTRO DE UN ORDENADOR DISTINTO.

Dentro de esta sección, se hace un análisis exploratorio inicial en el cual si se ejecutan las celdas, se podrá conocer los primeros 5 encabezados del dataset original, el tipo de variables
con las que cuenta el dataset y se podrá determinar si hay presencia o no de datos nulos.


--------------------------------------------------------------------Distribución de la variable de respuesta--------------------------------------------------------------------------------


El objetivo principal de esta sección es poder determinar si existe un desbalanceo de las clases que componen a la variable de respuesta. Cuando se ejecuta la primera celda de esta sección
se puede visualizar un gráfico de barras en el cual las clases tienen aproximadamente 40.000 cada una de ellas. Por esta razón, se consideró que la variable de respuesta no requería de ningún tratamiento para hacer un submuestreo o un sobremuestreo.

Si se ejecuta la siguiente celda, se genera una matriz de histogramas que permite visualizar cuál es la distribución que tiene cada una de las variables dentro del dataset. Con base en los resultados gráficos obtenidos, se tomó la decisión de distribuir las variables en los siguiente conjuntos: 

		* VARIABLES NUMÉRICAS: Age, BMI, GenHlth, MentHlth y PhysHlth (5 en total)
		* VARIABLES CATEGÓRICAS: Sex, HighChol, CholCheck, Smoker, HeartDiseaseorAttack, PhysActivity, Fruits, Veggies, HvyAlcoholConsump,DiffWalk,Hypertension (12 en total)


--------------------------------------------------------------------Detección de datos atípicos---------------------------------------------------------------------------------------------

Esta sección está destinada a determinar si dentro del dataset a utilizar dentro del notebook posee datos atípicos o no, y en caso de que existan, aplicar un algoritmo de eliminación controlada de datos atípicos.

En la tercera celda de esta sección, se utiliza la librería seaborn para graficar un diagrama de caja y bigotes de las variables numéricas que nos permita evaluar qué registros están por encima y por debajo de los rangos intercuartílicos. En esta representación gráfica, se observa que las variables BMI, MentHlth y PhysHtlth poseen valores atípicos. Por lo anterior, la cuarta celda de esta sección contiene las líneas de código que permiten efectuar la eliminación de los datos atípicos dentro del dataset. El parámetro más importante de la función LocalOutlierFactor es "contamination". Esta se estableció en un 10% para no prescindir de una cantidad considerable del dataset que posteriormente conduzca a sesgos en los modelos a implementar. 

Posteriormente, luego de aplicar el algoritmo de eliminación controlada de datos atípicos y graficar un nuevo diagrama de caja y bigotes, se hizo uso dentro de la novena celda de esta sección de la función entr() para calcular la entropía de Shannon de los datos antes de y después de la eliminación de datos atípicos. Si se comparan los valores de entropía, se puede evidenciar que no hay gran diferencia entre dichas cantidades antes y después de la eliminación. Por lo tanto, se puede concluir que la distribución original de los datos se mantiene. Para corroborar este hecho, se grafica una matriz de gráficos de dispersión de las variables numéricas con ayuda de la instrucción pd.plotting.scatter_matrix()

Finalmente, se evalúa a través de una matriz de correlación entre pares de variables qué tan fuerte es la dependencia que existe entre todas las variables del dataset. Con ayuda de la librería seaborn, se graficó un mapa de calor que permite evidenciar que no existen correlaciones fuertes entre ningún par de variables dentro del dataset, esto debido a que todos los valores dentro del mapa de calor son muy inferiores a la unidad, con excepción de la diagonal principal.

---------------------------------------------------------------SECCIÓN DE APLICACIÓN DE MODELOS--------------------------------------------------------------------------------------------

En esta sección se dará a conocer la secuencia de ejecución que debe seguirse para poder obtener los resultados que se alcanzaron dentro del notebook. Inicialmente, es necesario mencionar que para cada uno de los modelos de machine learning implementados, se evaluaron dos escenarios: Uno en el que se utilizó el dataset original previo a la eliminación de datos atípicos, y otro dataset que consiste de aquellos datos en los que sí se eliminaron datos atípicos.

---------------------------------------------------------APLICACIÓN DEL MODELO DE REGRESIÓN LOGÍSTICA-------------------------------------------------------------------------------------

ESCENARIO SIN ELIMINACIÓN DE DATOS ATÍPICOS:

La primera celda de esta sección permite crear las Variables X e Y que almacenan a las variables de entrada y a la variable de respuesta, respectivamente. Acto seguido, la segunda celda de esta sección permite hacer la partición del dataset que se utilizará para el entrenamiento y para posteriores pruebas de los modelos. El método train_test_split() se empleó para tomar de manera aleatoria el 80% de los datos del dataset original para utilizarlos en la etapa de entrenamiento del modelo.

En la tercera celda de esta sección, se tienen las líneas de código que permiten ejecutar el algoritmo de búsqueda de rejilla (Grid Search). Los hiper parámetros que se variaron para así obtener sus valores óptimos fueron: penalty, solver y C. Es importante utilizar el hiper parámetro n_jobs = -1 para hacer uso de todos los núcleos con los que cuente la máquina en la cual se esté ejecutando el algoritmo. Esta celda toma aproximadamente 31.8 segundos en ejecutarse.

Luego de haber ejecutado la búsqueda de rejilla, se obtiene que los mejores hiper parámetros son: {'C':0.1, 'penalty':'l1', 'solver':'saga'} 

ESCENARIO CON ELIMINACIÓN DE DATOS ATÍPICOS:

La primera celda de esta sección permite crear las Variables X_LOF e Y_LOF que almacenan a las variables de entrada y a la variable de respuesta, respectivamente. Acto seguido, la segunda celda de esta sección permite hacer la partición del dataset que se utilizará para el entrenamiento y para posteriores pruebas de los modelos. El método train_test_split() se empleó para tomar de manera aleatoria el 80% de los datos del dataset original para utilizarlos en la etapa de entrenamiento del modelo.

En la tercera celda de esta sección, se tienen las líneas de código que permiten ejecutar el algoritmo de búsqueda de rejilla (Grid Search). Los hiper parámetros que se variaron para así obtener sus valores óptimos fueron: penalty, solver y C. Es importante utilizar el hiper parámetro n_jobs = -1 para hacer uso de todos los núcleos con los que cuente la máquina en la cual se esté ejecutando el algoritmo. Esta celda toma aproximadamente 30.7 segundos en ejecutarse.

Luego de haber ejecutado la búsqueda de rejilla, se obtiene que los mejores hiper parámetros son: {'C':0.1, 'penalty':'l1', 'solver':'liblinear'}


---------------------------------------------------------APLICACIÓN DEL MODELO KNN---------------------------------------------------------------------------------------------------------

ESCENARIO SIN ELIMINACIÓN DE DATOS ATÍPICOS:

La primera celda se utiliza para calcular una heurística para determinar una aproximación del número de vecinos mas cercanos a emplear, por medio del cálculo de la raiz cuadrada de las observaciones dentro del conjunto de datos.

La segunda celda se utiliza para ejecutar un algoritmo de búsqueda de rejilla (Grid Search). Los hiper parámetros que se variaron para obtener los valores óptimos fueron: n_neighbors y metric. Esta celda toma aproximadamente 24.37 minutos en ejecutarse.

Luego de haber ejecutado la búsqueda de rejilla, se obtiene que los mejores hiper parámetros son: {'n_neighbors':93, 'metric':manhattan}

ESCENARIO CON ELIMINACIÓN DE DATOS ATÍPICOS:

La primera celda se utiliza para ejecutar un algoritmo de búsqueda de rejilla (Grid Search). Los hiper parámetros que se variaron para obtener los valores óptimos fueron: n_neighbors y metric. Esta celda toma aproximadamente 26.24 minutos en ejecutarse.

Luego de haber ejecutado la búsqueda de rejilla, se obtiene que los mejores hiper parámetros son: {'n_neighbors':21, 'metric':manhattan}


----------------------------------------APLICACIÓN DEL MODELO NAIVE BAYES CON LA MODIFICACIÓN DE BERNOULLI----------------------------------------------------------------------------------

ESCENARIO SIN ELIMINACIÓN DE DATOS ATÍPICOS:

En la primera celda de esta sección, se tienen las líneas de código que permiten ejecutar el algoritmo de búsqueda de rejilla (Grid Search). Los hiper parámetros que se variaron para así obtener sus valores óptimos fueron: binarize, alpha y force_alpha. Es importante utilizar el hiper parámetro n_jobs = -1 para hacer uso de todos los núcleos con los que cuente la máquina en la cual se esté ejecutando el algoritmo. Esta celda toma aproximadamente 7.27 segundos en ejecutarse.

Luego de haber ejecutado la búsqueda de rejilla, se obtiene que los mejores hiper parámetros son: {'alpha':0, 'binarize': 0.35, 'force_alpha':'True'} 

ESCENARIO CON ELIMINACIÓN DE DATOS ATÍPICOS:

En la primera celda de esta sección, se tienen las líneas de código que permiten ejecutar el algoritmo de búsqueda de rejilla (Grid Search). Los hiper parámetros que se variaron para así obtener sus valores óptimos fueron: binarize, alpha y force_alpha. Es importante utilizar el hiper parámetro n_jobs = -1 para hacer uso de todos los núcleos con los que cuente la máquina en la cual se esté ejecutando el algoritmo. Esta celda toma aproximadamente 2.6 segundos en ejecutarse.

Luego de haber ejecutado la búsqueda de rejilla, se obtiene que los mejores hiper parámetros son: {'alpha':0.5, 'binarize': 0.35, 'force_alpha':'False'}


------------------------------------------------------------------APLICACIÓN DEL MODELO RANDOM FOREST--------------------------------------------------------------------------------------

ESCENARIO SIN ELIMINACIÓN DE DATOS ATÍPICOS:

La primera celda se utiliza para ejecutar un algoritmo de búsqueda de rejilla (Grid Search). Los hiper parámetros que se variaron para obtener los valores óptimos fueron: n_estimators, max_depth y criterion. Es importante utilizar el hiper parámetro n_jobs = -1 para hacer uso de todos los núcleos con los que cuente la máquina en la cual se esté ejecutando el algoritmo. Esta celda toma aproximadamente 3.1 minutos en ejecutarse.

Luego de haber ejecutado la búsqueda de rejilla, se obtiene que los mejores hiper parámetros son: {'n_estimators':500, 'max_depth':5, 'criterion':gini}

ESCENARIO CON ELIMINACIÓN DE DATOS ATÍPICOS:

La primera celda se utiliza para ejecutar un algoritmo de búsqueda de rejilla (Grid Search). Los hiper parámetros que se variaron para obtener los valores óptimos fueron: n_estimators, max_depth y criterion. Es importante utilizar el hiper parámetro n_jobs = -1 para hacer uso de todos los núcleos con los que cuente la máquina en la cual se esté ejecutando el algoritmo. Esta celda toma aproximadamente 2.43 minutos en ejecutarse.

Luego de haber ejecutado la búsqueda de rejilla, se obtiene que los mejores hiper parámetros son: {'n_estimators':1000, 'max_depth':5, 'criterion':gini}

###########################################################################################################################################################################################

NOTA: LOS MODELOS QUE A CONTINUACIÓN SE MENCIONAN, NO EMPLEARON EL 80% DEL DATASET PARA ENTRENARSE. EN SU LUGAR SE TOMARON MUESTRAS DE 2.000, 8.000 Y 40.000. ESTA DECISIÓN SE TOMÓ TENIENDO EN CUENTA EL ELEVADO COSTO COMPUTACIONAL QUE ESTOS TIENEN

###########################################################################################################################################################################################

--------------------------------------------------------APLICACIÓN DEL MODELO SUPPORT VECTOR CLASIFFIER------------------------------------------------------------------------------------

TAMAÑO DE MUESTRA DE 2.000 DATOS

SIN ELIMINACIÓN DE DATOS ATÍPICOS

En la primera celda de esta sección, se utiliza el método train_test_split() para tomar exactamente dos mil datos del dataset de entrenamiento. Es importante que se ejecute primero que las celdas posteriores para así poder sobre escribir las variables X_train, X_test, Y_train e Y_test.

Posteriormente, en la segunda celda de esta sección, se realiza una búsqueda de rejilla para cuantificar cuáles son los valores de los mejores hiperparámetros del modelo. Los hiper parámetros que se utilizaron dentro de la búsqueda fueron: kernel, C y gamma. Es importante hacer uso del hiper parámetro n_jobs = -1 para que se empleen todos los núcleos del procesador con el que cuente la máquina en la cual se esté ejecutando el algoritmo.

Luego de haber ejecutado el algoritmo GridSearchCV(), se obtiene que los mejores hiper parámetros son: {'C':10, 'gamma':'auto', 'kernel':'linear'} El tiempo de ejecución del algoritmo de búsqueda de rejilla es de aproximadamente 2.23 segundos.


TAMAÑO DE MUESTRA DE 8.000 DATOS

SIN ELIMINACIÓN DE DATOS ATÍPICOS

En la primera celda de esta sección, se utiliza el método train_test_split() para tomar exactamente ocho mil datos del dataset de entrenamiento. Es importante que se ejecute primero que las celdas posteriores para así poder sobre escribir las variables X_train, X_test, Y_train e Y_test

Posteriormente, en la segunda celda de esta sección, se realiza una búsqueda de rejilla para cuantificar cuáles son los valores de los mejores hiperparámetros del modelo. Los hiper parámetros que se utilizaron dentro de la búsqueda fueron: kernel, C y gamma. Es importante hacer uso del hiper parámetro n_jobs = -1 para que se empleen todos los núcleos del procesador con el que cuente la máquina en la cual se esté ejecutando el algoritmo.

Luego de haber ejecutado el algoritmo GridSearchCV(), se obtiene que los mejores hiper parámetros son: {'C':1, 'gamma':'auto', 'kernel':'rbf'} El tiempo de ejecución del algoritmo de búsqueda de rejilla es de aproximadamente 51.5 segundos.


TAMAÑO DE MUESTRA DE 40.000 DATOS

SIN ELIMINACIÓN DE DATOS ATÍPICOS

En la primera celda de esta sección, se utiliza el método train_test_split() para tomar exactamente cuarenta mil datos del dataset de entrenamiento. Es importante que se ejecute primero que las celdas posteriores para así poder sobre escribir las variables X_train, X_test, Y_train e Y_test

Posteriormente, en la segunda celda de esta sección, se realiza una búsqueda de rejilla para cuantificar cuáles son los valores de los mejores hiperparámetros del modelo. Los hiper parámetros que se utilizaron dentro de la búsqueda fueron: kernel, C y gamma. Es importante hacer uso del hiper parámetro n_jobs = -1 para que se empleen todos los núcleos del procesador con el que cuente la máquina en la cual se esté ejecutando el algoritmo.

Luego de haber ejecutado el algoritmo GridSearchCV(), se obtiene que los mejores hiper parámetros son: {'C':1, 'gamma':'auto', 'kernel':'rbf'} El tiempo de ejecución del algoritmo de búsqueda de rejilla es de aproximadamente 37 minutos con 24 segundos.


TAMAÑO DE MUESTRA DE 2.000 DATOS

CON ELIMINACIÓN DE DATOS ATÍPICOS

En la primera celda de esta sección, se utiliza el método train_test_split() para tomar exactamente dos mil datos del dataset de entrenamiento. Es importante que se ejecute primero que las celdas posteriores para así poder sobre escribir las variables X_train_LOF, X_test_LOF, Y_train_LOF e Y_test_LOF.

Posteriormente, en la segunda celda de esta sección, se realiza una búsqueda de rejilla para cuantificar cuáles son los valores de los mejores hiperparámetros del modelo. Los hiper parámetros que se utilizaron dentro de la búsqueda fueron: kernel, C y gamma. Es importante hacer uso del hiper parámetro n_jobs = -1 para que se empleen todos los núcleos del procesador con el que cuente la máquina en la cual se esté ejecutando el algoritmo.

Luego de haber ejecutado el algoritmo GridSearchCV(), se obtiene que los mejores hiper parámetros son: {'C':10, 'gamma':'auto', 'kernel':'rbf'} El tiempo de ejecución del algoritmo de búsqueda de rejilla es de aproximadamente 2.18 segundos.


TAMAÑO DE MUESTRA DE 8.000 DATOS

CON ELIMINACIÓN DE DATOS ATÍPICOS

En la primera celda de esta sección, se utiliza el método train_test_split() para tomar exactamente ocho mil datos del dataset de entrenamiento. Es importante que se ejecute primero que las celdas posteriores para así poder sobre escribir las variables X_train_LOF, X_test_LOF, Y_train_LOF e Y_test_LOF

Posteriormente, en la segunda celda de esta sección, se realiza una búsqueda de rejilla para cuantificar cuáles son los valores de los mejores hiperparámetros del modelo. Los hiper parámetros que se utilizaron dentro de la búsqueda fueron: kernel, C y gamma. Es importante hacer uso del hiper parámetro n_jobs = -1 para que se empleen todos los núcleos del procesador con el que cuente la máquina en la cual se esté ejecutando el algoritmo.

Luego de haber ejecutado el algoritmo GridSearchCV(), se obtiene que los mejores hiper parámetros son: {'C':10, 'gamma':'auto', 'kernel':'rbf'} El tiempo de ejecución del algoritmo de búsqueda de rejilla es de aproximadamente 53.7 segundos.


TAMAÑO DE MUESTRA DE 40.000 DATOS

CON ELIMINACIÓN DE DATOS ATÍPICOS

En la primera celda de esta sección, se utiliza el método train_test_split() para tomar exactamente cuarenta mil datos del dataset de entrenamiento. Es importante que se ejecute primero que las celdas posteriores para así poder sobre escribir las variables X_train_LOF, X_test_LOF, Y_train_LOF e Y_test_LOF

Posteriormente, en la segunda celda de esta sección, se realiza una búsqueda de rejilla para cuantificar cuáles son los valores de los mejores hiperparámetros del modelo. Los hiper parámetros que se utilizaron dentro de la búsqueda fueron: kernel, C y gamma. Es importante hacer uso del hiper parámetro n_jobs = -1 para que se empleen todos los núcleos del procesador con el que cuente la máquina en la cual se esté ejecutando el algoritmo.

Luego de haber ejecutado el algoritmo GridSearchCV(), se obtiene que los mejores hiper parámetros son: {'C':1, 'gamma':'auto', 'kernel':'rbf'} El tiempo de ejecución del algoritmo de búsqueda de rejilla es de aproximadamente 36 minutos con 32 segundos.


--------------------------------------------------------APLICACIÓN DEL MODELO XGBOOST------------------------------------------------------------------------------------

TAMAÑO DE MUESTRA DE 2.000 DATOS

SIN ELIMINACIÓN DE DATOS ATÍPICOS

En la primera celda de esta sección, se utiliza el método train_test_split() para tomar exactamente dos mil datos del dataset de entrenamiento. Es importante que se ejecute primero que las celdas posteriores para así poder sobre escribir las variables X_train, X_test, Y_train e Y_test.

Posteriormente, en la segunda celda de esta sección, se realiza una búsqueda de rejilla para cuantificar cuáles son los valores de los mejores hiperparámetros del modelo. Los hiper parámetros que se utilizaron dentro de la búsqueda fueron: learning_rate, n_estimators y max_depth. Es importante hacer uso del hiper parámetro n_jobs = -1 para que se empleen todos los núcleos del procesador con el que cuente la máquina en la cual se esté ejecutando el algoritmo.

Luego de haber ejecutado el algoritmo GridSearchCV(), se obtiene que los mejores hiper parámetros son: {'learning_rate':0.03, 'n_estimators':500, 'max_depth':3} El tiempo de ejecución del algoritmo de búsqueda de rejilla es de aproximadamente 11.6 segundos.


TAMAÑO DE MUESTRA DE 8.000 DATOS

SIN ELIMINACIÓN DE DATOS ATÍPICOS

En la primera celda de esta sección, se utiliza el método train_test_split() para tomar exactamente ocho mil datos del dataset de entrenamiento. Es importante que se ejecute primero que las celdas posteriores para así poder sobre escribir las variables X_train, X_test, Y_train e Y_test

Posteriormente, en la segunda celda de esta sección, se realiza una búsqueda de rejilla para cuantificar cuáles son los valores de los mejores hiperparámetros del modelo. Los hiper parámetros que se utilizaron dentro de la búsqueda fueron: learning_rate, n_estimators y max_depth. Es importante hacer uso del hiper parámetro n_jobs = -1 para que se empleen todos los núcleos del procesador con el que cuente la máquina en la cual se esté ejecutando el algoritmo.

Luego de haber ejecutado el algoritmo GridSearchCV(), se obtiene que los mejores hiper parámetros son: {'learning_rate':0.1, 'n_estimators':100, 'max_depth':3} El tiempo de ejecución del algoritmo de búsqueda de rejilla es de aproximadamente 32.1 segundos.


TAMAÑO DE MUESTRA DE 40.000 DATOS

SIN ELIMINACIÓN DE DATOS ATÍPICOS

En la primera celda de esta sección, se utiliza el método train_test_split() para tomar exactamente cuarenta mil datos del dataset de entrenamiento. Es importante que se ejecute primero que las celdas posteriores para así poder sobre escribir las variables X_train, X_test, Y_train e Y_test

Posteriormente, en la segunda celda de esta sección, se realiza una búsqueda de rejilla para cuantificar cuáles son los valores de los mejores hiperparámetros del modelo. Los hiper parámetros que se utilizaron dentro de la búsqueda fueron: learning_rate, n_estimators y max_depth. Es importante hacer uso del hiper parámetro n_jobs = -1 para que se empleen todos los núcleos del procesador con el que cuente la máquina en la cual se esté ejecutando el algoritmo.

Luego de haber ejecutado el algoritmo GridSearchCV(), se obtiene que los mejores hiper parámetros son: {'learning_rate':0.06, 'n_estimators':500, 'max_depth':3} El tiempo de ejecución del algoritmo de búsqueda de rejilla es de aproximadamente 2.31 minutos.


TAMAÑO DE MUESTRA DE 2.000 DATOS

CON ELIMINACIÓN DE DATOS ATÍPICOS

En la primera celda de esta sección, se utiliza el método train_test_split() para tomar exactamente dos mil datos del dataset de entrenamiento. Es importante que se ejecute primero que las celdas posteriores para así poder sobre escribir las variables X_train_LOF, X_test_LOF, Y_train_LOF e Y_test_LOF.

Posteriormente, en la segunda celda de esta sección, se realiza una búsqueda de rejilla para cuantificar cuáles son los valores de los mejores hiperparámetros del modelo. Los hiper parámetros que se utilizaron dentro de la búsqueda fueron: learning_rate, n_estimators y max_depth. Es importante hacer uso del hiper parámetro n_jobs = -1 para que se empleen todos los núcleos del procesador con el que cuente la máquina en la cual se esté ejecutando el algoritmo.

Luego de haber ejecutado el algoritmo GridSearchCV(), se obtiene que los mejores hiper parámetros son: {'learning_rate':0.03, 'n_estimators':500, 'max_depth':1} El tiempo de ejecución del algoritmo de búsqueda de rejilla es de aproximadamente 9.64 segundos.


TAMAÑO DE MUESTRA DE 8.000 DATOS

CON ELIMINACIÓN DE DATOS ATÍPICOS

En la primera celda de esta sección, se utiliza el método train_test_split() para tomar exactamente ocho mil datos del dataset de entrenamiento. Es importante que se ejecute primero que las celdas posteriores para así poder sobre escribir las variables X_train_LOF, X_test_LOF, Y_train_LOF e Y_test_LOF

Posteriormente, en la segunda celda de esta sección, se realiza una búsqueda de rejilla para cuantificar cuáles son los valores de los mejores hiperparámetros del modelo. Los hiper parámetros que se utilizaron dentro de la búsqueda fueron: learning_rate, n_estimators y max_depth. Es importante hacer uso del hiper parámetro n_jobs = -1 para que se empleen todos los núcleos del procesador con el que cuente la máquina en la cual se esté ejecutando el algoritmo.

Luego de haber ejecutado el algoritmo GridSearchCV(), se obtiene que los mejores hiper parámetros son: {'learning_rate':0.03, 'n_estimators':500, 'max_depth':3} El tiempo de ejecución del algoritmo de búsqueda de rejilla es de aproximadamente 31.8 segundos.


TAMAÑO DE MUESTRA DE 40.000 DATOS

CON ELIMINACIÓN DE DATOS ATÍPICOS

En la primera celda de esta sección, se utiliza el método train_test_split() para tomar exactamente cuarenta mil datos del dataset de entrenamiento. Es importante que se ejecute primero que las celdas posteriores para así poder sobre escribir las variables X_train_LOF, X_test_LOF, Y_train_LOF e Y_test_LOF

Posteriormente, en la segunda celda de esta sección, se realiza una búsqueda de rejilla para cuantificar cuáles son los valores de los mejores hiperparámetros del modelo. Los hiper parámetros que se utilizaron dentro de la búsqueda fueron: learning_rate, n_estimators y max_depth. Es importante hacer uso del hiper parámetro n_jobs = -1 para que se empleen todos los núcleos del procesador con el que cuente la máquina en la cual se esté ejecutando el algoritmo.

Luego de haber ejecutado el algoritmo GridSearchCV(), se obtiene que los mejores hiper parámetros son: {'learning_rate':0.06, 'n_estimators':500, 'max_depth':3} El tiempo de ejecución del algoritmo de búsqueda de rejilla es de aproximadamente 2.31 minutos.


--------------------------------------------------------APLICACIÓN DEL MODELO ADABOOST------------------------------------------------------------------------------------

TAMAÑO DE MUESTRA DE 2.000 DATOS

SIN ELIMINACIÓN DE DATOS ATÍPICOS

En la primera celda de esta sección, se utiliza el método train_test_split() para tomar exactamente dos mil datos del dataset de entrenamiento. Es importante que se ejecute primero que las celdas posteriores para así poder sobre escribir las variables X_train, X_test, Y_train e Y_test.

Posteriormente, en la segunda celda de esta sección, se realiza una búsqueda de rejilla para cuantificar cuáles son los valores de los mejores hiperparámetros del modelo. Los hiper parámetros que se utilizaron dentro de la búsqueda fueron: learning_rate, n_estimators y algorithm. Es importante hacer uso del hiper parámetro n_jobs = -1 para que se empleen todos los núcleos del procesador con el que cuente la máquina en la cual se esté ejecutando el algoritmo.

Luego de haber ejecutado el algoritmo GridSearchCV(), se obtiene que los mejores hiper parámetros son: {'learning_rate':0.06, 'n_estimators':100, 'algorithm':SAMME} El tiempo de ejecución del algoritmo de búsqueda de rejilla es de aproximadamente 18 segundos.


TAMAÑO DE MUESTRA DE 8.000 DATOS

SIN ELIMINACIÓN DE DATOS ATÍPICOS

En la primera celda de esta sección, se utiliza el método train_test_split() para tomar exactamente ocho mil datos del dataset de entrenamiento. Es importante que se ejecute primero que las celdas posteriores para así poder sobre escribir las variables X_train, X_test, Y_train e Y_test

Posteriormente, en la segunda celda de esta sección, se realiza una búsqueda de rejilla para cuantificar cuáles son los valores de los mejores hiperparámetros del modelo. Los hiper parámetros que se utilizaron dentro de la búsqueda fueron: learning_rate, n_estimators y algorithm. Es importante hacer uso del hiper parámetro n_jobs = -1 para que se empleen todos los núcleos del procesador con el que cuente la máquina en la cual se esté ejecutando el algoritmo.

Luego de haber ejecutado el algoritmo GridSearchCV(), se obtiene que los mejores hiper parámetros son: {'learning_rate':0.06, 'n_estimators':100, 'algorithm':SAMME} El tiempo de ejecución del algoritmo de búsqueda de rejilla es de aproximadamente 42.2 segundos.


TAMAÑO DE MUESTRA DE 40.000 DATOS

SIN ELIMINACIÓN DE DATOS ATÍPICOS

En la primera celda de esta sección, se utiliza el método train_test_split() para tomar exactamente cuarenta mil datos del dataset de entrenamiento. Es importante que se ejecute primero que las celdas posteriores para así poder sobre escribir las variables X_train, X_test, Y_train e Y_test

Posteriormente, en la segunda celda de esta sección, se realiza una búsqueda de rejilla para cuantificar cuáles son los valores de los mejores hiperparámetros del modelo. Los hiper parámetros que se utilizaron dentro de la búsqueda fueron: learning_rate, n_estimators y algorithm. Es importante hacer uso del hiper parámetro n_jobs = -1 para que se empleen todos los núcleos del procesador con el que cuente la máquina en la cual se esté ejecutando el algoritmo.

Luego de haber ejecutado el algoritmo GridSearchCV(), se obtiene que los mejores hiper parámetros son: {'learning_rate':0.3, 'n_estimators':500, 'algorithm':SAMME.R} El tiempo de ejecución del algoritmo de búsqueda de rejilla es de aproximadamente 3.34 minutos.


TAMAÑO DE MUESTRA DE 2.000 DATOS

CON ELIMINACIÓN DE DATOS ATÍPICOS

En la primera celda de esta sección, se utiliza el método train_test_split() para tomar exactamente dos mil datos del dataset de entrenamiento. Es importante que se ejecute primero que las celdas posteriores para así poder sobre escribir las variables X_train_LOF, X_test_LOF, Y_train_LOF e Y_test_LOF.

Posteriormente, en la segunda celda de esta sección, se realiza una búsqueda de rejilla para cuantificar cuáles son los valores de los mejores hiperparámetros del modelo. Los hiper parámetros que se utilizaron dentro de la búsqueda fueron: learning_rate, n_estimators y algorithm. Es importante hacer uso del hiper parámetro n_jobs = -1 para que se empleen todos los núcleos del procesador con el que cuente la máquina en la cual se esté ejecutando el algoritmo.

Luego de haber ejecutado el algoritmo GridSearchCV(), se obtiene que los mejores hiper parámetros son: {'learning_rate':0.06, 'n_estimators':1000, 'algorithm':SAMME} El tiempo de ejecución del algoritmo de búsqueda de rejilla es de aproximadamente 19.6 segundos.


TAMAÑO DE MUESTRA DE 8.000 DATOS

CON ELIMINACIÓN DE DATOS ATÍPICOS

En la primera celda de esta sección, se utiliza el método train_test_split() para tomar exactamente ocho mil datos del dataset de entrenamiento. Es importante que se ejecute primero que las celdas posteriores para así poder sobre escribir las variables X_train_LOF, X_test_LOF, Y_train_LOF e Y_test_LOF

Posteriormente, en la segunda celda de esta sección, se realiza una búsqueda de rejilla para cuantificar cuáles son los valores de los mejores hiperparámetros del modelo. Los hiper parámetros que se utilizaron dentro de la búsqueda fueron: learning_rate, n_estimators y algorithm. Es importante hacer uso del hiper parámetro n_jobs = -1 para que se empleen todos los núcleos del procesador con el que cuente la máquina en la cual se esté ejecutando el algoritmo.

Luego de haber ejecutado el algoritmo GridSearchCV(), se obtiene que los mejores hiper parámetros son: {'learning_rate':0.3, 'n_estimators':1000, 'algorithm':SAMME} El tiempo de ejecución del algoritmo de búsqueda de rejilla es de aproximadamente 46.2 segundos.


TAMAÑO DE MUESTRA DE 40.000 DATOS

CON ELIMINACIÓN DE DATOS ATÍPICOS

En la primera celda de esta sección, se utiliza el método train_test_split() para tomar exactamente cuarenta mil datos del dataset de entrenamiento. Es importante que se ejecute primero que las celdas posteriores para así poder sobre escribir las variables X_train_LOF, X_test_LOF, Y_train_LOF e Y_test_LOF

Posteriormente, en la segunda celda de esta sección, se realiza una búsqueda de rejilla para cuantificar cuáles son los valores de los mejores hiperparámetros del modelo. Los hiper parámetros que se utilizaron dentro de la búsqueda fueron: learning_rate, n_estimators y algorithm. Es importante hacer uso del hiper parámetro n_jobs = -1 para que se empleen todos los núcleos del procesador con el que cuente la máquina en la cual se esté ejecutando el algoritmo.

Luego de haber ejecutado el algoritmo GridSearchCV(), se obtiene que los mejores hiper parámetros son: {'learning_rate':0.1, 'n_estimators':500, 'algorithm':SAMME.R} El tiempo de ejecución del algoritmo de búsqueda de rejilla es de aproximadamente 3.30 minutos.


***************************************************************************IMPLEMENTACIÓN DE CROSS-VALIDATION*******************************************************************************

Una vez se encontraron los mejores hiper parámetros para cada uno de los modelos tanto para el escenario con y sin eliminación de datos atípicos, se implementó un método de validación cruzada, el cual se describe a continuación: 

+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ESCENARIO SIN ELIMINACIÓN DE DATOS ATÍPICOS------------------------------------------------------------------------------------

En la primera celda de esta sección, se ejecuta nuevamente una partición aleatoria del dataset para tomar 80% de este para ejecutar un ajuste con los modelos cuyos mejores hiperparámetros fueron los que provienen de las múltiples búsquedas de rejilla implementadas. El método fit() permitió llevar acabo este ajuste con todos los modelos evaluados en un tiempo estimado de 21 minutos con 20 segundos.

Posteriormente, del 20% del dataset que no se utilizó para hacer ajuste en los modelos, se extrajo un 10% adicional para llevar a cabo el cross validation. Esta partición se realizó con otro train_test_split() en el cual se pasaron como argumentos X_test, Y_test y train_size = 0.5.

Finalmente, se realiza el cross validation con ayuda del método cross_validate() dentro de un ciclo for, haciendo uso a su vez del parámetro n_jobs = -1, el cual permite paralelizar el algoritmo dentro de todos los núcleos del procesador de la máquina.

Luego de un tiempo aproximado de 56 minutos con 46 segundos, se logró el mayor test_f1 con el modelo modelSVC. 

En la sexta celda de esta sección se graficó la curva ROC, la cual arrojó resultados de área bajo la curva muy similares entre los modelos. Este resultado se evidencia a su vez en el gráfico de caja y bigotes que se grafica posteriormente con ayuda de la librería seaborn.

Por último, el 10% restante del dataset que no se implementó ni en el método fit() ni en la etapa de validación cruzada, se utilizó para llevar a cabo predicciones con el modelo SVC a través del método predict(). Estos resultados de predicción fueron graficados en una matriz de confusión que arrojó un F1 score para la clase 0 igual a 0.72 y para la clase 1 igual a 0.76.


+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ESCENARIO CON ELIMINACIÓN DE DATOS ATÍPICOS------------------------------------------------------------------------------------

En la primera celda de esta sección, se ejecuta nuevamente una partición aleatoria del dataset para tomar 80% de este para ejecutar un ajuste con los modelos cuyos mejores hiperparámetros fueron los que provienen de las múltiples búsquedas de rejilla implementadas. El método fit() permitió llevar acabo este ajuste con todos los modelos evaluados en un tiempo estimado de 16 minutos con 31 segundos.

Posteriormente, del 20% del dataset que no se utilizó para hacer ajuste en los modelos, se extrajo un 10% adicional para llevar a cabo el cross validation. Esta partición se realizó con otro train_test_split() en el cual se pasaron como argumentos X_test, Y_test y train_size = 0.5.

Finalmente, se realiza el cross validation con ayuda del método cross_validate() dentro de un ciclo for, haciendo uso a su vez del parámetro n_jobs = -1, el cual permite paralelizar el algoritmo dentro de todos los núcleos del procesador de la máquina.

Luego de un tiempo aproximado de 46 minutos con 28 segundos, se logró el mayor test_f1 con el modelo modelXGBC_LOF. 

En la sexta celda de esta sección se graficó la curva ROC, la cual arrojó resultados de área bajo la curva muy similares entre los modelos. Este resultado se evidencia a su vez en el gráfico de caja y bigotes que se grafica posteriormente con ayuda de la librería seaborn.

Por último, el 10% restante del dataset que no se implementó ni en el método fit() ni en la etapa de validación cruzada, se utilizó para llevar a cabo predicciones con el modelo XGBC_LOF a través del método predict(). Estos resultados de predicción fueron graficados en una matriz de confusión que arrojó un F1 score para la clase 0 igual a 0.74 y para la clase 1 igual a 0.77.
