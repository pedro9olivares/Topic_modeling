# Módulo 1: Topic modeling y LDA
Este repositorio contiene el código de un modelo generativo de lenguaje, uno de k-means clustering y uno de LDA. Cada modelo se implementó en Python.

## Índice
* [Introducción](#introducción)
* [Modelos generativos de lenguaje](#modelos-generativos-de-lenguaje)
* [K-means clustering](#k-means-clustering)
* [Latent Dirichlet allocation](#latent-dirichlet-allocation)

## Introducción
El modelado de temas (topic modeling) es una técnica de machine learning que se usa para descubrir qué temas o tópicos ocurren en una colección de documentos.

El modelo más popular para hacer topic modeling es el de **latent Dirichlet allocation (LDA)**, el cual es un modelo estadístico *generativo* y *no supervisado*. Para entender estos dos últimos conceptos, se implementa un **modelo generativo de lenguaje** y el modelo de **k-means clustering**, respectivamente.

(Cada modelo se implementa por separado: modelo generativo, k-means y LDA).

![image](https://user-images.githubusercontent.com/61219691/159791223-3ca9c684-4576-4072-ac02-ab648d825fb5.png)
## Modelos generativos de lenguaje
Un modelo generativo de lenguaje es un modelo estadístico que se encarga de generar texto a partir de un vocabulario dado. En nuestro caso, el vocabulario estará dado por todas las palabras contenidas en el primer libro de Harry Potter. 

Hay dos maneras de generar el texto, la primera es asumiendo una *distribución uniforme* entre todas las palabras del vocabulario y la segunda es asumiendo la *distribución real* de las palabras de nuestro vocabulario.

Se puede utilizar cualquier archivo .txt como vocabulario, si así se desea. Para hacerlo, solamente hace falta:
1. subir su .txt a un repositorio de GitHub,
2. obtener la liga raw y
3. cambiar la siguiente línea por dicha liga raw:
```python
!wget https://raw.githubusercontent.com/sharanyavb/harry-potter/master/Books_Text/HP1.txt
```

### Generación de texto con distribución uniforme
Al asumir una distribución uniforme, solamente estamos eligiendo palabras al azar de nuestro vocabulario y concatenándolas. Predeciblemente, este método no generará textos con mucho sentido.

Después de un preprocesamiento de los datos (eliminar palabras repetidas, expandir contracciones), obtenemos resultados como el siguiente:

<img width="510" alt="image" src="https://user-images.githubusercontent.com/61219691/159108532-a96fcf4b-fb05-4a7b-b9e5-41d6cb1ac872.png">

### Generación de texto con distribución real
Al tomar en cuenta la distribución real de las palabras, podremos generar un texto mucho más significativo y entendible. En nuestro ejemplo, podemos observar la siguiente distribución: 

<img width="610" alt="image" src="https://user-images.githubusercontent.com/61219691/159108128-288f91f1-fde0-4a68-8a99-b6a70b477169.png">

Así, implementamos una función `selecciona_siguiente_token(secuencia)` que inicialmente escoge una palabra al azar del vocabulario para iniciar la secuencia de palabras que se generará. Después, recibirá iterativamente la secuencia de palabras hasta entonces generada para concatenar la palabra más probable que siga. Utilizando esta función obtenemos textos más significativos, como el siguiente:

![image](https://user-images.githubusercontent.com/61219691/159108951-dc7f47ea-a0bb-4215-9f2e-9d2d579a8073.png)

## K-means clustering
El algoritmo de k-means clustering tiene como objetivo particionar una base de datos o *dataset* en k grupos o *clústers*, donde cada registro del dataset pertenece al clúster cuyo valor medio es más cercano. Este algoritmo es un ejemplo de un método de aprendizaje no supervisado, ya que *a priori* no tenemos una clasificación o etiquetado de nuestros registros.

En nuestro caso, utilizaremos un dataset donde cada registro contiene una fecha e información ambiental (temperatura, humedad, CO2, etc.) para un cierto lugar de trabajo u oficina. Lo que se busca es encontrar k-clústers que describan el nivel de comodidad que sentiría un empleado en ciertas condiciones ambientales [(link del challenge original).](https://challengedata.ens.fr/challenges/15)

Matemáticamente, si 

<img src="https://render.githubusercontent.com/render/math?math=S = \{ x_i \}_{i\leq N} "> 

es nuestro dataset con cada registro <img src="https://render.githubusercontent.com/render/math?math=x_i\in \mathbb{R}^d "> , buscamos hacer

<img src="https://render.githubusercontent.com/render/math?math=S = S_1\cup S_2\cup ...\cup S_k "> 

para un k determinado, en donde cada <img src="https://render.githubusercontent.com/render/math?math=S_i"> representa un clúster cuyo valor medio está dado por <img src="https://render.githubusercontent.com/render/math?math=\mu_i ">.

### Implementación del algoritmo
1. Seleccionar k puntos al azar dentro de S, por ejemplo, k = 3. Estos serán nuestros primeros centros de clúster <img src="https://render.githubusercontent.com/render/math?math=\mu_i ">.

![image](https://user-images.githubusercontent.com/61219691/159411472-5ed2a8ef-5230-4a11-a52b-208a44c16e4c.png)

2. Para cada <img src="https://render.githubusercontent.com/render/math?math=\x_i">, nos preguntamos a qué centro de clúster se acerca o parece más (con <img src="https://render.githubusercontent.com/render/math?math=x_i\in \mathbb{R}^2"> utilizamos la distancia euclidiana, por ejemplo).

![image](https://user-images.githubusercontent.com/61219691/159411396-73398df7-bffa-4fb7-91d7-35f488856e3c.png)

3. Como nuestra elección inicial de centros de clúster puede ser mala, calculamos un nuevo promedio de cada clúster <img src="https://render.githubusercontent.com/render/math?math=\mu_i' "> y repetimos a partir del segundo inciso, los pasos que sean necesarios, hasta obtener convergencia.

![image](https://user-images.githubusercontent.com/61219691/159411514-7a105fb5-c781-45f1-a005-3bebdb37a730.png)



## Latent Dirichlet allocation

Dos problemas que surgen en k-means son que no se permite la intersección entre clústers o tópicos y la famosa maldición de la dimensión. Para remediar estos incovenientes hacemos uso de LDA que, aparte de ser generativo y no supervisado, es un modelo estadístico que hace uso de **variables latentes**. 

### Variables latentes
Las variables latentes, obtenidas a través de un PCA (principal component analysis), permiten que conjuntos de documentos puedan ser explicados por características no observadas, sino inferidas. 

[Un ejemplo muy popular](https://www.nature.com/articles/nature07331) y poderoso es el siguiente: dada una base de datos de material genético europeo *sin información geográfica*, se pueden inferir distancias geográficas entre individuos que reflejan el mapa de Europa.

![image](https://user-images.githubusercontent.com/61219691/159780585-a8ca4129-2d6d-47ed-9644-d8955f56b52b.png)

### Características estadísticas
En nuestro ejemplo, trabajaremos con todas las noticias de ABC del 2020 en Australia para descubrir qué temas fueron los más hablados: el surgimiento del Covid, las restricciones de viaje, las elecciones en EU, etc. Las características estadísticas que diferencian al LDA son las siguientes:
#### Distribución de temas en cada noticia (O ó ϴ) 
Al igual que en k-means, se define un hiperparámetro k que corresponderá al número de particiones. Ahora, sin embargo, a cada noticia le asignaremos su relación porcentual con cada tema.

Suponiendo 5 tópicos {0, 1, 2, 3 , 4}, las noticias muestran la siguiente distribución con respecto a los temas: 

![image](https://user-images.githubusercontent.com/61219691/159786867-07129c45-b134-47ea-adc9-1ffa90d910df.png)

#### Distribución de palabras en cada tema (μ)
A su vez, por cada tópico podemos ver qué tan frecuentemente ocurre cierta palabra:

![image](https://user-images.githubusercontent.com/61219691/159786631-106a0576-512f-48ba-92f0-d3937afdaf6f.png)

#### Visualización del modelo
Finalmente, usando la librería `pyLDAvis` podemos visualizar nuestras distribuciones para, cualitativamente, descubrir los tópicos que ocurren en nuestros documentos.

![image](https://user-images.githubusercontent.com/61219691/159788282-2a06d1da-575e-4e11-baa1-14b332f8235c.png)


