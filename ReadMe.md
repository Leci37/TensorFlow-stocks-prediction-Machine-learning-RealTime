
**INTRODUCCIÓN**

El mercado bursátil se mueve por indicadores técnicos, existen varios tipos de volatilidad, de volumen de ciclo, de velas, soportes, resistencias, medias móviles…

Una excelente página para ver todos los indicadores técnicos bursátiles es webull <https://app.webull.com/trade?source=seo-google-home> 

Imagen: webull con indicadores Estocástico, MACD y RSI

![](Aspose.Words.03869524-0d3b-41f3-9e4c-14757704de58.001.png)


Sobre las gráficas de bolsa se ha inventado TODO lo posible para predecir la bolsa , con resultados dispares, dejando claro la dificultad de predecir el comportamiento humano

Estos indicadores indican donde comprar y vender , existen muchas creencias sobre ellos (nos referimos en creencias , ya que si funcionaran siempre todos seriamos ricos) 

Todo indicador técnico se puede obtener mediante operaciones matemáticas programables

Tres ejemplos:

**RSI** o índice de fuerza relativa es un oscilador que refleja la fuerza relativa

Mayor que 70 sobrecomprado , indica que va a bajar

Menos que 70 sobrevendido, indica que va a subir 

**MACD** es el acrónimo de Moving Average Convergence / Divergence. El MACD en bolsa se utiliza para medir la robustez del movimiento del precio. A través del cruce de la línea de este indicador y de la media

Se opera atendiendo a los cruces entre estas dos líneas

O bien se opera cuando ambas superan el cero

**Vela: Estrella de la mañana** El patrón de estrella de la mañana se considera una señal de esperanza en una tendencia bajista del mercado![](Aspose.Words.03869524-0d3b-41f3-9e4c-14757704de58.002.png)



Estos indicadores estan presentes, en refutadas y populares paginas web como investing.com para ser analizados por el mercado <https://es.investing.com/equities/apple-computer-inc-technical>

La dificultad es mayúscula de predecir el precio de cualquier acción. Inflación , guerras, populismos, todo ello condiciona la economía , y se hace difícil , sino imposible predecir qué hará Vladimir Putin mañana. 

Ahí entra el principio de profecía autocumplica de explicado así por Robert K. Merton, 

La profecía que se autorrealiza es, al principio, una definición “falsa” de la situación, que despierta un nuevo comportamiento que hace que la falsa concepción original de la situación se vuelva  “verdadera”.

![](Aspose.Words.03869524-0d3b-41f3-9e4c-14757704de58.003.png)




**OBJETIVO**

Entendiendo el principio de profecía autocumplida, se puede obtener el patrón de las mismas, mediante la recogida masiva de patrones técnicos, su cálculo y el estudio de sus patrones


Para ello , se emplearán técnica como el big data mediante librerías Pandas Python, el machine learning mediante Sklearn, XGB  y redes neuronales mediante la librería abierta de google Tensor Flow 

El resultado se mostrará de forma sencilla y amigable mediante alertas en móvil o ordenador.

Ejemplo de alerta en tiempo real a través del bot de telegram <https://t.me/Whale_Hunter_Alertbot> 

Los modelos machine learnig Sklearn, XGB y Tensor Flow , mediante el aprendizaje de los últimos meses detectan el punto de venta. Para detectar este punto de venta se han tenido en cuenta una serie de indicadores: olap\_VMAP, ma\_SMA\_50, ichi\_senkou\_a, olap\_BBAND\_dif ,mtum\_MACD\_ext, olap\_BBAND\_MIDDLE, mtum\_MACD\_ext\_signal, fibo\_s1, volu\_PVI\_1, ma\_KAMA\_5, etcétera.

En la imagen se muestran: MACD, RSI , estocástico y Balance of power (Elder Ray) 

La alerta es mandada en la línea vertical, durante los 4 siguientes periodos la acción decrece un 2,4%. Cada periodo-vela de la imagen indica 15 minutos.

![](Aspose.Words.03869524-0d3b-41f3-9e4c-14757704de58.004.png)![](Aspose.Words.03869524-0d3b-41f3-9e4c-14757704de58.005.png)




**FUNCIONAMIENTO**

**#1 Recogida de datos**

Recoger datos para entrenar el modelo

yhoo\_generate\_big\_all\_csv.py

Se obtienen los datos de cierre , mediante yahoo API finance , y se calculan centenares de patrones técnicos mediante las librerías pandas\_ta y talib 

yhoo\_history\_stock.get\_SCALA\_csv\_stocks\_history\_Download\_list()

El modelo para ser capaz de entrenarse en detectar puntos de compra y de venta, crea la columna buy\_seel\_point tiene valor de: 0, -100, 100. Estos son detectados en función de los máximos cambios, (positivos 100, negativo -100) en el histórico de los últimos meses, este punto será con los que se entrene el entrenamiento, también llamado el *ground true* 

Se asignará valor en buy\_seel\_point si el incremento o decremento de la acción es mayor del 2,5% en un periodo de 3 horas, mediante la funcion get\_buy\_sell\_points\_Roll

Una vez obtenidos los datos históricos de la acción y calculados todos los indicadores técnicos, un total de 1068, se generan ficheros de tipo AAPL\_stock\_history\_MONTH\_3\_AD.csv

Ejemplo del fichero con los ocho primeros indicadores:

![](Aspose.Words.03869524-0d3b-41f3-9e4c-14757704de58.006.png)

Esta obtención de datos es personalizable, se puede obtener y entrenar modelos de cualquier acción del Nasdaq, para otros indicadores o cripto-activos , también es posible mediante pequeños cambios.

A través de la clase **Option\_Historical** existe la posibilidad de crear ficheros de datos históricos: anuales, mensuales y  diarios.

**class Option\_Historical**(Enum): YEARS\_3 = 1, MONTH\_3 = 2, MONTH\_3\_AD = 3, DAY\_6 = 4, DAY\_1 = 5

Se generan los ficheros *\d\_price\min\_max\AAPL\_min\_max\_stock\_MONTH\_3.csv* , los cuales guardan el max y min valor de cada columna , para que sea leido en Model\_predictions\_Nrows.py para un  rapido  fit\_scaler() (es el proceso “limpieza” que requieren los datos antes de entrar en los modelos de entrenamiento AI)  . Esta operación es de vital importancia para una correcta optimización en la lectura de datos en tiempo real.




**#1.1 Tipos de indicadores**

Durante la generación de fichero de recogida de datos del punto 1 AAPL\_stock\_history\_MONTH\_3\_AD.csv se calculan 1068 indicadores técnicos, los cuales se dividen en subtipos, en función de **prefijos** en el nombre.

Lista de prefijos y ejemplo de nombre de alguno de ellos.

- Overlap o superposición:  **olap\_**

olap\_BBAND\_UPPER, olap\_BBAND\_MIDDLE, olap\_BBAND\_LOWER,** 

- Momentum:  **mtum\_**

mtum\_MACD, mtum\_MACD\_signal, mtum\_RSI, mtum\_STOCH\_k,

- Volatilidad: **vola\_**

vola\_KCBe\_20\_2, vola\_KCUe\_20\_2, vola\_RVI\_14

- Patrones de ciclo: **cycl\_**

cycl\_DCPHASE, cycl\_PHASOR\_inph, cycl\_PHASOR\_quad

- Patrones de velas: **cdl\_**

cdl\_RICKSHAWMAN, cdl\_RISEFALL3METHODS, cdl\_SEPARATINGLINES

- Estadística: **sti\_**

sti\_STDDEV, sti\_TSF, sti\_VAR

- Medias móviles: **ma\_**

ma\_SMA\_100, ma\_WMA\_10, ma\_DEMA\_20, ma\_EMA\_100, ma\_KAMA\_10, 

- Tendencia: **tend\_** y **ti\_**

tend\_renko\_TR, tend\_renko\_brick, ti\_acc\_dist, ti\_chaikin\_10\_3

- Resistencias y soportes sufijos:  **\_s3, \_s2, \_s1, \_pp, \_r1, \_r2, \_r3**

fibo\_s3, fibo\_s2, fibo\_s1, fibo\_pp, fibo\_r1, fibo\_r2, fibo\_r3

demark\_s1, demark\_pp, demark\_r1

- Punto de intersección con resistencia o soporte: **pcrh\_**

pcrh\_demark\_s1, pcrh\_demark\_pp, pcrh\_demark\_r1

- Punto de intersección con media móvil o de medias móviles entre ellas: **mcrh\_**

mcrh\_SMA\_20\_TRIMA\_50, mcrh\_SMA\_20\_WMA\_50, mcrh\_SMA\_20\_DEMA\_100

- Indicadores de cambios en el índice bursátil , nasdaq: **NQ\_**

NQ\_SMA\_20, NQ\_SMA\_100

Nota: Para ver los 1068 indicadores usados ir a las hojas adjuntas al final del documento.


**#2 Filtrado de indicadores**

` `**ejecutar para saber qué columnas son relevantes para la obtención del modelo**

Feature\_selection\_create\_json.py

Hay que saber cuáles de las centenares columnas de datos técnicos , es válida para entrenar el modelo neuronal, y cuales son solo ruido. Esto se hará mediante correlaciones y modelos de Random Forest.

Responde a la pregunta:

¿qué columnas son las más relevantes para puntos de compra o venta?

Genera los ficheros *best\_selection*, los cuales son un raking de mejores datos técnicos para entrenar el modelo , se pretende pasar de 1068 columnas a unas 120

Por ejemplo, para la acción Amazon, detección puntos de compra,  en el periodo Junio a Octubre 2022, los indicadores más valiosos son:

- Senkuo de la nube de Ichimoku
- Volatilidad de Chaikin 
- On-balance volume

Ejemplo del fichero *plots\_relations/best\_selection\_AMNZ\_pos.json*

**"index"**: {

`  `**"12"**: [

`     `**"ichi\_senkou\_b"**
**
`  `],

`  `**"10"**: [

`     `**"volu\_Chaikin\_AD"**
**
`  `],

`  `**"9"**: [

`     `**"volu\_OBV"**
**
`  `],



**#3 entrenar modelos TensorFlow, XGB y Sklearn** 

Model\_creation\_models\_for\_a\_stock.py

para ello se requiere la selección de mejores columnas del punto #2

Hay cuatro tipos de algoritmos predictivos, modelos AI:

- **Gradient Boosting** está formado por un conjunto de [árboles de decisión](https://www.cienciadedatos.net/documentos/py07_arboles_decision_python.html) individuales, entrenados de forma secuencial, de forma que cada nuevo árbol trata de mejorar los errores de los árboles anteriores. Librería Sklearn
- **Random Forest** Los bosques aleatorios  son un método de aprendizaje conjunto para clasificación, regresión y otras tareas que opera mediante la construcción de una multitud de árboles de decisión en el momento del entrenamiento. Librería Sklearn 
- **XGBoost** es una biblioteca de aumento de gradiente distribuida optimizada diseñada para ser altamente eficiente, flexible y portátil. Implementa algoritmos de aprendizaje automático bajo el marco Gradient Boosting. Librería XGBoost 
- **TensorFlow TF** es una biblioteca de código abierto para aprendizaje automático a través de un rango de tareas, y desarrollado por Google para satisfacer sus necesidades de sistemas capaces de construir y entrenar redes neuronales para detectar y descifrar patrones y correlaciones, análogos al aprendizaje y razonamiento usados por los humanos. Librería TensorFlow 


Existen modelos POS (compra) o modelo NEG (venta) y existe un modelo BOTH (BOTH se descarta,  ya que los modelos de predicción son binario , sólo aceptan 2 posiciones, verdad o mentira)

Este punto genera modelos de predicción .sav para XGB y Sklearn. .h5  para Tensor Flow

Ejemplos de nombrado: XGboost\_U\_neg\_vgood16\_s28.sav y TF\_AMZN\_pos\_low1\_s128.h5

Formato de los nombres:

- Tipo de AI con las que se entrena puede ser:
  - ` `XGboost, TF, TF64, GradientBoost y RandomForest
- Ticker de la acción AMZN para amazon , AAPL para Apple …
- Detecta puntos de compra o de venta pos o neg
- Cuantos indicadores han sido usados en el aprendizaje, pueden ser de 4 tipos en función de la relevancia dada por el punto *#2 Filtrado de indicadores*. Este ranking es organizado en la clase **MODEL\_TYPE\_COLM**, 
  - vgood16 los mejores 16 indicadores
  - good9 los mejores 32 indicadores
  - reg4 los mejores 64 indicadores 
  - low1 los mejores 128 indicadores 
- Solo para modelos TF. En función de la densidad de las neuronas usadas, definidas en la clase a\_manage\_stocks\_dict.**MODEL\_TF\_DENSE\_TYPE\_ONE\_DIMENSI** puede tomar valor: s28 s64 y s128

Estas combinaciones implican que por cada acción se crean 5 tipos de AI, cada una en pos y neg, además por cada combinación se añade las 4 configuraciones de indicadores técnicos.  Esto genera 40 modelos de AI, los cuales serán seleccionados en el punto: *#4 evaluar la CALIDAD de esos modelos*

Cada vez que se genera un modelo IA, se genera un fichero de registro: *TF\_balance\TF\_AAPL\_pos\_reg4.h5\_accuracy\_87.6%\_\_loss\_2.74\_\_epochs\_10[160].csv*

Este contiene los datos de precisión y de pérdida del modelo, así como los registros del entrenamiento del mismo.




**#4 evaluar la CALIDAD de esos modelos** 

Model\_creation\_scoring.py

Para hacer una predicción con los AI, se recogen nuevos datos y se calculan los indicadores técnicos con los que ha sido entrenado según los ficheros *best\_selection*.

Cuando los modelos .h5 y .sav son preguntados:

` `¿Esto es un punto de compra-venta? 

Estos responden un número que puede variar entre 0,1 y 4 

Cuanto más alto sea el número mayor probabilidad de que sea un punto de compra-venta correcto.

Cada modelo tiene una escala de puntuación en el cual se considera punto de compra venta. para unos modelos con una puntuación de más 0,4 será suficiente (normalmente los XGboost) , mientras que para otros requieren más de 1,5  (normalmente los TF).

¿Cómo se sabe cuál es la puntuación umbral para cada modelo ?

La clase Model\_creation\_scoring.py  genera los ficheros umbral de puntuación *threshold*, los cuales dicen cual es el punto umbral en el cual se considera punto de compra-venta.

Cada modelo AI contará con su propio fichero de tipo:

*Models/Scoring/AAPL\_neg\_\_when\_model\_ok\_threshold.csv*

Por cada acción el punto #*3 entrenar los modelos TF, XGB y Sklearn*  se generan 40 modelos de AI. Esta clase evalúa y selecciona los modelos más precisos de tal forma que solo se ejecutarán en tiempo real los más precisos (normalmente se seleccionan entre 4 y 8)

*Models/Scoring/AAPL\_neg\_\_groupby\_buy\_sell\_point\_000.json*

**"list\_good\_params"**: [

`  `**"r\_rf\_AFRM\_pos\_low1\_"**,

`  `**"r\_TF64\_AFRM\_pos\_vgood16\_"**,

`  `**"r\_TF64\_AFRM\_pos\_good9\_"**,

`  `**"r\_TF\_AFRM\_pos\_reg4\_"**

],


**#4.1 evaluar esos BENEFICIO real de modelos**

Model\_predictions\_N\_eval\_profits.py

Responde a la pregunta: 

¿Si se deja N días ejecutándose, cuánto dinero hipotético se gana ?

Nota: esto debe ejecutarse con datos que no hayan sido usados en el modelo de entrenamiento, preferentemente

*Models/eval\_Profits/\_AAPL\_neg\_ALL\_stock\_20221021\_\_20221014.csv*


**#5 Hacer predicciones de la última semana**

Model\_predictions\_Nrows.py

Se puede hacer predicciones con los datos en tiempo real de la acción.

A través de la llamada a la función cada 10-12min, descarga los datos de la acción en tiempo real a través de la API financiera de yahoo.

df\_compar, df\_vender = get\_RealTime\_buy\_seel\_points()

Esta ejecución genera el fichero de registro *d\_result/prediction\_results\_N\_rows.csv*

Este fichero contiene información sobre cada predicción que se ha realizado. Contiene las siguientes columnas:

- Date: fecha de las prediccion 
- Stock: accion 
- buy\_sell: puede ser NEG o POS , dependiendo de que sea operación de compra o venta 
- Close: Es el valor escalado del valor de cierre (no el valor real)
- Volume: Es el valor escalado del Volumen (no el valor real)
- 88%: Formato fraccion  ( **5/6** ) ¿Cuántos modelos han predicho que es un punto de operación válido por encima del 88%? Cinco de los seis analizados 
- 93%: Formato fraccion  ( **5/6** ), número de modelos por encima de 93%
- 95%: Formato fraccion  ( **5/6** ), número de modelos por encima de 95%
- TF: Formato fraccion  ( **5/6** ), número de modelos por encima de 93%, cuya prediccion se ha hecho con modelos Tensor Flow 
- Models\_names: nombre de los modelos que han dado positivo, con el % de acierto (88%, 93%, 95%) como sufijo 

Ejemplo de registro

` `**2022-11-07 16:00:00 MELI NEG -51.8 -85.80 5/6 0/6 0/6 1/2 TF\_reg4\_s128\_88%, rf\_good9\_88%, rf\_low1\_88%, rf\_reg4\_88%, rf\_vgood16\_88%,**

Para ser considerada una predicción válida para operar, al menos,  debe tener la mitad de la fracción de puntuación en las columnas  93% y TF .

Más de la mitad de los modelos han predicho con una puntuación superior al 93% que es un punto bueno para operar 



**#5.1 mandar alertas en tiempo real**

yhoo\_POOL\_enque\_Thread.py *multihilo encolado 2s por accion* 

Existe la posibilidad de mandar las alertas de compra venta de la acción, al telegram o mail

se evalúan los múltiples modelos entrenados AI , y solo los mayores de 96% de probabilidad (según lo entrenado previamente) son notificados

Cada 15 minutos , se calculan **todos** los indicadores en real time por cada acción y se evalúan en los modelos AI

En la alerta se indica qué modelos están detectando puntos de compra y venta, correctos en los que ejecutar la transacción 

Estas alertas de compra-venta caducan en, más menos 7 minutos, dado la volatilidad del mercado

También se adjunta el precio al que se detectó, la hora, y los enlaces a las web de noticias.

Nota: las noticias financieras siempre deben prevalecer a los indicadores técnicos. 

Lo que se muestra en alerta DEBUG, es la información de *d\_result/prediction\_results\_N\_rows.csv* del Punto: 5 hacer predicciones de la última semana Test

Enlace del Bot telegram : <https://t.me/Whale_Hunter_Alertbot> 

![](Aspose.Words.03869524-0d3b-41f3-9e4c-14757704de58.007.png)





[PUESTA EN MARCHA RÁPIDA](#_jl51ycf02hp3)

[PUESTA EN MARCHA](#_xzwbezl2nv0n)

[#1 Recogida de datos históricos](#_aib4v2ze0vtf)

[1.0 (Recomendado) alphavantage API](#_jqsb0sh385h7)

[1.1 Hay que generar el histórico de OHLCV  de la acción](#_9imsoa6gq6c4)

[2 Filtrado de indicadores técnicos](#_xpflmg3gomi4)

[3 Generar entrenamiento de modelos TensorFlow, XGB y Sklearn](#_n6rx9fxjb28o)

[4 Evaluar calidad de los modelos predictivos](#_mkpwg03yocx5)

[5 Predicciones](#_ud7j5t10ip07)

[5.0 hacer predicciones de la última semana Test Opcional](#_4g0zhpdmgc9l)

[5.1 Conseguir datos OHLCV en tiempo real](#_eb8e4tbg8zc2)

[5.2 Configurar chatID y tokes en Telegram](#_fok3ybskh0b)

[5.3 Envío de alertas en tiempo real Telegram](#_389cmg1t5cxg)

[Mejoras en los modelos predictivos, usando multidimensionales](#_7lsrywqwgsc9)

[TODO Lista de mejoras sugeridas:](#_pjkcmb35dfbs)

[Nombres de indicadores:](#_hkuvz7j2vziw)




### PUESTA EN MARCHA RÁPIDA
Instalar requisitos 

pip install -r requirements.txt

Ejecutar Utils/API\_alphavantage\_get\_old\_history.py

Ejecutar yhoo\_generate\_big\_all\_csv.py

Ejecutar Model\_creation\_models\_for\_a\_stock.py

Ejecutar Model\_creation\_scoring.py

Ejecutar Model\_predictions\_Nrows.py   Opcional, predicciones ultima semana 

Predicciones tiempo real:

Ejecutar Utils/Volume\_WeBull\_get\_tikcers.py Ignorar en caso de usar configuración por defecto 

Configurar token bot ver punto  **5.2** Configurar chatID y tokes en Telegram

Ejecutar predict\_POOL\_enque\_Thread.py


### PUESTA EN MARCHA
(Los tiempos de ejecución son estimados para un intel i3 y 8GB de RAM)


**0.0** El intérprete con el que se ha hecho el tutorial es python 3.8 , IDE Pycharm, precaución con la compatibilidad entre versiones de la librería pandas y python
Por ejemplo: a dia de hoy no usar python 3.10 , ya que es incompatible con pandas <https://stackoverflow.com/questions/69586701/unable-to-install-pandas-for-python> 


**0.1** Descargar e instalar requisitos, el proyecto es potente y exigente en cuanto a librerías

pip install -r requirements.txt

**0.2** Buscar en todos los ficheros la cadena ***\*\*DOCU\*\****

esto permite echar un ojo a todos los ficheros que son ejecutables  del tutorial puesta en marcha fácilmente 

**0.3** En el fichero a\_manage\_stocks\_dict.py se guardan todas las configuraciones, observar el documento  y saber donde se encuentra.

En él existe el diccionario DICT\_COMPANYS

El cual contiene los ID (GameStops cotiza con el ID: **GME**)  de las empresas a analizar se puede personalizar y crear clase a partir de los tikers del **nasdaq** , por defecto se usará la clave **@FOLO3** la cual analizará estas 39 empresas.

**"@FOLO3"**: 
\*
`   `[**"UPST"**, **"MELI"**, **"TWLO"**, **"RIVN"**, **"SNOW"**, **"LYFT"**, **"ADBE"**, **"UBER"**, **"ZI"**, **"QCOM"**, **"PYPL"**, **"SPOT"**, **"RUN"**, **"GTLB"**, **"MDB"**, **"NVDA"**, **"AMD"** , **"ADSK"**, **"AMZN"**, **"CRWD"**, **"NVST"**, **"HUBS"**, **"EPAM"**, **"PINS"**, **"TTD"**, **"SNAP"**, **"APPS"**, **"ASAN"**, **"AFRM"**, **"DOCN"**, **"ETSY"**, **"DDOG"**, **"SHOP"**, **"NIO"**, **"U"**, **"GME"**, **"RBLX"**, **"CRSR"**],

Si se desea una ejecución más rápida se recomienda borrar elementos de la lista y dejar tres

#### #1 Recogida de datos históricos
##### **1.0** (Recomendado) alphavantage API
` `La API yfinance , si se desea intervalos entre precio y precio en intervalos de 15min  está limitada a 2 meses, para obtener más datos de tiempo hasta de 2 años atrás (a más datos mejores modelos predictivos)  se usa la versión gratuita de la API <https://www.alphavantage.co/documentation/>  

Ejecutar Utils/API\_alphavantage\_get\_old\_history.py

La clase es personalizable: intervalos de accion, meses a preguntar y acción ID

Nota: al ser la versión gratuita, existe retrato entre petición y petición, para obtener un solo historico de 2 años se toma 2-3 minutos por acción 

Una vez ejecutado se rellenará la carpeta: *d\_price/RAW\_alpha* con .csv históricos OHLCV de precios de acción. Estos ficheros serán leídos en el paso siguiente. Ejemplo de nombre: alpha\_GOOG\_15min\_20221031\_\_20201112.csv

Revisar que se hayan generado uno por cada acción en *d\_price/RAW\_alpha*.


##### **1.1** Hay que generar el histórico de OHLCV  de la acción
Así como el histórico de patrones técnicos. En calcular todos los patrones técnicos tarda +-1 minuto por acción 

Ejecutar yhoo\_generate\_big\_all\_csv.py

Una vez ejecutado se rellenará la carpeta: *d\_price* con .csv históricos OHLCV de precios de acción.

Tres tipos de ficheros son generados ( Ejemplo de tipo de nombre para acción: AMD):

- *AMD\_SCALA\_stock\_history\_MONTH\_3\_AD.csv* con todos los patrones técnicos calculados y aplicado un fit scaler(-100, 100), es decir los precios de acción están escalados (tamaño: 30-90mb)
- *d\_price/min\_max/AMD\_min\_max\_stock\_MONTH\_3\_AD.csv* con las claves del escalado  (tamaño: 2-7kb)
- *AMD\_stock\_history\_MONTH\_3\_AD.csv* el histórico puro de los OHLCV (tamaño: 2-7mb)

Nota: *MONTH\_3\_AD* significa 3 meses de *API* yfinance  más el histórico recogido de alphavantage. Punto 1.0

Revisar que se hayan generado uno por cada acción.


#### 2 Filtrado de indicadores técnicos
Hay que separar los indicadores técnicos que tienen relación con los puntos de compra o venta y cuales son ruido. 20 segundos por acción 

Ejecutar Model\_creation\_scoring.py

Se generan tres ficheros por cada acción en la carpeta: *plots\_relations* , relaciones para compra “pos”, relaciones para venta “neg” y relaciones para ambos “both”

- *plots\_relations/best\_selection\_AMD\_both.json*

Estos ficheros contienen un ranking de qué indicador técnico es mejor para cada acción 

Revisar que se hayan generado tres .json por cada acción en *plots\_relations* .

#### 3 Generar entrenamiento de modelos TensorFlow, XGB y Sklearn 
Entrenar los modelos , por cada acción se entrenan 36 modelos distintos.

15 minutos por acción.

Ejecutar Model\_creation\_models\_for\_a\_stock.py

Por cada acción se generan los siguientes ficheros:

Carpeta *Models/Sklearn\_smote*:

- XGboost\_AMD\_yyy\_xxx\_.sav
- RandomForest\_AMD\_yyy\_xxx\_.sav
- XGboost\_AMD\_yyy\_xxx\_.sav

Carpeta *Models/TF\_balance*:

- TF\_AMD\_yyy\_xxx\_zzz.h5
- TF\_AMD\_yyy\_xxx\_zzz.h5\_accuracy\_71.08%\_\_loss\_0.59\_\_epochs\_10[160].csv

xxx puede tomar valor  vgood16 good9 reg4 y low1 

yyy puede tomar valor “pos” y “neg”

zzz puede tomar valor s28 s64 y s128

Revisar que se hayan generado todas las combinaciones de ficheros expuestos por cada acción en las subcarpetas de  /*Models*.


#### 4 Evaluar calidad de los modelos predictivos 
De los 36 modelos creados por cada historico OHLCV  de cada acción, sólo se ejecutarán en tiempo real los mejores, para seleccionar y evaluar esos mejores

Ejecutar Model\_creation\_scoring.py

En la carpeta *Models/Scoring*

AMD\_yyy\_\_groupby\_buy\_sell\_point\_000.json

AMD\_yyy\_\_when\_model\_ok\_threshold.csv

Revisar que se hayan generado dos por cada acción.


#### 5 Predicciones
##### **5.0** hacer predicciones de la última semana Test Opcional 
Ejecutar Model\_predictions\_Nrows.py

Esta ejecución genera el fichero de registro *d\_result/prediction\_results\_N\_rows.csv*

Genera un fichero de ejemplo con predicciones de la última semana, datos obtenidos con yfinance 

Revisar que existen los registros 


##### **5.1** Conseguir datos OHLCV en tiempo real
En caso de querer predecir acciones en la lista @FOLO3, ignorar este punto 

Es difícil conseguir los OHLCV en tiempo real, en especial el volumen (yfinance da volumen en tiempo real, pero este no es un valor correcto y al cabo de 1-2 horas cambia, haciendo inviable usar yfinance para predicciones en tiempo real)

Para obtener volúmenes correctos en tiempo real, se realizan consultas a webull, por cada acción cada 2,5 minutos, se requiere un ID webull , están descargado en caché los por defecto @FOLO3 en *a\_manage\_stocks\_dict.py.DICT\_WEBULL\_ID*

Pero si se desea usar acciones fuera de la lista @FOLO3 

En Utils/Volume\_WeBull\_get\_tikcers.py

Cambiar la lista de ejemplo:

` `list\_stocks = [**"NEWS"**, **"STOCKS"**, **"WEBULL"**, **"IDs"**]

Por los ticker nasdaq , de los webull ID que se desea obtener.

Ejecutar Utils/Volume\_WeBull\_get\_tikcers.py

Una vez ejecutado se mostrará una lista en pantalla, esa debe ser añadida en  *a\_manage\_stocks\_dict.py.DICT\_WEBULL\_ID*

***"MELI"** : 913323000,*

***"TWLO"** : 913254000,*


##### **5.2** Configurar chatID y tokes en Telegram
Hay que obtener el token telegram y crear un canal 

Se puede conseguir el token siguiendo el tutorial: <https://www.siteguarding.com/en/how-to-get-telegram-bot-api-token> 

Con el token actualizar la variable de ztelegram\_send\_message\_handle.py

*#Get from telegram*

TOKEN = **"00000000xxxxxxx"**

Una vez hecho obtenido el token se deben obtener los chatId de los usuarios y administrador 

Los usuarios solo reciben alertas de venta compra y arranque , mientras que el administrador recibe la de los usuarios además de los posibles problemas.

Para obtener los chatId de cada usuario ejecutar ztelegram\_send\_message\_UptateUser.py y a continuacion escribir cualquier mensaje al bot, aparece tanto en consola de ejecucion como al usuario el chadID 

*[>>> BOT] Message Send on 2022-11-08 22:30:31*

`	`*Text: You "User nickname " send me:* 

*"Hello world""*

` `*ChatId: "5058733760"*

`	`*From: Bot name*

`	`*Message ID: 915*

`	`*CHAT ID: 500000760*

*-----------------------------------------------*

Recoger el *CHAT ID: 500000760*

Con los chatId de los usuarios deseados, añadirlos a la lista LIST\_PEOPLE\_IDS\_CHAT

en  ztelegram\_send\_message\_handle.py


##### **5.3** Envío de alertas en tiempo real Telegram
El criterio de enviar alerta o no , está definido en el método  ztelegram\_send\_message.will\_send\_alert() .Si mas de la mitad de modelos tienen una puntuación mayor al 93% o los modelos TF tienen una puntuación mayor de 93%, se envía alerta a los usuarios consumidores 

Ejecutar predict\_POOL\_enque\_Thread.py

Esta clase hay 2 tipos de hilos 

- Producer , pregunta constantemente por los datos  OHLCV, una vez los obtienen los introduce en un cola 
- Consumer (2 hilos corren simultáneamente) van sacando los datos OHLCV de la cola , calculan los parámetros técnicos, hacen la predicción de los modelos, las registran en zTelegram\_Registers.csv , y si estas cumplen los requisitos se envían por telegram 





####
####
#### Mejoras en los modelos predictivos, usando multidimensionales 
Mejoras en los modelos predictivos TF usando tensores (múltiples matrices en el tiempo)  y no matrices (mono temporales, diseño actual). 

En la clase Model\_TF\_definitions.ModelDefinition.py

A través de ella se obtienen las configuraciones de los modelos, densidad, número de neuronas etc.

Existen dos métodos:

- get\_dicts\_models\_One\_dimension() se usa actualmente y genera configuraciones de modelos TF para matrices 
- get\_dicts\_models\_multi\_dimension() no se encuentra en uso, está preparado para dar múltiples configuraciones de modelos que usen tensores 

Existe el metodo Utils.Utils\_model\_predict.df\_to\_df\_multidimension\_array(dataframe, BACHT\_SIZE\_LOOKBACK), el cual transforma df de 2 dimensiones [columnas , filas]  a df de tres dimensiones [columnas , files, BACHT\_SIZE\_LOOKBACK ]

BACHT\_SIZE\_LOOKBACK  significa cuantos registros en pasado se añaden al df, el número es configurable y valor por defecto es ocho.

Para comenzar el desarrollo debe ser llamar al método con BACHT\_SIZE\_LOOKBACK con un valor número entero, el método devolverá un df multidimensional  [columnas , files, BACHT\_SIZE\_LOOKBACK ], con el que alimentar los modelos TF

Utils\_model\_predict.scaler\_split\_TF\_onbalance(df, label\_name=Y\_TARGET, BACHT\_SIZE\_LOOKBACK=8)

**Mejora**: Una vez se devuelven esos arrays multidimensionales, se obtienen modelos con get\_dicts\_models\_multi\_dimension() no se logra hacer entrenar un modelo y hacer una predicción con array multidimensionales 




#### **TODO** Lista de mejoras sugeridas:

Permitir analizar acciones fuera del nasdaq, cambiar en :

yhoo\_history\_stock.\_\_select\_dowload\_time\_config()

Utils/API\_alphavantage\_get\_old\_history.py

Redirigir los restantes print() a  Logger.logr.debug()

Traducir a traves de <https://www.deepl.com/> los posibles mensajes restantes en español a ingles 

Cambiar el funcionamiento del bot, que baste com mandar e comando \start , y retirar el caso de ejecucion de ztelegram\_send\_message\_UptateUser.py descrito en el punto: 5.2

Enviar alerta en tiempo real mail

Revisar Stock prediction fail LSTM 

LSTM time series + stock price prediction = FAI 

<https://www.kaggle.com/code/carlmcbrideellis/lstm-time-series-stock-price-prediction-fail> 




##### Nombres de indicadores:

**Date	buy\_sell\_point	Open	High	Low	Close	Volume	per\_Close	per\_Volume	has\_preMarket	per\_preMarket	olap\_BBAND\_UPPER	olap\_BBAND\_MIDDLE	olap\_BBAND\_LOWER	olap\_BBAND\_UPPER\_crash	olap\_BBAND\_LOWER\_crash	olap\_BBAND\_dif	olap\_HT\_TRENDLINE	olap\_MIDPOINT	olap\_MIDPRICE	olap\_SAR	olap\_SAREXT	mtum\_ADX	mtum\_ADXR	mtum\_APO	mtum\_AROON\_down	mtum\_AROON\_up	mtum\_AROONOSC	mtum\_BOP	mtum\_CCI	mtum\_CMO	mtum\_DX	mtum\_MACD	mtum\_MACD\_signal	mtum\_MACD\_list	mtum\_MACD\_crash	mtum\_MACD\_ext	mtum\_MACD\_ext\_signal	mtum\_MACD\_ext\_list	mtum\_MACD\_ext\_crash	mtum\_MACD\_fix	mtum\_MACD\_fix\_signal	mtum\_MACD\_fix\_list	mtum\_MACD\_fix\_crash	mtum\_MFI	mtum\_MINUS\_DI	mtum\_MINUS\_DM	mtum\_MOM	mtum\_PLUS\_DI	mtum\_PLUS\_DM	mtum\_PPO	mtum\_ROC	mtum\_ROCP	mtum\_ROCR	mtum\_ROCR100	mtum\_RSI	mtum\_STOCH\_k	mtum\_STOCH\_d	mtum\_STOCH\_kd	mtum\_STOCH\_crash	mtum\_STOCH\_Fa\_k	mtum\_STOCH\_Fa\_d	mtum\_STOCH\_Fa\_kd	mtum\_STOCH\_Fa\_crash	mtum\_STOCH\_RSI\_k	mtum\_STOCH\_RSI\_d	mtum\_STOCH\_RSI\_kd	mtum\_STOCH\_RSI\_crash	mtum\_TRIX	mtum\_ULTOSC	mtum\_WILLIAMS\_R	volu\_Chaikin\_AD	volu\_Chaikin\_ADOSC	volu\_OBV	vola\_ATR	vola\_NATR	vola\_TRANGE	cycl\_DCPERIOD	cycl\_DCPHASE	cycl\_PHASOR\_inph	cycl\_PHASOR\_quad	cycl\_SINE\_sine	cycl\_SINE\_lead	cycl\_HT\_TRENDMODE	cdl\_2CROWS	cdl\_3BLACKCROWS	cdl\_3INSIDE	cdl\_3LINESTRIKE	cdl\_3OUTSIDE	cdl\_3STARSINSOUTH	cdl\_3WHITESOLDIERS	cdl\_ABANDONEDBABY	cdl\_ADVANCEBLOCK	cdl\_BELTHOLD	cdl\_BREAKAWAY	cdl\_CLOSINGMARUBOZU	cdl\_CONCEALBABYSWALL	cdl\_COUNTERATTACK	cdl\_DARKCLOUDCOVER	cdl\_DOJI	cdl\_DOJISTAR	cdl\_DRAGONFLYDOJI	cdl\_ENGULFING	cdl\_EVENINGDOJISTAR	cdl\_EVENINGSTAR	cdl\_GAPSIDESIDEWHITE	cdl\_GRAVESTONEDOJI	cdl\_HAMMER	cdl\_HANGINGMAN	cdl\_HARAMI	cdl\_HARAMICROSS	cdl\_HIGHWAVE	cdl\_HIKKAKE	cdl\_HIKKAKEMOD	cdl\_HOMINGPIGEON	cdl\_IDENTICAL3CROWS	cdl\_INNECK	cdl\_INVERTEDHAMMER	cdl\_KICKING	cdl\_KICKINGBYLENGTH	cdl\_LADDERBOTTOM	cdl\_LONGLEGGEDDOJI	cdl\_LONGLINE	cdl\_MARUBOZU	cdl\_MATCHINGLOW	cdl\_MATHOLD	cdl\_MORNINGDOJISTAR	cdl\_MORNINGSTAR	cdl\_ONNECK	cdl\_PIERCING	cdl\_RICKSHAWMAN	cdl\_RISEFALL3METHODS	cdl\_SEPARATINGLINES	cdl\_SHOOTINGSTAR	cdl\_SHORTLINE	cdl\_SPINNINGTOP	cdl\_STALLEDPATTERN	cdl\_STICKSANDWICH	cdl\_TAKURI	cdl\_TASUKIGAP	cdl\_THRUSTING	cdl\_TRISTAR	cdl\_UNIQUE3RIVER	cdl\_UPSIDEGAP2CROWS	cdl\_XSIDEGAP3METHODS	sti\_BETA	sti\_CORREL	sti\_LINEARREG	sti\_LINEARREG\_ANGLE	sti\_LINEARREG\_INTERCEPT	sti\_LINEARREG\_SLOPE	sti\_STDDEV	sti\_TSF	sti\_VAR	ma\_DEMA\_5	ma\_EMA\_5	ma\_KAMA\_5	ma\_SMA\_5	ma\_T3\_5	ma\_TEMA\_5	ma\_TRIMA\_5	ma\_WMA\_5	ma\_DEMA\_10	ma\_EMA\_10	ma\_KAMA\_10	ma\_SMA\_10	ma\_T3\_10	ma\_TEMA\_10	ma\_TRIMA\_10	ma\_WMA\_10	ma\_DEMA\_20	ma\_EMA\_20	ma\_KAMA\_20	ma\_SMA\_20	ma\_T3\_20	ma\_TEMA\_20	ma\_TRIMA\_20	ma\_WMA\_20	ma\_DEMA\_50	ma\_EMA\_50	ma\_KAMA\_50	ma\_SMA\_50	ma\_T3\_50	ma\_TEMA\_50	ma\_TRIMA\_50	ma\_WMA\_50	ma\_DEMA\_100	ma\_EMA\_100	ma\_KAMA\_100	ma\_SMA\_100	ma\_T3\_100	ma\_TEMA\_100	ma\_TRIMA\_100	ma\_WMA\_100	trad\_s3	trad\_s2	trad\_s1	trad\_pp	trad\_r1	trad\_r2	trad\_r3	clas\_s3	clas\_s2	clas\_s1	clas\_pp	clas\_r1	clas\_r2	clas\_r3	fibo\_s3	fibo\_s2	fibo\_s1	fibo\_pp	fibo\_r1	fibo\_r2	fibo\_r3	wood\_s3	wood\_s2	wood\_s1	wood\_pp	wood\_r1	wood\_r2	wood\_r3	demark\_s1	demark\_pp	demark\_r1	cama\_s3	cama\_s2	cama\_s1	cama\_pp	cama\_r1	cama\_r2	cama\_r3	ti\_acc\_dist	ti\_chaikin\_10\_3	ti\_choppiness\_14	ti\_coppock\_14\_11\_10	ti\_donchian\_lower\_20	ti\_donchian\_center\_20	ti\_donchian\_upper\_20	ti\_ease\_of\_movement\_14	ti\_force\_index\_13	ti\_hma\_20	ti\_kelt\_20\_lower	ti\_kelt\_20\_upper	ti\_mass\_index\_9\_25	ti\_supertrend\_20	ti\_vortex\_pos\_5	ti\_vortex\_neg\_5	ti\_vortex\_pos\_14	ti\_vortex\_neg\_14	cycl\_EBSW\_40\_10	mtum\_AO\_5\_34	mtum\_BIAS\_SMA\_26	mtum\_AR\_26	mtum\_BR\_26	mtum\_CFO\_9	mtum\_CG\_10	mtum\_CTI\_12	mtum\_DMP\_14	mtum\_DMN\_14	mtum\_ER\_10	mtum\_BULLP\_13	mtum\_BEARP\_13	mtum\_FISHERT\_9\_1	mtum\_FISHERTs\_9\_1	mtum\_INERTIA\_20\_14	mtum\_K\_9\_3	mtum\_D\_9\_3	mtum\_J\_9\_3	mtum\_PGO\_14	mtum\_PSL\_12	mtum\_PVO\_12\_26\_9	mtum\_PVOh\_12\_26\_9	mtum\_PVOs\_12\_26\_9	mtum\_QQE\_14\_5\_4236\_RSIMA	mtum\_QQEl\_14\_5\_4236	mtum\_QQEs\_14\_5\_4236	mtum\_RSX\_14	mtum\_STC\_10\_12\_26\_05	mtum\_STCmacd\_10\_12\_26\_05	mtum\_STCstoch\_10\_12\_26\_05	mtum\_SMI\_5\_20\_5	mtum\_SMIs\_5\_20\_5	mtum\_SMIo\_5\_20\_5	olap\_ALMA\_10\_60\_085	olap\_HWMA\_02\_01\_01	olap\_JMA\_7\_0	olap\_MCGD\_10	olap\_PWMA\_10	olap\_SINWMA\_14	olap\_SSF\_10\_2	olap\_SWMA\_10	olap\_VMAP	olap\_VWMA\_10	perf\_CUMLOGRET\_1	perf\_CUMPCTRET\_1	perf\_z\_30\_1	perf\_ha	sti\_ENTP\_10	sti\_KURT\_30	sti\_TOS\_STDEVALL\_LR	sti\_TOS\_STDEVALL\_L\_1	sti\_TOS\_STDEVALL\_U\_1	sti\_TOS\_STDEVALL\_L\_2	sti\_TOS\_STDEVALL\_U\_2	sti\_TOS\_STDEVALL\_L\_3	sti\_TOS\_STDEVALL\_U\_3	sti\_ZS\_30	tend\_LDECAY\_5	tend\_PSARl\_002\_02	tend\_PSARs\_002\_02	tend\_PSARaf\_002\_02	tend\_PSARr\_002\_02	tend\_VHF\_28	vola\_HWM	vola\_HWU	vola\_HWL	vola\_KCLe\_20\_2	vola\_KCBe\_20\_2	vola\_KCUe\_20\_2	vola\_RVI\_14	vola\_THERMO\_20\_2\_05	vola\_THERMOma\_20\_2\_05	vola\_THERMOl\_20\_2\_05	vola\_THERMOs\_20\_2\_05	vola\_TRUERANGE\_1	vola\_UI\_14	volu\_EFI\_13	volu\_NVI\_1	volu\_PVI\_1	volu\_PVOL	volu\_PVR	volu\_PVT	mtum\_murrey\_math	mtum\_td\_seq	mtum\_td\_seq\_sig	tend\_hh	tend\_hl	tend\_ll	tend\_lh	tend\_hh\_crash	tend\_hl\_crash	tend\_ll\_crash	tend\_lh\_crash	ichi\_tenkan\_sen	ichi\_kijun\_sen	ichi\_senkou\_a	ichi\_senkou\_b	ichi\_isin\_cloud	ichi\_crash	ichi\_chikou\_span	tend\_renko\_TR	tend\_renko\_ATR	tend\_renko\_brick	tend\_renko\_change	pcrh\_trad\_s3	pcrh\_trad\_s2	pcrh\_trad\_s1	pcrh\_trad\_pp	pcrh\_trad\_r1	pcrh\_trad\_r2	pcrh\_trad\_r3	pcrh\_clas\_s3	pcrh\_clas\_s2	pcrh\_clas\_s1	pcrh\_clas\_pp	pcrh\_clas\_r1	pcrh\_clas\_r2	pcrh\_clas\_r3	pcrh\_fibo\_s3	pcrh\_fibo\_s2	pcrh\_fibo\_s1	pcrh\_fibo\_pp	pcrh\_fibo\_r1	pcrh\_fibo\_r2	pcrh\_fibo\_r3	pcrh\_wood\_s3	pcrh\_wood\_s2	pcrh\_wood\_s1	pcrh\_wood\_pp	pcrh\_wood\_r1	pcrh\_wood\_r2	pcrh\_wood\_r3	pcrh\_demark\_s1	pcrh\_demark\_pp	pcrh\_demark\_r1	pcrh\_cama\_s3	pcrh\_cama\_s2	pcrh\_cama\_s1	pcrh\_cama\_pp	pcrh\_cama\_r1	pcrh\_cama\_r2	pcrh\_cama\_r3	mcrh\_DEMA\_5\_DEMA\_10	mcrh\_DEMA\_5\_EMA\_10	mcrh\_DEMA\_5\_KAMA\_10	mcrh\_DEMA\_5\_SMA\_10	mcrh\_DEMA\_5\_T3\_10	mcrh\_DEMA\_5\_TEMA\_10	mcrh\_DEMA\_5\_TRIMA\_10	mcrh\_DEMA\_5\_WMA\_10	mcrh\_DEMA\_5\_DEMA\_20	mcrh\_DEMA\_5\_EMA\_20	mcrh\_DEMA\_5\_KAMA\_20	mcrh\_DEMA\_5\_SMA\_20	mcrh\_DEMA\_5\_T3\_20	mcrh\_DEMA\_5\_TEMA\_20	mcrh\_DEMA\_5\_TRIMA\_20	mcrh\_DEMA\_5\_WMA\_20	mcrh\_DEMA\_5\_DEMA\_50	mcrh\_DEMA\_5\_EMA\_50	mcrh\_DEMA\_5\_KAMA\_50	mcrh\_DEMA\_5\_SMA\_50	mcrh\_DEMA\_5\_T3\_50	mcrh\_DEMA\_5\_TEMA\_50	mcrh\_DEMA\_5\_TRIMA\_50	mcrh\_DEMA\_5\_WMA\_50	mcrh\_DEMA\_5\_DEMA\_100	mcrh\_DEMA\_5\_EMA\_100	mcrh\_DEMA\_5\_KAMA\_100	mcrh\_DEMA\_5\_SMA\_100	mcrh\_DEMA\_5\_T3\_100	mcrh\_DEMA\_5\_TEMA\_100	mcrh\_DEMA\_5\_TRIMA\_100	mcrh\_DEMA\_5\_WMA\_100	mcrh\_DEMA\_5\_ti\_h20	mcrh\_EMA\_5\_DEMA\_10	mcrh\_EMA\_5\_EMA\_10	mcrh\_EMA\_5\_KAMA\_10	mcrh\_EMA\_5\_SMA\_10	mcrh\_EMA\_5\_T3\_10	mcrh\_EMA\_5\_TEMA\_10	mcrh\_EMA\_5\_TRIMA\_10	mcrh\_EMA\_5\_WMA\_10	mcrh\_EMA\_5\_DEMA\_20	mcrh\_EMA\_5\_EMA\_20	mcrh\_EMA\_5\_KAMA\_20	mcrh\_EMA\_5\_SMA\_20	mcrh\_EMA\_5\_T3\_20	mcrh\_EMA\_5\_TEMA\_20	mcrh\_EMA\_5\_TRIMA\_20	mcrh\_EMA\_5\_WMA\_20	mcrh\_EMA\_5\_DEMA\_50	mcrh\_EMA\_5\_EMA\_50	mcrh\_EMA\_5\_KAMA\_50	mcrh\_EMA\_5\_SMA\_50	mcrh\_EMA\_5\_T3\_50	mcrh\_EMA\_5\_TEMA\_50	mcrh\_EMA\_5\_TRIMA\_50	mcrh\_EMA\_5\_WMA\_50	mcrh\_EMA\_5\_DEMA\_100	mcrh\_EMA\_5\_EMA\_100	mcrh\_EMA\_5\_KAMA\_100	mcrh\_EMA\_5\_SMA\_100	mcrh\_EMA\_5\_T3\_100	mcrh\_EMA\_5\_TEMA\_100	mcrh\_EMA\_5\_TRIMA\_100	mcrh\_EMA\_5\_WMA\_100	mcrh\_EMA\_5\_ti\_h20	mcrh\_KAMA\_5\_DEMA\_10	mcrh\_KAMA\_5\_EMA\_10	mcrh\_KAMA\_5\_KAMA\_10	mcrh\_KAMA\_5\_SMA\_10	mcrh\_KAMA\_5\_T3\_10	mcrh\_KAMA\_5\_TEMA\_10	mcrh\_KAMA\_5\_TRIMA\_10	mcrh\_KAMA\_5\_WMA\_10	mcrh\_KAMA\_5\_DEMA\_20	mcrh\_KAMA\_5\_EMA\_20	mcrh\_KAMA\_5\_KAMA\_20	mcrh\_KAMA\_5\_SMA\_20	mcrh\_KAMA\_5\_T3\_20	mcrh\_KAMA\_5\_TEMA\_20	mcrh\_KAMA\_5\_TRIMA\_20	mcrh\_KAMA\_5\_WMA\_20	mcrh\_KAMA\_5\_DEMA\_50	mcrh\_KAMA\_5\_EMA\_50	mcrh\_KAMA\_5\_KAMA\_50	mcrh\_KAMA\_5\_SMA\_50	mcrh\_KAMA\_5\_T3\_50	mcrh\_KAMA\_5\_TEMA\_50	mcrh\_KAMA\_5\_TRIMA\_50	mcrh\_KAMA\_5\_WMA\_50	mcrh\_KAMA\_5\_DEMA\_100	mcrh\_KAMA\_5\_EMA\_100	mcrh\_KAMA\_5\_KAMA\_100	mcrh\_KAMA\_5\_SMA\_100	mcrh\_KAMA\_5\_T3\_100	mcrh\_KAMA\_5\_TEMA\_100	mcrh\_KAMA\_5\_TRIMA\_100	mcrh\_KAMA\_5\_WMA\_100	mcrh\_KAMA\_5\_ti\_h20	mcrh\_SMA\_5\_DEMA\_10	mcrh\_SMA\_5\_EMA\_10	mcrh\_SMA\_5\_KAMA\_10	mcrh\_SMA\_5\_SMA\_10	mcrh\_SMA\_5\_T3\_10	mcrh\_SMA\_5\_TEMA\_10	mcrh\_SMA\_5\_TRIMA\_10	mcrh\_SMA\_5\_WMA\_10	mcrh\_SMA\_5\_DEMA\_20	mcrh\_SMA\_5\_EMA\_20	mcrh\_SMA\_5\_KAMA\_20	mcrh\_SMA\_5\_SMA\_20	mcrh\_SMA\_5\_T3\_20	mcrh\_SMA\_5\_TEMA\_20	mcrh\_SMA\_5\_TRIMA\_20	mcrh\_SMA\_5\_WMA\_20	mcrh\_SMA\_5\_DEMA\_50	mcrh\_SMA\_5\_EMA\_50	mcrh\_SMA\_5\_KAMA\_50	mcrh\_SMA\_5\_SMA\_50	mcrh\_SMA\_5\_T3\_50	mcrh\_SMA\_5\_TEMA\_50	mcrh\_SMA\_5\_TRIMA\_50	mcrh\_SMA\_5\_WMA\_50	mcrh\_SMA\_5\_DEMA\_100	mcrh\_SMA\_5\_EMA\_100	mcrh\_SMA\_5\_KAMA\_100	mcrh\_SMA\_5\_SMA\_100	mcrh\_SMA\_5\_T3\_100	mcrh\_SMA\_5\_TEMA\_100	mcrh\_SMA\_5\_TRIMA\_100	mcrh\_SMA\_5\_WMA\_100	mcrh\_SMA\_5\_ti\_h20	mcrh\_T3\_5\_DEMA\_10	mcrh\_T3\_5\_EMA\_10	mcrh\_T3\_5\_KAMA\_10	mcrh\_T3\_5\_SMA\_10	mcrh\_T3\_5\_T3\_10	mcrh\_T3\_5\_TEMA\_10	mcrh\_T3\_5\_TRIMA\_10	mcrh\_T3\_5\_WMA\_10	mcrh\_T3\_5\_DEMA\_20	mcrh\_T3\_5\_EMA\_20	mcrh\_T3\_5\_KAMA\_20	mcrh\_T3\_5\_SMA\_20	mcrh\_T3\_5\_T3\_20	mcrh\_T3\_5\_TEMA\_20	mcrh\_T3\_5\_TRIMA\_20	mcrh\_T3\_5\_WMA\_20	mcrh\_T3\_5\_DEMA\_50	mcrh\_T3\_5\_EMA\_50	mcrh\_T3\_5\_KAMA\_50	mcrh\_T3\_5\_SMA\_50	mcrh\_T3\_5\_T3\_50	mcrh\_T3\_5\_TEMA\_50	mcrh\_T3\_5\_TRIMA\_50	mcrh\_T3\_5\_WMA\_50	mcrh\_T3\_5\_DEMA\_100	mcrh\_T3\_5\_EMA\_100	mcrh\_T3\_5\_KAMA\_100	mcrh\_T3\_5\_SMA\_100	mcrh\_T3\_5\_T3\_100	mcrh\_T3\_5\_TEMA\_100	mcrh\_T3\_5\_TRIMA\_100	mcrh\_T3\_5\_WMA\_100	mcrh\_T3\_5\_ti\_h20	mcrh\_TEMA\_5\_DEMA\_10	mcrh\_TEMA\_5\_EMA\_10	mcrh\_TEMA\_5\_KAMA\_10	mcrh\_TEMA\_5\_SMA\_10	mcrh\_TEMA\_5\_T3\_10	mcrh\_TEMA\_5\_TEMA\_10	mcrh\_TEMA\_5\_TRIMA\_10	mcrh\_TEMA\_5\_WMA\_10	mcrh\_TEMA\_5\_DEMA\_20	mcrh\_TEMA\_5\_EMA\_20	mcrh\_TEMA\_5\_KAMA\_20	mcrh\_TEMA\_5\_SMA\_20	mcrh\_TEMA\_5\_T3\_20	mcrh\_TEMA\_5\_TEMA\_20	mcrh\_TEMA\_5\_TRIMA\_20	mcrh\_TEMA\_5\_WMA\_20	mcrh\_TEMA\_5\_DEMA\_50	mcrh\_TEMA\_5\_EMA\_50	mcrh\_TEMA\_5\_KAMA\_50	mcrh\_TEMA\_5\_SMA\_50	mcrh\_TEMA\_5\_T3\_50	mcrh\_TEMA\_5\_TEMA\_50	mcrh\_TEMA\_5\_TRIMA\_50	mcrh\_TEMA\_5\_WMA\_50	mcrh\_TEMA\_5\_DEMA\_100	mcrh\_TEMA\_5\_EMA\_100	mcrh\_TEMA\_5\_KAMA\_100	mcrh\_TEMA\_5\_SMA\_100	mcrh\_TEMA\_5\_T3\_100	mcrh\_TEMA\_5\_TEMA\_100	mcrh\_TEMA\_5\_TRIMA\_100	mcrh\_TEMA\_5\_WMA\_100	mcrh\_TEMA\_5\_ti\_h20	mcrh\_TRIMA\_5\_DEMA\_10	mcrh\_TRIMA\_5\_EMA\_10	mcrh\_TRIMA\_5\_KAMA\_10	mcrh\_TRIMA\_5\_SMA\_10	mcrh\_TRIMA\_5\_T3\_10	mcrh\_TRIMA\_5\_TEMA\_10	mcrh\_TRIMA\_5\_TRIMA\_10	mcrh\_TRIMA\_5\_WMA\_10	mcrh\_TRIMA\_5\_DEMA\_20	mcrh\_TRIMA\_5\_EMA\_20	mcrh\_TRIMA\_5\_KAMA\_20	mcrh\_TRIMA\_5\_SMA\_20	mcrh\_TRIMA\_5\_T3\_20	mcrh\_TRIMA\_5\_TEMA\_20	mcrh\_TRIMA\_5\_TRIMA\_20	mcrh\_TRIMA\_5\_WMA\_20	mcrh\_TRIMA\_5\_DEMA\_50	mcrh\_TRIMA\_5\_EMA\_50	mcrh\_TRIMA\_5\_KAMA\_50	mcrh\_TRIMA\_5\_SMA\_50	mcrh\_TRIMA\_5\_T3\_50	mcrh\_TRIMA\_5\_TEMA\_50	mcrh\_TRIMA\_5\_TRIMA\_50	mcrh\_TRIMA\_5\_WMA\_50	mcrh\_TRIMA\_5\_DEMA\_100	mcrh\_TRIMA\_5\_EMA\_100	mcrh\_TRIMA\_5\_KAMA\_100	mcrh\_TRIMA\_5\_SMA\_100	mcrh\_TRIMA\_5\_T3\_100	mcrh\_TRIMA\_5\_TEMA\_100	mcrh\_TRIMA\_5\_TRIMA\_100	mcrh\_TRIMA\_5\_WMA\_100	mcrh\_TRIMA\_5\_ti\_h20	mcrh\_WMA\_5\_DEMA\_10	mcrh\_WMA\_5\_EMA\_10	mcrh\_WMA\_5\_KAMA\_10	mcrh\_WMA\_5\_SMA\_10	mcrh\_WMA\_5\_T3\_10	mcrh\_WMA\_5\_TEMA\_10	mcrh\_WMA\_5\_TRIMA\_10	mcrh\_WMA\_5\_WMA\_10	mcrh\_WMA\_5\_DEMA\_20	mcrh\_WMA\_5\_EMA\_20	mcrh\_WMA\_5\_KAMA\_20	mcrh\_WMA\_5\_SMA\_20	mcrh\_WMA\_5\_T3\_20	mcrh\_WMA\_5\_TEMA\_20	mcrh\_WMA\_5\_TRIMA\_20	mcrh\_WMA\_5\_WMA\_20	mcrh\_WMA\_5\_DEMA\_50	mcrh\_WMA\_5\_EMA\_50	mcrh\_WMA\_5\_KAMA\_50	mcrh\_WMA\_5\_SMA\_50	mcrh\_WMA\_5\_T3\_50	mcrh\_WMA\_5\_TEMA\_50	mcrh\_WMA\_5\_TRIMA\_50	mcrh\_WMA\_5\_WMA\_50	mcrh\_WMA\_5\_DEMA\_100	mcrh\_WMA\_5\_EMA\_100	mcrh\_WMA\_5\_KAMA\_100	mcrh\_WMA\_5\_SMA\_100	mcrh\_WMA\_5\_T3\_100	mcrh\_WMA\_5\_TEMA\_100	mcrh\_WMA\_5\_TRIMA\_100	mcrh\_WMA\_5\_WMA\_100	mcrh\_WMA\_5\_ti\_h20	mcrh\_DEMA\_10\_DEMA\_20	mcrh\_DEMA\_10\_EMA\_20	mcrh\_DEMA\_10\_KAMA\_20	mcrh\_DEMA\_10\_SMA\_20	mcrh\_DEMA\_10\_T3\_20	mcrh\_DEMA\_10\_TEMA\_20	mcrh\_DEMA\_10\_TRIMA\_20	mcrh\_DEMA\_10\_WMA\_20	mcrh\_DEMA\_10\_DEMA\_50	mcrh\_DEMA\_10\_EMA\_50	mcrh\_DEMA\_10\_KAMA\_50	mcrh\_DEMA\_10\_SMA\_50	mcrh\_DEMA\_10\_T3\_50	mcrh\_DEMA\_10\_TEMA\_50	mcrh\_DEMA\_10\_TRIMA\_50	mcrh\_DEMA\_10\_WMA\_50	mcrh\_DEMA\_10\_DEMA\_100	mcrh\_DEMA\_10\_EMA\_100	mcrh\_DEMA\_10\_KAMA\_100	mcrh\_DEMA\_10\_SMA\_100	mcrh\_DEMA\_10\_T3\_100	mcrh\_DEMA\_10\_TEMA\_100	mcrh\_DEMA\_10\_TRIMA\_100	mcrh\_DEMA\_10\_WMA\_100	mcrh\_DEMA\_10\_ti\_h20	mcrh\_EMA\_10\_DEMA\_20	mcrh\_EMA\_10\_EMA\_20	mcrh\_EMA\_10\_KAMA\_20	mcrh\_EMA\_10\_SMA\_20	mcrh\_EMA\_10\_T3\_20	mcrh\_EMA\_10\_TEMA\_20	mcrh\_EMA\_10\_TRIMA\_20	mcrh\_EMA\_10\_WMA\_20	mcrh\_EMA\_10\_DEMA\_50	mcrh\_EMA\_10\_EMA\_50	mcrh\_EMA\_10\_KAMA\_50	mcrh\_EMA\_10\_SMA\_50	mcrh\_EMA\_10\_T3\_50	mcrh\_EMA\_10\_TEMA\_50	mcrh\_EMA\_10\_TRIMA\_50	mcrh\_EMA\_10\_WMA\_50	mcrh\_EMA\_10\_DEMA\_100	mcrh\_EMA\_10\_EMA\_100	mcrh\_EMA\_10\_KAMA\_100	mcrh\_EMA\_10\_SMA\_100	mcrh\_EMA\_10\_T3\_100	mcrh\_EMA\_10\_TEMA\_100	mcrh\_EMA\_10\_TRIMA\_100	mcrh\_EMA\_10\_WMA\_100	mcrh\_EMA\_10\_ti\_h20	mcrh\_KAMA\_10\_DEMA\_20	mcrh\_KAMA\_10\_EMA\_20	mcrh\_KAMA\_10\_KAMA\_20	mcrh\_KAMA\_10\_SMA\_20	mcrh\_KAMA\_10\_T3\_20	mcrh\_KAMA\_10\_TEMA\_20	mcrh\_KAMA\_10\_TRIMA\_20	mcrh\_KAMA\_10\_WMA\_20	mcrh\_KAMA\_10\_DEMA\_50	mcrh\_KAMA\_10\_EMA\_50	mcrh\_KAMA\_10\_KAMA\_50	mcrh\_KAMA\_10\_SMA\_50	mcrh\_KAMA\_10\_T3\_50	mcrh\_KAMA\_10\_TEMA\_50	mcrh\_KAMA\_10\_TRIMA\_50	mcrh\_KAMA\_10\_WMA\_50	mcrh\_KAMA\_10\_DEMA\_100	mcrh\_KAMA\_10\_EMA\_100	mcrh\_KAMA\_10\_KAMA\_100	mcrh\_KAMA\_10\_SMA\_100	mcrh\_KAMA\_10\_T3\_100	mcrh\_KAMA\_10\_TEMA\_100	mcrh\_KAMA\_10\_TRIMA\_100	mcrh\_KAMA\_10\_WMA\_100	mcrh\_KAMA\_10\_ti\_h20	mcrh\_SMA\_10\_DEMA\_20	mcrh\_SMA\_10\_EMA\_20	mcrh\_SMA\_10\_KAMA\_20	mcrh\_SMA\_10\_SMA\_20	mcrh\_SMA\_10\_T3\_20	mcrh\_SMA\_10\_TEMA\_20	mcrh\_SMA\_10\_TRIMA\_20	mcrh\_SMA\_10\_WMA\_20	mcrh\_SMA\_10\_DEMA\_50	mcrh\_SMA\_10\_EMA\_50	mcrh\_SMA\_10\_KAMA\_50	mcrh\_SMA\_10\_SMA\_50	mcrh\_SMA\_10\_T3\_50	mcrh\_SMA\_10\_TEMA\_50	mcrh\_SMA\_10\_TRIMA\_50	mcrh\_SMA\_10\_WMA\_50	mcrh\_SMA\_10\_DEMA\_100	mcrh\_SMA\_10\_EMA\_100	mcrh\_SMA\_10\_KAMA\_100	mcrh\_SMA\_10\_SMA\_100	mcrh\_SMA\_10\_T3\_100	mcrh\_SMA\_10\_TEMA\_100	mcrh\_SMA\_10\_TRIMA\_100	mcrh\_SMA\_10\_WMA\_100	mcrh\_SMA\_10\_ti\_h20	mcrh\_T3\_10\_DEMA\_20	mcrh\_T3\_10\_EMA\_20	mcrh\_T3\_10\_KAMA\_20	mcrh\_T3\_10\_SMA\_20	mcrh\_T3\_10\_T3\_20	mcrh\_T3\_10\_TEMA\_20	mcrh\_T3\_10\_TRIMA\_20	mcrh\_T3\_10\_WMA\_20	mcrh\_T3\_10\_DEMA\_50	mcrh\_T3\_10\_EMA\_50	mcrh\_T3\_10\_KAMA\_50	mcrh\_T3\_10\_SMA\_50	mcrh\_T3\_10\_T3\_50	mcrh\_T3\_10\_TEMA\_50	mcrh\_T3\_10\_TRIMA\_50	mcrh\_T3\_10\_WMA\_50	mcrh\_T3\_10\_DEMA\_100	mcrh\_T3\_10\_EMA\_100	mcrh\_T3\_10\_KAMA\_100	mcrh\_T3\_10\_SMA\_100	mcrh\_T3\_10\_T3\_100	mcrh\_T3\_10\_TEMA\_100	mcrh\_T3\_10\_TRIMA\_100	mcrh\_T3\_10\_WMA\_100	mcrh\_T3\_10\_ti\_h20	mcrh\_TEMA\_10\_DEMA\_20	mcrh\_TEMA\_10\_EMA\_20	mcrh\_TEMA\_10\_KAMA\_20	mcrh\_TEMA\_10\_SMA\_20	mcrh\_TEMA\_10\_T3\_20	mcrh\_TEMA\_10\_TEMA\_20	mcrh\_TEMA\_10\_TRIMA\_20	mcrh\_TEMA\_10\_WMA\_20	mcrh\_TEMA\_10\_DEMA\_50	mcrh\_TEMA\_10\_EMA\_50	mcrh\_TEMA\_10\_KAMA\_50	mcrh\_TEMA\_10\_SMA\_50	mcrh\_TEMA\_10\_T3\_50	mcrh\_TEMA\_10\_TEMA\_50	mcrh\_TEMA\_10\_TRIMA\_50	mcrh\_TEMA\_10\_WMA\_50	mcrh\_TEMA\_10\_DEMA\_100	mcrh\_TEMA\_10\_EMA\_100	mcrh\_TEMA\_10\_KAMA\_100	mcrh\_TEMA\_10\_SMA\_100	mcrh\_TEMA\_10\_T3\_100	mcrh\_TEMA\_10\_TEMA\_100	mcrh\_TEMA\_10\_TRIMA\_100	mcrh\_TEMA\_10\_WMA\_100	mcrh\_TEMA\_10\_ti\_h20	mcrh\_TRIMA\_10\_DEMA\_20	mcrh\_TRIMA\_10\_EMA\_20	mcrh\_TRIMA\_10\_KAMA\_20	mcrh\_TRIMA\_10\_SMA\_20	mcrh\_TRIMA\_10\_T3\_20	mcrh\_TRIMA\_10\_TEMA\_20	mcrh\_TRIMA\_10\_TRIMA\_20	mcrh\_TRIMA\_10\_WMA\_20	mcrh\_TRIMA\_10\_DEMA\_50	mcrh\_TRIMA\_10\_EMA\_50	mcrh\_TRIMA\_10\_KAMA\_50	mcrh\_TRIMA\_10\_SMA\_50	mcrh\_TRIMA\_10\_T3\_50	mcrh\_TRIMA\_10\_TEMA\_50	mcrh\_TRIMA\_10\_TRIMA\_50	mcrh\_TRIMA\_10\_WMA\_50	mcrh\_TRIMA\_10\_DEMA\_100	mcrh\_TRIMA\_10\_EMA\_100	mcrh\_TRIMA\_10\_KAMA\_100	mcrh\_TRIMA\_10\_SMA\_100	mcrh\_TRIMA\_10\_T3\_100	mcrh\_TRIMA\_10\_TEMA\_100	mcrh\_TRIMA\_10\_TRIMA\_100	mcrh\_TRIMA\_10\_WMA\_100	mcrh\_TRIMA\_10\_ti\_h20	mcrh\_WMA\_10\_DEMA\_20	mcrh\_WMA\_10\_EMA\_20	mcrh\_WMA\_10\_KAMA\_20	mcrh\_WMA\_10\_SMA\_20	mcrh\_WMA\_10\_T3\_20	mcrh\_WMA\_10\_TEMA\_20	mcrh\_WMA\_10\_TRIMA\_20	mcrh\_WMA\_10\_WMA\_20	mcrh\_WMA\_10\_DEMA\_50	mcrh\_WMA\_10\_EMA\_50	mcrh\_WMA\_10\_KAMA\_50	mcrh\_WMA\_10\_SMA\_50	mcrh\_WMA\_10\_T3\_50	mcrh\_WMA\_10\_TEMA\_50	mcrh\_WMA\_10\_TRIMA\_50	mcrh\_WMA\_10\_WMA\_50	mcrh\_WMA\_10\_DEMA\_100	mcrh\_WMA\_10\_EMA\_100	mcrh\_WMA\_10\_KAMA\_100	mcrh\_WMA\_10\_SMA\_100	mcrh\_WMA\_10\_T3\_100	mcrh\_WMA\_10\_TEMA\_100	mcrh\_WMA\_10\_TRIMA\_100	mcrh\_WMA\_10\_WMA\_100	mcrh\_WMA\_10\_ti\_h20	mcrh\_DEMA\_20\_DEMA\_50	mcrh\_DEMA\_20\_EMA\_50	mcrh\_DEMA\_20\_KAMA\_50	mcrh\_DEMA\_20\_SMA\_50	mcrh\_DEMA\_20\_T3\_50	mcrh\_DEMA\_20\_TEMA\_50	mcrh\_DEMA\_20\_TRIMA\_50	mcrh\_DEMA\_20\_WMA\_50	mcrh\_DEMA\_20\_DEMA\_100	mcrh\_DEMA\_20\_EMA\_100	mcrh\_DEMA\_20\_KAMA\_100	mcrh\_DEMA\_20\_SMA\_100	mcrh\_DEMA\_20\_T3\_100	mcrh\_DEMA\_20\_TEMA\_100	mcrh\_DEMA\_20\_TRIMA\_100	mcrh\_DEMA\_20\_WMA\_100	mcrh\_EMA\_20\_DEMA\_50	mcrh\_EMA\_20\_EMA\_50	mcrh\_EMA\_20\_KAMA\_50	mcrh\_EMA\_20\_SMA\_50	mcrh\_EMA\_20\_T3\_50	mcrh\_EMA\_20\_TEMA\_50	mcrh\_EMA\_20\_TRIMA\_50	mcrh\_EMA\_20\_WMA\_50	mcrh\_EMA\_20\_DEMA\_100	mcrh\_EMA\_20\_EMA\_100	mcrh\_EMA\_20\_KAMA\_100	mcrh\_EMA\_20\_SMA\_100	mcrh\_EMA\_20\_T3\_100	mcrh\_EMA\_20\_TEMA\_100	mcrh\_EMA\_20\_TRIMA\_100	mcrh\_EMA\_20\_WMA\_100	mcrh\_KAMA\_20\_DEMA\_50	mcrh\_KAMA\_20\_EMA\_50	mcrh\_KAMA\_20\_KAMA\_50	mcrh\_KAMA\_20\_SMA\_50	mcrh\_KAMA\_20\_T3\_50	mcrh\_KAMA\_20\_TEMA\_50	mcrh\_KAMA\_20\_TRIMA\_50	mcrh\_KAMA\_20\_WMA\_50	mcrh\_KAMA\_20\_DEMA\_100	mcrh\_KAMA\_20\_EMA\_100	mcrh\_KAMA\_20\_KAMA\_100	mcrh\_KAMA\_20\_SMA\_100	mcrh\_KAMA\_20\_T3\_100	mcrh\_KAMA\_20\_TEMA\_100	mcrh\_KAMA\_20\_TRIMA\_100	mcrh\_KAMA\_20\_WMA\_100	mcrh\_SMA\_20\_DEMA\_50	mcrh\_SMA\_20\_EMA\_50	mcrh\_SMA\_20\_KAMA\_50	mcrh\_SMA\_20\_SMA\_50	mcrh\_SMA\_20\_T3\_50	mcrh\_SMA\_20\_TEMA\_50	mcrh\_SMA\_20\_TRIMA\_50	mcrh\_SMA\_20\_WMA\_50	mcrh\_SMA\_20\_DEMA\_100	mcrh\_SMA\_20\_EMA\_100	mcrh\_SMA\_20\_KAMA\_100	mcrh\_SMA\_20\_SMA\_100	mcrh\_SMA\_20\_T3\_100	mcrh\_SMA\_20\_TEMA\_100	mcrh\_SMA\_20\_TRIMA\_100	mcrh\_SMA\_20\_WMA\_100	mcrh\_T3\_20\_DEMA\_50	mcrh\_T3\_20\_EMA\_50	mcrh\_T3\_20\_KAMA\_50	mcrh\_T3\_20\_SMA\_50	mcrh\_T3\_20\_T3\_50	mcrh\_T3\_20\_TEMA\_50	mcrh\_T3\_20\_TRIMA\_50	mcrh\_T3\_20\_WMA\_50	mcrh\_T3\_20\_DEMA\_100	mcrh\_T3\_20\_EMA\_100	mcrh\_T3\_20\_KAMA\_100	mcrh\_T3\_20\_SMA\_100	mcrh\_T3\_20\_T3\_100	mcrh\_T3\_20\_TEMA\_100	mcrh\_T3\_20\_TRIMA\_100	mcrh\_T3\_20\_WMA\_100	mcrh\_TEMA\_20\_DEMA\_50	mcrh\_TEMA\_20\_EMA\_50	mcrh\_TEMA\_20\_KAMA\_50	mcrh\_TEMA\_20\_SMA\_50	mcrh\_TEMA\_20\_T3\_50	mcrh\_TEMA\_20\_TEMA\_50	mcrh\_TEMA\_20\_TRIMA\_50	mcrh\_TEMA\_20\_WMA\_50	mcrh\_TEMA\_20\_DEMA\_100	mcrh\_TEMA\_20\_EMA\_100	mcrh\_TEMA\_20\_KAMA\_100	mcrh\_TEMA\_20\_SMA\_100	mcrh\_TEMA\_20\_T3\_100	mcrh\_TEMA\_20\_TEMA\_100	mcrh\_TEMA\_20\_TRIMA\_100	mcrh\_TEMA\_20\_WMA\_100	mcrh\_TRIMA\_20\_DEMA\_50	mcrh\_TRIMA\_20\_EMA\_50	mcrh\_TRIMA\_20\_KAMA\_50	mcrh\_TRIMA\_20\_SMA\_50	mcrh\_TRIMA\_20\_T3\_50	mcrh\_TRIMA\_20\_TEMA\_50	mcrh\_TRIMA\_20\_TRIMA\_50	mcrh\_TRIMA\_20\_WMA\_50	mcrh\_TRIMA\_20\_DEMA\_100	mcrh\_TRIMA\_20\_EMA\_100	mcrh\_TRIMA\_20\_KAMA\_100	mcrh\_TRIMA\_20\_SMA\_100	mcrh\_TRIMA\_20\_T3\_100	mcrh\_TRIMA\_20\_TEMA\_100	mcrh\_TRIMA\_20\_TRIMA\_100	mcrh\_TRIMA\_20\_WMA\_100	mcrh\_WMA\_20\_DEMA\_50	mcrh\_WMA\_20\_EMA\_50	mcrh\_WMA\_20\_KAMA\_50	mcrh\_WMA\_20\_SMA\_50	mcrh\_WMA\_20\_T3\_50	mcrh\_WMA\_20\_TEMA\_50	mcrh\_WMA\_20\_TRIMA\_50	mcrh\_WMA\_20\_WMA\_50	mcrh\_WMA\_20\_DEMA\_100	mcrh\_WMA\_20\_EMA\_100	mcrh\_WMA\_20\_KAMA\_100	mcrh\_WMA\_20\_SMA\_100	mcrh\_WMA\_20\_T3\_100	mcrh\_WMA\_20\_TEMA\_100	mcrh\_WMA\_20\_TRIMA\_100	mcrh\_WMA\_20\_WMA\_100	mcrh\_DEMA\_50\_DEMA\_100	mcrh\_DEMA\_50\_EMA\_100	mcrh\_DEMA\_50\_KAMA\_100	mcrh\_DEMA\_50\_SMA\_100	mcrh\_DEMA\_50\_T3\_100	mcrh\_DEMA\_50\_TEMA\_100	mcrh\_DEMA\_50\_TRIMA\_100	mcrh\_DEMA\_50\_WMA\_100	mcrh\_DEMA\_50\_ti\_h20	mcrh\_EMA\_50\_DEMA\_100	mcrh\_EMA\_50\_EMA\_100	mcrh\_EMA\_50\_KAMA\_100	mcrh\_EMA\_50\_SMA\_100	mcrh\_EMA\_50\_T3\_100	mcrh\_EMA\_50\_TEMA\_100	mcrh\_EMA\_50\_TRIMA\_100	mcrh\_EMA\_50\_WMA\_100	mcrh\_EMA\_50\_ti\_h20	mcrh\_KAMA\_50\_DEMA\_100	mcrh\_KAMA\_50\_EMA\_100	mcrh\_KAMA\_50\_KAMA\_100	mcrh\_KAMA\_50\_SMA\_100	mcrh\_KAMA\_50\_T3\_100	mcrh\_KAMA\_50\_TEMA\_100	mcrh\_KAMA\_50\_TRIMA\_100	mcrh\_KAMA\_50\_WMA\_100	mcrh\_KAMA\_50\_ti\_h20	mcrh\_SMA\_50\_DEMA\_100	mcrh\_SMA\_50\_EMA\_100	mcrh\_SMA\_50\_KAMA\_100	mcrh\_SMA\_50\_SMA\_100	mcrh\_SMA\_50\_T3\_100	mcrh\_SMA\_50\_TEMA\_100	mcrh\_SMA\_50\_TRIMA\_100	mcrh\_SMA\_50\_WMA\_100	mcrh\_SMA\_50\_ti\_h20	mcrh\_T3\_50\_DEMA\_100	mcrh\_T3\_50\_EMA\_100	mcrh\_T3\_50\_KAMA\_100	mcrh\_T3\_50\_SMA\_100	mcrh\_T3\_50\_T3\_100	mcrh\_T3\_50\_TEMA\_100	mcrh\_T3\_50\_TRIMA\_100	mcrh\_T3\_50\_WMA\_100	mcrh\_T3\_50\_ti\_h20	mcrh\_TEMA\_50\_DEMA\_100	mcrh\_TEMA\_50\_EMA\_100	mcrh\_TEMA\_50\_KAMA\_100	mcrh\_TEMA\_50\_SMA\_100	mcrh\_TEMA\_50\_T3\_100	mcrh\_TEMA\_50\_TEMA\_100	mcrh\_TEMA\_50\_TRIMA\_100	mcrh\_TEMA\_50\_WMA\_100	mcrh\_TEMA\_50\_ti\_h20	mcrh\_TRIMA\_50\_DEMA\_100	mcrh\_TRIMA\_50\_EMA\_100	mcrh\_TRIMA\_50\_KAMA\_100	mcrh\_TRIMA\_50\_SMA\_100	mcrh\_TRIMA\_50\_T3\_100	mcrh\_TRIMA\_50\_TEMA\_100	mcrh\_TRIMA\_50\_TRIMA\_100	mcrh\_TRIMA\_50\_WMA\_100	mcrh\_TRIMA\_50\_ti\_h20	mcrh\_WMA\_50\_DEMA\_100	mcrh\_WMA\_50\_EMA\_100	mcrh\_WMA\_50\_KAMA\_100	mcrh\_WMA\_50\_SMA\_100	mcrh\_WMA\_50\_T3\_100	mcrh\_WMA\_50\_TEMA\_100	mcrh\_WMA\_50\_TRIMA\_100	mcrh\_WMA\_50\_WMA\_100	mcrh\_WMA\_50\_ti\_h20	mcrh\_DEMA\_100\_ti\_h20	mcrh\_EMA\_100\_ti\_h20	mcrh\_KAMA\_100\_ti\_h20	mcrh\_SMA\_100\_ti\_h20	mcrh\_T3\_100\_ti\_h20	mcrh\_TEMA\_100\_ti\_h20	mcrh\_TRIMA\_100\_ti\_h20	mcrh\_WMA\_100\_ti\_h20	NQ\_Close	NQ\_Volume	NQ\_per\_Close	NQ\_per\_Volume	NQ\_SMA\_20	NQ\_SMA\_100**

