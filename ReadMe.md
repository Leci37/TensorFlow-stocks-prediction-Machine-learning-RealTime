OBJETIVO
Entendiendo el principio de profecía autocumplida, se puede obtener el patrón de las mismas, mediante la recogida masiva de patrones técnicos, su cálculo y el estudio de sus patrones

Para ello , se emplearán técnica como el big data mediante librerías Pandas Python, el machine learning mediante Sklearn, XGB  y redes neuronales mediante la librería abierta de google Tensor Flow 

El resultado se mostrará de forma sencilla y amigable mediante alertas en móvil o ordenador.
Ejemplo de alerta en tiempo real a través del bot de telegram https://t.me/Whale_Hunter_Alertbot 

Los modelos machine learnig Sklearn, XGB y Tensor Flow , mediante el aprendizaje de los últimos meses detectan el punto de venta. Para detectar este punto de venta se han tenido en cuenta una serie de indicadores: olap_VMAP, ma_SMA_50, ichi_senkou_a, olap_BBAND_dif ,mtum_MACD_ext, olap_BBAND_MIDDLE, mtum_MACD_ext_signal, fibo_s1, volu_PVI_1, ma_KAMA_5, etcétera.

En la imagen se muestran: MACD, RSI , estocástico y Balance of power (Elder Ray) 
La alerta es mandada en la línea vertical, durante los 4 siguientes periodos la acción decrece un 2,4%. Cada periodo-vela de la imagen indica 15 minutos.





FUNCIONAMIENTO

#1 Recogida de datos
Recoger datos para entrenar el modelo
yhoo_generate_big_all_csv.py
Se obtienen los datos de cierre , mediante yahoo API finance , y se calculan centenares de patrones técnicos mediante las librerías pandas_ta y talib 
yhoo_history_stock.get_SCALA_csv_stocks_history_Download_list()

El modelo para ser capaz de entrenarse en detectar puntos de compra y de venta, crea la columna buy_seel_point tiene valor de: 0, -100, 100. Estos son detectados en función de los máximos cambios, (positivos 100, negativo -100) en el histórico de los últimos meses, este punto será con los que se entrene el entrenamiento, también llamado el ground true 
Se asignará valor en buy_seel_point si el incremento o decremento de la acción es mayor del 2,5% en un periodo de 3 horas, mediante la funcion get_buy_sell_points_Roll

Una vez obtenidos los datos históricos de la acción y calculados todos los indicadores técnicos, un total de 1068, se generan ficheros de tipo AAPL_stock_history_MONTH_3_AD.csv
Ejemplo del fichero con los ocho primeros indicadores:


Esta obtención de datos es personalizable, se puede obtener y entrenar modelos de cualquier acción del Nasdaq, para otros indicadores o cripto-activos , también es posible mediante pequeños cambios.

A través de la clase Option_Historical existe la posibilidad de crear ficheros de datos históricos: anuales, mensuales y  diarios.
class Option_Historical(Enum): YEARS_3 = 1, MONTH_3 = 2, MONTH_3_AD = 3, DAY_6 = 4, DAY_1 = 5

Se generan los ficheros \d_price\min_max\AAPL_min_max_stock_MONTH_3.csv , los cuales guardan el max y min valor de cada columna , para que sea leido en Model_predictions_Nrows.py para un  rapido  fit_scaler() (es el proceso “limpieza” que requieren los datos antes de entrar en los modelos de entrenamiento AI)  . Esta operación es de vital importancia para una correcta optimización en la lectura de datos en tiempo real.




#1.1 Tipos de indicadores
Durante la generación de fichero de recogida de datos del punto 1 AAPL_stock_history_MONTH_3_AD.csv se calculan 1068 indicadores técnicos, los cuales se dividen en subtipos, en función de prefijos en el nombre.
Lista de prefijos y ejemplo de nombre de alguno de ellos.

Overlap o superposición:  olap_
olap_BBAND_UPPER, olap_BBAND_MIDDLE, olap_BBAND_LOWER, 
Momentum:  mtum_
mtum_MACD, mtum_MACD_signal, mtum_RSI, mtum_STOCH_k,
Volatilidad: vola_
vola_KCBe_20_2, vola_KCUe_20_2, vola_RVI_14
Patrones de ciclo: cycl_
cycl_DCPHASE, cycl_PHASOR_inph, cycl_PHASOR_quad
Patrones de velas: cdl_
cdl_RICKSHAWMAN, cdl_RISEFALL3METHODS, cdl_SEPARATINGLINES
Estadística: sti_
sti_STDDEV, sti_TSF, sti_VAR
Medias móviles: ma_
ma_SMA_100, ma_WMA_10, ma_DEMA_20, ma_EMA_100, ma_KAMA_10, 
Tendencia: tend_ y ti_
tend_renko_TR, tend_renko_brick, ti_acc_dist, ti_chaikin_10_3
Resistencias y soportes sufijos:  _s3, _s2, _s1, _pp, _r1, _r2, _r3
fibo_s3, fibo_s2, fibo_s1, fibo_pp, fibo_r1, fibo_r2, fibo_r3
demark_s1, demark_pp, demark_r1
Punto de intersección con resistencia o soporte: pcrh_
pcrh_demark_s1, pcrh_demark_pp, pcrh_demark_r1
Punto de intersección con media móvil o de medias móviles entre ellas: mcrh_
mcrh_SMA_20_TRIMA_50, mcrh_SMA_20_WMA_50, mcrh_SMA_20_DEMA_100
Indicadores de cambios en el índice bursátil , nasdaq: NQ_
NQ_SMA_20, NQ_SMA_100

Nota: Para ver los 1068 indicadores usados ir a las hojas adjuntas al final del documento.


#2 Filtrado de indicadores
 ejecutar para saber qué columnas son relevantes para la obtención del modelo
Feature_selection_create_json.py
Hay que saber cuáles de las centenares columnas de datos técnicos , es válida para entrenar el modelo neuronal, y cuales son solo ruido. Esto se hará mediante correlaciones y modelos de Random Forest.
Responde a la pregunta:
¿qué columnas son las más relevantes para puntos de compra o venta?

Genera los ficheros best_selection, los cuales son un raking de mejores datos técnicos para entrenar el modelo , se pretende pasar de 1068 columnas a unas 120
Por ejemplo, para la acción Amazon, detección puntos de compra,  en el periodo Junio a Octubre 2022, los indicadores más valiosos son:
Senkuo de la nube de Ichimoku
Volatilidad de Chaikin 
On-balance volume

Ejemplo del fichero plots_relations/best_selection_AMNZ_pos.json
"index": {
  "12": [
     "ichi_senkou_b"
  ],
  "10": [
     "volu_Chaikin_AD"
  ],
  "9": [
     "volu_OBV"
  ],



#3 entrenar los modelos TF, XGB y Sklearn 
Model_creation_models_for_a_stock.py
para ello se requiere la selección de mejores columnas del punto #2
Hay cuatro tipos de algoritmos predictivos, modelos AI:
Gradient Boosting está formado por un conjunto de árboles de decisión individuales, entrenados de forma secuencial, de forma que cada nuevo árbol trata de mejorar los errores de los árboles anteriores. Librería Sklearn
Random Forest Los bosques aleatorios  son un método de aprendizaje conjunto para clasificación, regresión y otras tareas que opera mediante la construcción de una multitud de árboles de decisión en el momento del entrenamiento. Librería Sklearn 
XGBoost es una biblioteca de aumento de gradiente distribuida optimizada diseñada para ser altamente eficiente, flexible y portátil. Implementa algoritmos de aprendizaje automático bajo el marco Gradient Boosting. Librería XGBoost 
TensorFlow TF es una biblioteca de código abierto para aprendizaje automático a través de un rango de tareas, y desarrollado por Google para satisfacer sus necesidades de sistemas capaces de construir y entrenar redes neuronales para detectar y descifrar patrones y correlaciones, análogos al aprendizaje y razonamiento usados por los humanos. Librería TensorFlow 


Existen modelos POS (compra) o modelo NEG (venta) y existe un modelo BOTH (BOTH se descarta,  ya que los modelos de predicción son binario , sólo aceptan 2 posiciones, verdad o mentira)
Este punto genera modelos de predicción .sav para XGB y Sklearn. .h5  para Tensor Flow
Ejemplos de nombrado: XGboost_U_neg_vgood16_.sav y TF_AMZN_neg_vgood16_64.h5

Formato de los nombres:
Tipo de AI con las que se entrena puede ser:
 XGboost, TF, TF64, GradientBoost y RandomForest
Ticker de la acción AMZN para amazon , AAPL para Apple …
Detecta puntos de compra o de venta pos o neg
Cuantos indicadores han sido usados en el aprendizaje, pueden ser de 4 tipos en función de la relevancia dada por el punto #2 Filtrado de indicadores. Este ranking es organizado en la clase MODEL_TYPE_COLM, 
vgood16 los mejores 16 indicadores
good9 los mejores 32 indicadores
reg4 los mejores 64 indicadores 
low1 los mejores 128 indicadores 

Estas combinaciones implican que por cada acción se crean 5 tipos de AI, cada una en pos y neg, además por cada combinación se añade las 4 configuraciones de indicadores técnicos.  Esto genera 40 modelos de AI, los cuales serán seleccionados en el punto: #4 evaluar la CALIDAD de esos modelos

Cada vez que se genera un modelo IA, se genera un fichero de registro: TF_balance\TF_AAPL_pos_reg4.h5_accuracy_87.6%__loss_2.74__epochs_10[160].csv
Este contiene los datos de precisión y de pérdida del modelo, así como los registros del entrenamiento del mismo.




#4 evaluar la CALIDAD de esos modelos 
Model_creation_scoring.py
Para hacer una predicción con los AI, se recogen nuevos datos y se calculan los indicadores técnicos con los que ha sido entrenado según los ficheros best_selection.
Cuando los modelos .h5 y .sav son preguntados:
 ¿Esto es un punto de compra-venta? 
Estos responden un número que puede variar entre 0,1 y 4 
Cuanto más alto sea el número mayor probabilidad de que sea un punto de compra-venta correcto.

Cada modelo tiene una escala de puntuación en el cual se considera punto de compra venta. para unos modelos con una puntuación de más 0,4 será suficiente (normalmente los XGboost) , mientras que para otros requieren más de 1,5  (normalmente los TF).

¿Cómo se sabe cuál es la puntuación umbral para cada modelo ?
La clase Model_creation_scoring.py  genera los ficheros umbral de puntuación threshold, los cuales dicen cual es el punto umbral en el cual se considera punto de compra-venta.
Cada modelo AI contará con su propio fichero de tipo:
Models/Scoring/AAPL_neg__when_model_ok_threshold.csv

Por cada acción el punto #3 entrenar los modelos TF, XGB y Sklearn  se generan 40 modelos de AI. Esta clase evalúa y selecciona los modelos más precisos de tal forma que solo se ejecutarán en tiempo real los más precisos (normalmente se seleccionan entre 4 y 8)
Models/Scoring/AAPL_neg__groupby_buy_sell_point_000.json
"list_good_params": [
  "r_rf_AFRM_pos_low1_",
  "r_TF64_AFRM_pos_vgood16_",
  "r_TF64_AFRM_pos_good9_",
  "r_TF_AFRM_pos_reg4_"
],


#4.1 evaluar esos BENEFICIO real de modelos
Model_predictions_N_eval_profits.py
Responde a la pregunta: 
¿Si se deja N días ejecutándose, cuánto dinero hipotético se gana ?
Nota: esto debe ejecutarse con datos que no hayan sido usados en el modelo de entrenamiento, preferentemente
Models/eval_Profits/_AAPL_neg_ALL_stock_20221021__20221014.csv


#5 hacer predicciones en tiempo real
Model_predictions_Nrows.py
Se puede hacer predicciones con los datos en tiempo real de la acción.
A través de la llamada a la función cada 10-12min, descarga los datos de la acción en tiempo real a través de la API financiera de yahoo.
df_compar, df_vender = get_RealTime_buy_seel_points()


#5.1 mandar alertas en tiempo real
ztelegram_send_message.py liniar monohilo 32s por accion  (obsoleta)
yhoo_POOL_enque_Thread.py multihilo encolado 2s por accion 

Existe la posibilidad de mandar las alertas de compra venta de la acción, al telegram o mail
se evalúan los múltiples modelos entrenados AI , y solo los mayores de 96% de probabilidad (según lo entrenado previamente) son notificados
Cada 15 minutos , se calculan todos los indicadores en real time por cada acción y se evalúan en los modelos AI
En la alerta se indica qué modelos están detectando puntos de compra y venta, correctos en los que ejecutar la transacción 
Estas alertas de compra-venta caducan en, más menos 7 minutos, dado la volatilidad del mercado
También se adjunta el precio al que se detectó, la hora, y los enlaces a las web de noticias.
Nota: las noticias financieras siempre deben prevalecer a los indicadores técnicos. 

Enlace del Bot telegram : https://t.me/Whale_Hunter_Alertbot 
