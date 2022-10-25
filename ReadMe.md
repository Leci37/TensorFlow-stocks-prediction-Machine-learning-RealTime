#Objetivo
Entendiendo el principio de profecía autocumplida, se puede obtener el patrón de las mismas, mediante la recogida masiva de patrones técnicos, su cálculo y el estudio de sus patrones

Para ello , se emplearán técnica como el big data mediante librerías Pandas Python, el machine learning mediante Sklearn, XGB  y redes neuronales mediante la librería abierta de google Tensor Flow 


#Funcionamiento del código

#1 recoger datos para entrenar el modelo
yhoo_generate_big_all_csv.py
TODO news sentiment
CSV_NAME = "@VOLA"
list_companys_FVOLA = a_manage_stocks_dict.DICT_COMPANYS[CSV_NAME]
df_download = 
Se obtienen los datos de cierre , mediante yahoo API finance , y se calculan unos centenares de patrones técnicos mediante las librerías pandas_ta y talib 
yhoo_history_stock.get_favs_SCALA_csv_stocks_history_Download_list(list_companys_FVOLA, CSV_NAME, opion )

tanto para compra como venta la columna buy_seel_point tiene 0, -100, 100 , estos son detectados en función de los máximos cambios en el histórico, este punto será el de referencia de entrenamiento el ground true 
También genera los ficheros \d_price\min_max\ABNB_min_max_stock_MONTH_3.csv , los cuales guardan el max y min valor de cada columna , para que sea leido en Model_predictions_Nrows.py para un correcto y rapido  fit_scaler() 



#2 ejecutar para saber qué columnas son relevantes para la obtención del modelo
Feature_selection_create_json.py
Hay que saber cuáles de las centenares columnas de datos técnicos , es válida para entrenar el modelo neuronal, y cuales son solo ruido. Mediante correlaciones y modelos de Random Forest.
Responde a la pregunta ¿cuáles son las columnas más relevantes para puntos de compra o venta?
Gerena ficheros ej: plots_relations/best_selection_AFRM_neg.json los cuales son un raking de mejores datos técnicos para entrenar el modelo , se pretende pasar de unas 1100 columnas a unas 120



#3 entrenar los modelos TF, XGB y sklenar,
Model_creation_models_for_a_stock.py
para ello se requiere la selección de mejores columnas del punto #2

aquí hay que escoger entre modelos POS (compra) o modelo NEG (venta) y existe 
un modelo BOTH él se descarta por el momento,  ya que los modelos de predicción son binario , sólo aceptan 2 posiciones y no 3
Este punto genera modelos de predicion como XGboost_U_neg_vgood16_.sav para XGB y TF_AMZN_neg_vgood16_64.h5 para Tensor Flow


#4 evaluar la CALIDAD de esos modelos 
Model_creation_scoring.py
Los .h5 y .sav del punto anterior, si se trata de hacer una predicción con ellos tornan, valores que pueden ir del 0,2 al 3,2 para compras
cada modelo tiene una puntuación en el cual se considera punto de acción 
¿Cómo se sabe cuál es esa puntuación para cada modelo ? 
esta clase genera los ficheros, los cuales dicen cual es el punto umbral en el cual se considera punto de acción 
Models/Scoring/AFRM_neg__groupby_buy_sell_point_328.json
Models/Scoring/AFRM_neg__when_model_ok_threshold.csv

#4.1 evaluar esos BENEFICIO real de modelos
Model_predictions_N_eval_profits.py
Responde a la pregunta
¿si dejo una semana cada modelo ejecutándose, cuánto dinero hipotético  gano ?
genera los ficheros
Models/eval_Profits/_@FOLO3_ALL_stock_20221021__20221014.csv


#5 hacer predicciones en tiempo real
Model_predictions_Nrows.py
a través de la llamada a la función cada 10-12min
corresponda el real time Option_Historical.DAY_6 (100filas) y con el de entrenamiento Option_Historical.MONTH_3
df_compar, df_vender = get_RealTime_buy_seel_points(S, yhoo_history_stock.Option_Historical.DAY_6, NUM_LAST_REGISTERS_PER_STOCK=6)

#5.1 mandar alertas en tiempo real
ztelegram_send_message.py
Enlace del Bot telegram : https://t.me/Whale_Hunter_Alertbot 
existe la posibilidad de mandar las alertas de compra venta de la acción, al telegram o mail
se evalúan los múltiples modelos entrenados con los centenares de indicadores técnicos , y solo los mayores de 96% de probabilidad (según lo entrenado previamente) son notificados
Cada 15 minutos , se calculan todos los indicadores en real time y se evalúan en los modelos 
En la alerta se indica qué modelos están detentando que operaciones (compra o venta)
Estas alertas el mercado las caduca en, más menos 7 minutos
También se adjunta el precio al que se detectó, la hora, y los enlaces a las web de noticias ( las noticias siempre deben prevalecer a los indicadores técnicos) 
