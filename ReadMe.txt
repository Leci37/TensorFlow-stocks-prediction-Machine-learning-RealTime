# 1 recoger datos para entrenar el modelo
#TODO news sentiment
CSV_NAME = "@VOLA"
list_companys_FVOLA = a_manage_stocks_dict.DICT_COMPANYS[CSV_NAME]
df_download = yhoo_history_stock.get_favs_SCALA_csv_stocks_history_Download_list(list_companys_FVOLA, CSV_NAME, opion )
tanto para compra como venta la culumna buy_seel_point tiene 0, -100, 100


#2 ejecutar para saber que columnas son revelantes para la obtencion del modelo
Feature_selection_create_json.py
gerena ficheros ej: plots_relations/best_selection_AFRM_neg.json los cuales son un raking de mejores columnas para el modelo , se pretende pasar de 1100 columnas a +-120
tanto para compra como venta la culumna buy_seel_point tiene 0, -100, 100
responden a la pregunta cuales son las columnas más relevantes para puntos de compra o venta


#3 entrenar los modelos TF, XGB y skleanr, para ello se requiere la selecion de mejores columnas del punto 2
Model_creation_models_for_a_stock.py
aqui hay que escoger entre modelos pos (compra) o modelo neg (venta)
un modelo both , se descarta por el momento ya que los modelos de preccion son binario , solo aceptan 2 posiciones y no 3


#4 evaluar esos CALIDAD de modelos , para obtener los rakings ¿que puntuacion es valida para decir "punto de compra-venta detectado?
Model_creation_scoring.py
genera los ficheros
Models/Scoring/AFRM_neg__groupby_buy_sell_point_328.json
Models/Scoring/AFRM_neg__when_model_ok_threshold.csv

#4.1 evaluar esos BENEFICIO real de modelos ,  ¿si dejo una semana cada modelo ejecutandose, cuanto dinero gano hipotetico ?
Model_predictions_N_eval_profits.py
genera los ficheros
Models/eval_Profits/_@FOLO3_ALL_stock_20221021__20221014.csv



#5 hacer predicciones en tiempo real
Model_predictions_Nrows.py
a traves de la llamada a la funcion cada 10-12min
df_compar, df_vender = get_RealTime_buy_seel_points(S, yhoo_history_stock.Option_Historical.DAY_6, NUM_LAST_REGISTERS_PER_STOCK=6)


