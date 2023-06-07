### DONE WITH `1_Get_technical_indicators.py` by default

---

##### 2 Filtering technical indicators (automatically by default)
This step is done automatically and the variable `GENERATED_JSON_RELATIONS` (in `get_technical_indicators.py`)  is set to True (default True). 
It is necessary to separate the technical indicators which are related to buy or sell points and which are noise. 20 seconds per share 

Three files are generated for each action in the folder: *plots_relations* , relations for purchase "pos", relations for sale "neg" and relations for both "both".

- *plots_relations/best_selection_AMD_both.json*

These files contain a ranking of which technical indicator is best for each stock. 

Check that three .json have been generated for each action in *plots_relations* .