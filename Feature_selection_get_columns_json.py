
import json

import a_manage_stocks_dict


class JsonColumns:
    stock_id = ""
    Option_Cat = ""
    path = ""
    vgood16 = []
    good9 = []
    reg4 = []
    def __init__(self, stock_id, Option_Cat):
        NUM_VGOOD = 9
        NUM_GOOD = 4
        self.vgood16 = []
        self.good9 = []
        self.reg4 = []
        self.stock_id = stock_id
        self.Option_Cat = Option_Cat
        dict_best_feat = {}
        self.path = "plots_relations/best_selection_" + self.stock_id  + "_" + self.Option_Cat + ".json"
        print("plots_relations/best_selection_" + self.stock_id  + "_" + self.Option_Cat + ".json")
        with open(self.path) as handle:
            dict_best_feat = json.loads(handle.read())['index']  # TODO remove 'index from json
        dict_best_feat.pop('1', None)  # borro la primera por no saturar

        {k: self.vgood16.extend(v) for k, v in dict_best_feat.items() if NUM_VGOOD <= int(k)}
        {k: self.good9.extend(v) for k, v in dict_best_feat.items() if NUM_VGOOD > int(k) >= NUM_GOOD}
        {k: self.reg4.extend(v) for k, v in dict_best_feat.items() if int(k) < NUM_GOOD}

    def get_vGood_and_Good(self):
        return self.vgood16 + self.good9

    def get_All(self):
        return self.vgood16 + self.good9 + self.reg4

    def get_Dict_JsonColumns(self):
        Dict_JsonColumns = {
            a_manage_stocks_dict.MODEL_TYPE_COLM.VGOOD.value:    self.vgood16,
            a_manage_stocks_dict.MODEL_TYPE_COLM.GOOD.value:     self.get_vGood_and_Good(),
            a_manage_stocks_dict.MODEL_TYPE_COLM.REG.value:      self.get_All()
        }
        return Dict_JsonColumns

