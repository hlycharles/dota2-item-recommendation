import dota2api

from utils import constants
item_map = constants.item_map

if __name__ == "__main__":
    api = dota2api.Initialise("8AF86A6D2F12B2CC2BEA05EEFB50FF0D")
    items = api.get_game_items()["items"]

    price_map = dict()
    for it in items:
        price_map[it["name"]] = it["cost"]

    id_price_map = dict()
    for k in item_map:
        k_name = "item_" + k
        item_id = item_map[k]
        item_price = price_map[k_name]
        id_price_map[item_id] = item_price

    out = ""
    out += "item_price = {\n"
    for k in id_price_map:
        out += "    " + str(k) + ": " + str(id_price_map[k]) + ",\n"
    out += "}"
    with open("price.txt", "w") as f:
        f.write(out)
