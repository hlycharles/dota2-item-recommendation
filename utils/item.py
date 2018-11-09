'''
extract all item name and corresponding ids
'''

import json
from multiprocessing import Pool

def get_purchased_items(match_file):
    items = set()
    with open(match_file) as f:
        match = json.load(f)
        players = match['players']
        for player in players:
            purchase_log = player['purchase_log']
            for purchase in purchase_log:
                item = purchase['key']
                items.add(item)
    return items

def store_items(items):
    out = ""
    for item_name, item_id in items.iteritems():
        line = '"' + item_name + '": ' + str(item_id) + ",\n"
        out += line
    with open('./items.txt', 'w') as f:
        f.write(out)

def build_items(matches):
    pool = Pool()
    items = pool.map(get_purchased_items, matches)
    pool.close()
    pool.join()
    all_items = reduce(lambda x, y: x.union(y), items)
    items_store = dict()
    item_count = 0
    for item in all_items:
        items_store[item] = item_count
        item_count += 1
    store_items(items_store)
