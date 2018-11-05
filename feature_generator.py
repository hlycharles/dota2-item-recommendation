import json
import os
from multiprocessing import Pool
from utils import objective_map, hero_map, unit_map, item_map
from item import build_items

time_slice_min = 60
time_slice = time_slice_min * 60
hero_count = 121
# is_present, time, unit, team, player_slot
objective_len = 5
# is_present, is_pick, team
pickban_len = 3
# count, last_purchase_time
purchase_len = 2
# count, last_time
runes_types = 7
runes_len = 2

data_folder = 'data_pro'
out_folder = 'examples'

def load_matches():
    matches = []
    subdirs = [name for name in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, name))]
    for subdir in subdirs:
        path = data_folder + '/' + subdir
        match_files = os.listdir(path)
        match_files = map(lambda p: path + '/' + p, match_files)
        matches.extend(match_files)
    return matches

def load_data(match_file):
    with open(match_file) as f:
        match = json.load(f)
        return match

def get_objectives(objectives):
    result = [0] * (len(objective_map) * objective_len)

    for objective in objectives:
        objective_time = objective['time']
        if (objective_time > time_slice):
            continue

        objective_type = objective['type']
        if ('key' in objective):
            objective_type = objective['key']
        if (not objective_type in objective_map):
            continue

        objective_id = objective_map[objective_type]
        base_index = objective_id * objective_len
        # mark as present
        result[base_index] = 1

        # time
        result[base_index + 1] = objective['time']

        # unit
        unit_id = 0
        if ('unit' in objective):
            objective_unit = objective['unit']
            if (objective_unit in hero_map):
                unit_id = hero_map[objective_unit]
            elif (objective_unit in unit_map):
                unit_id = unit_map[objective_unit] + hero_count
        result[base_index + 2] = unit_id

        # team
        team = 0
        if ('team' in objective):
            team = objective['team'] + 1
        result[base_index + 3] = team

        # player slot
        player_slot = 0
        if ('player_slot' in objective):
            player_slot = objective['player_slot']
        result[base_index + 4] = player_slot

    return result

def get_pickbans(pickbans):
    result = [0] * (hero_count * pickban_len)

    if (pickbans == None):
        return result

    for pickban in pickbans:
        hero_id = pickban['hero_id']
        base_index = (hero_id - 1) * pickban_len

        # is present
        result[base_index] = 1

        # is_pick
        is_pick = 1 if pickban['is_pick'] else 0
        result[base_index + 1] = is_pick

        # team
        result[base_index + 2] = pickban['team']

    return result

def get_recent_vals(vals, count = 5):
    result = []
    start_index = min(time_slice_min, len(vals) - 1)
    for i in range(count):
        index = start_index - i
        val = 0
        if (index >= 0):
            val = vals[index]
        result.append(val)
    return result


def get_match_feature(match):
    result = []

    # duration
    result.append(time_slice)

    # first blood time
    fbt_record = match['first_blood_time']
    first_blood_time = -1 if fbt_record > time_slice else fbt_record
    result.append(first_blood_time)

    # objectives
    objectives = get_objectives(match['objectives'])
    result.extend(objectives)

    # pickbans
    pickbans = get_pickbans(match['picks_bans'])
    result.extend(pickbans)

    # radiant gold advantage
    radiant_gold_adv = get_recent_vals(match['radiant_gold_adv'])
    result.extend(radiant_gold_adv)

    # radiant win
    # radiant_win = 1 if match['radiant_win'] else 0
    # result.append(radiant_win)

    # radiant xp advantage
    radiant_xp_adv = get_recent_vals(match['radiant_xp_adv'])
    result.extend(radiant_xp_adv)

    # patch
    patch = match['patch']
    result.append(patch)

    return result

def count_log(log):
    if (log == None):
        return []

    entry_count = 0
    most_recent_time = -1

    for entry in log:
        entry_time = entry['time']
        if (entry_time > time_slice):
            continue
        if (most_recent_time < 0 or entry_time > most_recent_time):
            most_recent_time = entry_time
        entry_count += 1

    most_recent_time = max(most_recent_time, 0)

    return [entry_count, most_recent_time]

def get_player_index(player_slot):
    if (player_slot <= 127):
        return player_slot
    return player_slot - 123

def get_player_slot(player_slot):
    result = [0] * 10
    result[get_player_index(player_slot)] = 1
    return result

def get_purchase_log(purchase_log):
    result = [0] * (len(item_map) * purchase_len)

    for purchase in purchase_log:
        purchase_time = purchase['time']
        purchase_name = purchase['key']
        if (not purchase_name in item_map):
            continue
        item_id = item_map[purchase_name]
        base_index = item_id * purchase_len
        result[base_index] += 1
        if (result[base_index] == 1 or purchase_time > result[base_index + 1]):
            result[base_index + 1] = purchase_time

    return result

def get_runes_log(runes_log):
    result = [0] * (runes_types * runes_len)

    for runes in runes_log:
        time = runes['time']
        key = runes['key']
        if (key >= runes_types):
            continue
        base_index = key * runes_len
        result[base_index] += 1
        if (result[base_index] ==1 or time > result[base_index + 1]):
            result[base_index + 1] = time

    return result


def get_player_feature(player):
    result = []

    # player slot
    player_slot = get_player_slot(player['player_slot'])
    result.extend(player_slot)

    # hero id
    hero_id = player['hero_id']
    result.append(hero_id)

    # buyback log
    buyback_log = count_log(player['buyback_log'])
    result.extend(buyback_log)

    # gold_t
    gold_t = get_recent_vals(player['gold_t'])
    result.extend(gold_t)

    # kills log
    kills_log = count_log(player['kills_log'])
    result.extend(kills_log)

    # leaver status
    leaver_status = player['leaver_status']
    result.append(leaver_status)

    # lh_t
    lh_t = get_recent_vals(player['lh_t'])
    result.extend(lh_t)

    # purchase log
    purchase_log = get_purchase_log(player['purchase_log'])
    result.extend(purchase_log)

    # runes log
    runes_log = get_runes_log(player['runes_log'])
    result.extend(runes_log)

    # sentry log
    sentry_log = count_log(player['sen_log'])
    result.extend(sentry_log)

    # sentry left log
    sentry_left_log = count_log(player['sen_left_log'])
    result.extend(sentry_left_log)

    # xp_t
    xp_t = get_recent_vals(player['xp_t'])
    result.extend(xp_t)

    # rank tier
    rank_tier = 0
    if ('rank_tier' in player and player['rank_tier'] != None):
        rank_tier = player['rank_tier']
    result.append(rank_tier)

    return result

def get_feature(match_file):
    match = load_data(match_file)

    result = [None] * 10

    shared_features = get_match_feature(match)

    for player in match['players']:
        palyer_features = get_player_feature(player)
        shared_features.extend(palyer_features)

    radiant_win = match['radiant_win']

    for player in match['players']:
        player_slot = get_player_slot(player['player_slot'])
        player_features = list(shared_features)
        player_features.extend(player_slot)

        player_index = get_player_index(player['player_slot'])
        is_radiant = player_index < 5
        player_win = 1 if radiant_win == is_radiant else 0

        result[player_index] = {
            "x": player_features,
            "y": player_win,
        }

    match_id = match['match_id']
    filename = out_folder + '/' + str(match_id) + '.json'
    with open(filename, 'w') as f:
        json.dump({
            'examples': result
        }, f)

    return result

def get_features(match_files):
    pool = Pool()
    features = pool.map(get_feature, match_files)
    pool.close()
    pool.join()
    return features

def check_valid(match_file):
    data = load_data(match_file)
    mode = data['game_mode']
    if (mode == 1 or mode == 2):
        return 1
    return 0

def get_slots(match_file):
    item = set()
    data = load_data(match_file)
    players = data['players']
    for player in players:
        player_slot = player['player_slot']
        item.add(player_slot)
    return item

def get_pb(match_file):
    item = set()
    data = load_data(match_file)
    pickbans = data['picks_bans']
    if (pickbans == None):
        return item
    for pickban in pickbans:
        hero_id = pickban['hero_id']
        item.add(hero_id)
    return item

if __name__ == "__main__":
    matches = load_matches()

    features = get_features(matches)

    # all_features = reduce(lambda x, y: x + y, features)
    # with open('./features.json', 'w') as f:
    #     json.dump({
    #         'examples': all_features
    #     }, f)

    # match = matches[2]
    # feature = get_feature(match)

    # print "total: " + str(len(matches))
    # pool = Pool()
    # valid = pool.map(check_valid, matches)
    # pool.close()
    # pool.join()
    # valid_count = sum(valid)
    # print "valid: " + str(valid_count)

    # pool = Pool()
    # player_slots = pool.map(get_pb, matches)
    # pool.close()
    # pool.join()
    # all_slots = reduce(lambda x, y: x.union(y), player_slots)
    # print all_slots
