'''
Generate features from raw api responses

For each match, the generator picks a random time slice and generates a
(feature, game_result) pair for each player in the match. The feature
includes both game level features and player level features for all
10 players in the match
'''

import json
import os
from multiprocessing import Pool
from utils import constants
import sys
import random

objective_map = constants.objective_map
hero_map = constants.hero_map
unit_map = constants.unit_map
item_map = constants.item_map
LEVEL_XPS = constants.LEVEL_XPS

hero_count = 121
# is_present, time, unit, team, player_index
objective_len = 5
# is_present, is_pick, team
pickban_len = 3
# count, last_purchase_time
purchase_len = 2
# count, last_time
runes_types = 7
runes_len = 2
ability_upgrades_len = 25

data_folder = 'data_pro'
out_folder = 'examples'


# --------------------
# IO helpers
# --------------------
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

# --------------------
# Individual feature extractors
# --------------------
def get_objectives(objectives, time_slice):
    result = [0] * (len(objective_map) * objective_len)

    for objective in objectives:
        objective_time = objective['time']
        if (objective_time > time_slice * 60):
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

        # player index
        player_index = 0
        if ('player_slot' in objective):
            player_index = get_player_index(objective['player_slot']) + 1
        result[base_index + 4] = player_index

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

def get_recent_vals(vals, time_slice, count = 5):
    result = []
    start_index = min(time_slice, len(vals) - 1)
    for i in range(count):
        index = start_index - i
        val = 0
        if (index >= 0):
            val = vals[index]
        result.append(val)
    return result

def count_log(log, time_slice):
    if (log == None):
        return [0, 0]

    entry_count = 0
    most_recent_time = -1

    for entry in log:
        entry_time = entry['time']
        if (entry_time > time_slice * 60):
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

def get_purchase_log(purchase_log, time_slice):
    result = [0] * (len(item_map) * purchase_len)

    for purchase in purchase_log:
        purchase_time = purchase['time']
        purchase_name = purchase['key']
        if (purchase_time > time_slice * 60):
            continue
        if (not purchase_name in item_map):
            continue
        item_id = item_map[purchase_name]
        base_index = item_id * purchase_len
        result[base_index] += 1
        if (result[base_index] == 1 or purchase_time > result[base_index + 1]):
            result[base_index + 1] = purchase_time

    return result

def get_runes_log(runes_log, time_slice):
    result = [0] * (runes_types * runes_len)

    for runes in runes_log:
        time = runes['time']
        key = runes['key']
        if (time > time_slice * 60):
            continue
        if (key >= runes_types or key < 0):
            continue
        base_index = key * runes_len
        result[base_index] += 1
        if (result[base_index] ==1 or time > result[base_index + 1]):
            result[base_index + 1] = time

    return result

def get_ability_upgrades(xp_t, upgrades, time_slice):
    result = [0] * ability_upgrades_len
    if (upgrades == None or xp_t == None):
        return result

    xp_index = min(time_slice, len(xp_t) - 1)
    xp = xp_t[xp_index]

    level = len(LEVEL_XPS)
    while (LEVEL_XPS[level - 1] > xp):
        level -= 1

    upgrades = upgrades[:level]
    result[:len(upgrades)] = upgrades

    return result

# --------------------
# match level feature extractor
# --------------------
def get_match_feature(match, is_radiant, time_slice):
    result = []

    # duration
    result.append(time_slice)

    # first blood time
    fbt_record = match['first_blood_time']
    first_blood_time = -1 if fbt_record > time_slice * 60 else fbt_record
    result.append(first_blood_time)

    # objectives
    objectives = get_objectives(match['objectives'], time_slice)
    result.extend(objectives)

    # radiant gold advantage
    radiant_gold_adv = get_recent_vals(match['radiant_gold_adv'], time_slice)
    if (not is_radiant):
        radiant_gold_adv = map(lambda x: -x, radiant_gold_adv)
    result.extend(radiant_gold_adv)

    # radiant xp advantage
    radiant_xp_adv = get_recent_vals(match['radiant_xp_adv'], time_slice)
    if (not is_radiant):
        radiant_xp_adv = map(lambda x: -x, radiant_xp_adv)
    result.extend(radiant_xp_adv)

    # patch
    patch = match['patch']
    result.append(patch)

    return result

# --------------------
# player level feature extractor
# --------------------
def get_player_feature(player, time_slice):
    result = []

    # is radiant
    is_radiant = 1 if player['player_slot'] <= 127 else 0
    result.append(is_radiant)

    # hero id
    hero_id = player['hero_id']
    result.append(hero_id)

    # buyback log
    buyback_log = count_log(player['buyback_log'], time_slice)
    result.extend(buyback_log)

    # ability upgrades
    ability_upgrades = get_ability_upgrades(player['xp_t'], player['ability_upgrades_arr'], time_slice)
    result.extend(ability_upgrades)

    # gold_t
    gold_t = get_recent_vals(player['gold_t'], time_slice)
    result.extend(gold_t)

    # kills log
    kills_log = count_log(player['kills_log'], time_slice)
    result.extend(kills_log)

    # leaver status
    leaver_status = player['leaver_status']
    result.append(leaver_status)

    # lh_t
    lh_t = get_recent_vals(player['lh_t'], time_slice)
    result.extend(lh_t)

    # purchase log
    purchase_log = get_purchase_log(player['purchase_log'], time_slice)
    result.extend(purchase_log)

    # runes log
    runes_log = get_runes_log(player['runes_log'], time_slice)
    result.extend(runes_log)

    # sentry log
    sentry_log = count_log(player['sen_log'], time_slice)
    result.extend(sentry_log)

    # sentry left log
    sentry_left_log = count_log(player['sen_left_log'], time_slice)
    result.extend(sentry_left_log)

    # xp_t
    xp_t = get_recent_vals(player['xp_t'], time_slice)
    result.extend(xp_t)

    # rank tier
    rank_tier = 0
    if ('rank_tier' in player and player['rank_tier'] != None):
        rank_tier = player['rank_tier']
    result.append(rank_tier)

    return result

# --------------------
# feature combinator
# --------------------
def get_feature(match_file):
    match = load_data(match_file)

    # generate random time slice
    duration = match['duration']
    max_time_slice = duration / 60
    time_slice = random.randint(min(5, max_time_slice), max_time_slice)

    result = [None] * 10

    radiant_shared_features = get_match_feature(match, True, time_slice)
    dire_shared_features = get_match_feature(match, False, time_slice)

    radiant_features = []
    dire_features = []

    for player in match['players']:
        palyer_features = get_player_feature(player, time_slice)
        is_radiant = player['player_slot'] <= 127
        if (is_radiant):
            radiant_features.append(palyer_features)
        else:
            dire_features.append(palyer_features)

    radiant_win = match['radiant_win']

    for i in range(len(radiant_features)):
        radiant_feature = radiant_features[i]
        player_features = list(radiant_shared_features)
        player_features.extend(radiant_feature)
        for j in range(len(radiant_features)):
            if (i == j):
                continue
            player_features.extend(radiant_features[j])
        for dire_feature in dire_features:
            player_features.extend(dire_feature)

        player_win = 1 if radiant_win else 0
        result[i] = {
            "x": player_features,
            "y": player_win,
        }

    for i in range(len(dire_features)):
        dire_feature = dire_features[i]
        player_features = list(dire_shared_features)
        player_features.extend(dire_feature)
        for j in range(len(dire_features)):
            if (i == j):
                continue
            player_features.extend(dire_features[j])
        for radiant_feature in radiant_features:
            player_features.extend(radiant_feature)

        player_win = 0 if radiant_win else 1
        result[i + 5] = {
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

if __name__ == "__main__":
    if (len(sys.argv) > 1):
        data_folder = sys.argv[1]
    if (len(sys.argv) > 2):
        out_folder = sys.argv[2]

    matches = load_matches()
    get_features(matches)
