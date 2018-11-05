import requests
import time
import json
import os

# constants
api_key = "cb88feab-20ee-46b8-8cc1-4b8733b6b45d"
api_url = "https://api.opendota.com/api/"
match_request_freq = 0.25
target_match = 50000
match_per_group = 100

MATCH_PUBLIC = 0
MATCH_PRO = 1
match_type = MATCH_PRO

data_folder = "data" if match_type == MATCH_PUBLIC else "data_pro"
endpoint = "publicMatches" if match_type == MATCH_PUBLIC else "proMatches"

DEBUG = False

def get_public_matches(maximimum_match_id):
    url = api_url + endpoint + "?less_than_match_id=" + str(maximimum_match_id)
    response = requests.get(url)
    if (response.status_code == 200):
        return response.json()

    return []

def get_match_data(id):
    url = api_url + "matches/" + str(id) + "?api_key=" + api_key
    matchResponse = requests.get(url)
    if (matchResponse.status_code == 200):
        return matchResponse.json()

    return {"error": matchResponse.status_code}

def verify_match(match):
    if ("error" in match):
        dprint("Error: " + str(match["error"]))
        return False

    if (not "players" in match):
        return False

    purchase_logs = list(map(lambda p: p["purchase_log"], match["players"]))

    if (purchase_logs[0] == None or len(purchase_logs[0]) == 0):
        dprint("No purchase")
        return False

    dprint("Pass")
    return True

def dprint(s):
    if (DEBUG):
        print(s)


if __name__ == "__main__":
    current_minimum_match_id = 9000000000
    if (os.path.exists(data_folder + "/public_match_search_id.txt")):
        id_file = open(data_folder + "/public_match_search_id.txt", "r")
        current_minimum_match_id = int(id_file.readline().rstrip())

    match_count = 0
    if (os.path.exists(data_folder + "/public_match_ids.txt")):
        ids_file = open(data_folder + "/public_match_ids.txt", "r")
        match_count = len(ids_file.readlines())

    if (not os.path.isdir(data_folder)):
        os.mkdir(data_folder)

    with open(data_folder + "/public_match_ids.txt", "a") as match_ids_file:
        while (current_minimum_match_id > 10000):
            if (match_count >= target_match):
                break

            public_matches = get_public_matches(current_minimum_match_id)
            public_match_ids = list(map(lambda x: x["match_id"], public_matches))
            for match_id in public_match_ids:
                if (match_id < current_minimum_match_id):
                    current_minimum_match_id = match_id

                time.sleep(match_request_freq)
                match = get_match_data(match_id)
                match_is_valid = verify_match(match)
                if (not match_is_valid):
                    continue

                group_index = match_count / match_per_group + 1
                group_path = data_folder + "/" + str(int(group_index))
                if (match_count % match_per_group == 0):
                    if (not os.path.isdir(group_path)):
                        os.mkdir(group_path)

                filename = group_path + "/public_match_" + str(match_id) + ".json"
                with open(filename, "w") as match_file:
                    json.dump(match, match_file)

                match_ids_file.write(str(match_id) + "\n")

                match_count += 1

            with open(data_folder + "/public_match_search_id.txt", "w") as id_file:
                id_file.write(str(current_minimum_match_id))
