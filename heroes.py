import requests
import json

api_url = "https://api.opendota.com/api/"

def get_heros():
    url = api_url + "heroes"
    response = requests.get(url)
    if (response.status_code == 200):
        return response.json()

def save_heros(heroes):
    out = ""
    for hero in heroes:
        hero_id = hero["id"]
        hero_name = hero["name"]
        line = '"' + hero_name + '"' + ": " + str(hero_id) + ",\n"
        out += line
    with open('./heroes.txt', 'w') as f:
        f.write(out)

if __name__ == '__main__':
    heroes = get_heros()
    save_heros(heroes)
