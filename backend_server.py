from flask import Flask, request
from flask import current_app as app
from flask import render_template
import pickle
import numpy as np
import random
from numpy import dot
from numpy.linalg import norm
import base64
from flask import jsonify
import json
data = []
with open('new_data.pkl', 'rb') as f:
    data = pickle.load(f)

users = {}
global_pins = {}
for row in data:
    if not row[3] in users:
        users[row[3]] = []
    users[row[3]].append(row)
    if (row[4].shape[0] != 1024):
        print(f'wierd row: {row[4].shape}')
    global_pins[str(row[0])] = row

averages = {}
user_sample =[]
for user in users:
    vectors = []
    for row in users[user]:
        vectors.append(row[4])
    average = np.mean(vectors, axis=0)
    averages[user] = average
    
example = ''
with open('example.json') as file:
    example = json.loads(file.read())

def cos(a, b):
    return dot(a, b)/(norm(a)*norm(b))

def get_closest_users(state, ignore=''):
    ranks = []
    for user in users:
        if user != ignore:
            ranks.append((cos(state, averages[user]), user))
    ranks.sort(key=lambda x: -x[0])
    return [ranks[0][1], ranks[2][1], ranks[3][1], ranks[4][1], ranks[5][1]]

def get_random_user():
    return random.choice(list(users))


def get_pins(user, num, pins=[], pins_dict = {}):
    for i in range(num):
        pin = random.choice(users[user])
        if not (pin[0] in pins_dict):
            pins_dict[pin[0]] = 1
            pins.append({ 'id': str(pin[0]), 'img': pin[1], 'link': pin[2], 'user': pin[3] })
        else:
            print(pin[0])
            print('pin collision')
    return pins

def get_seed():
    respose = {}
    state = np.random.random((1024,)).astype('float32')
    # print(state)
    state_str = encode_state(state)
    respose['state'] = state_str
    choosen = get_closest_users(state)
    choosen.append(get_random_user())
    choosen.append(get_random_user())
    pins = []
    pins_dict = {}
    for user in choosen:
        get_pins(user, 2, pins=pins, pins_dict=pins_dict)

    respose['pins'] = pins
    return respose

def get_choices(state_str):
    respose = {}
    respose['state'] = state_str
    state = decode_state(state_str)
    print(f'Current state: {state[0:10]}')
    closest = get_closest_users(state)
    closest.append(get_random_user())
    closest.append(get_random_user())
    pins = []
    pins_dict = {}
    for user in closest:
        get_pins(user, 2, pins=pins, pins_dict=pins_dict)
    respose['pins'] = pins
    return respose

def get_closest_products(state):
    ranks = []
    count = 0
    for user in users:
        for pin in users[user]:
            if len(pin[2]):
                if 'amazon' in pin[2]:
                    count += 1
                    ranks.append((cos(state, pin[4]), pin))
    ranks.sort(key=lambda x: -x[0])
    recoms = [ranks[0][1], ranks[2][1], ranks[3][1], ranks[4][1]]
    pins = {}
    for recom in recoms:
        pins[recom[0]] = { 'id': recom[0], 'img': recom[1], 'link': recom[2] }
    return pins

def encode_state(state):
    return base64.b64encode(state.tobytes()).decode('utf-8')

def decode_state(state_str):
    state_bin = base64.b64decode(state_str.encode('utf-8'))
    state = np.frombuffer(state_bin, dtype=np.float32)
    return state

app = Flask(__name__)

@app.route('/seed', methods=['PUT'])
def seed():
    content = request.get_json(force=True)
    if (len(content['state']) > 0):
        return jsonify(get_choices(content['state']))
    response = get_seed()
    print(f'Current state_str: {response["state"][0:10]}')
    print(response['pins'])
    return jsonify(response)

@app.route('/get_recomendations', methods=['PUT'])
def get_recomendations():
    content = request.get_json(force=True)
    state = decode_state(content['state'])
    print(f'Current state: {state[0:10]}')
    choices = content['choices']
    vectors = [ state ]
    print(f'choices: {choices}')
    for choice_id in choices:
        choice = choices[choice_id]
        # vectors.append(averages[choice['user']])
        vectors.append(global_pins[choice_id][4])
    state = np.mean(vectors, axis=0)
    state_str = encode_state(state)
    pins = get_closest_products(state)
    response = { 'state': state_str, 'pins': pins }
    print(f'Current state_str: {response["state"][0:10]}')
    return response

if __name__ == '__main__':
    app.run(port=8080)

