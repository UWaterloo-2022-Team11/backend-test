from flask import Flask, request
from flask import current_app as app
from flask import render_template
import pickle
import numpy as np
import random
from numpy import dot
from numpy.linalg import norm

# 1032 @ 1800
# 1033 # 2700
# 1038 @ 5100
# 1039 @ 6300
# 1042 @ 8100
# 1047 @ 8700
# 1048 @ 12900

# new config batch size 500, 20,20 20 
# 1116 15900
# 1118 16400
# 1121 17400
# 1131 18500
# 1135 20300
# 1137 21250
# 1222 38850
# 1232 44250
# 1239 48450
# 1246 53250
# 115 69450
# 0130 75750
# https://towardsdatascience.com/building-a-recommender-system-for-amazon-products-with-python-8e0010ec772c
data = []
with open('new_data.pkl', 'rb') as f:
    data = pickle.load(f)

users = {}
for row in data:
    if not row[3] in users:
        users[row[3]] = []
    users[row[3]].append(row)
averages = {}
user_sample =[]
for user in users:
    vectors = []
    # print(len(users[user]))
    for row in users[user]:
        vectors.append(row[4])
    average = np.mean(vectors, axis=0)
    averages[user] = average

def get_closest_users(self, user):
    ranks = []
    for user in users:
        ranks.append((self.cos(self.point, self.averages[user]), user))
    ranks.sort(key=lambda x: x[0])
    return [ranks[0][1], ranks[2][1], ranks[3][1], ranks[4][1], ranks[5][1]]

class State():
    def __init__(self) -> None:
        self.data = data
        self.users = users
        self.averages = averages
        # user names strings
        self.cur = []
        # user names strings
        self.top5 = []
        # some vector for manipulating point
        self.v = np.random.rand(3)
        self.point = None
        self.suggestions = []
        self.lw = 0.5
        self.dw = 0.1
        self.recoms = []

    def uv(self, name):
        return self.averages[name]
    
    def reset(self):
        self.cur = [random.choice(list(users)) for i in range(3)]
        self.point = (self.v[0]*self.uv(self.cur[0])+self.v[1]*self.uv(self.cur[1])+self.v[2]*self.uv(self.cur[2]))/3.0

    def update_point(self):
        self.point = (self.v[0]*self.uv(self.cur[0])+self.v[1]*self.uv(self.cur[1])+self.v[2]*self.uv(self.cur[2]))/3.0

    def get_closest_users(self):
        ranks = []
        for user in self.users:
            ranks.append((self.cos(self.point, self.averages[user]), user))
        ranks.sort(key=lambda x: x[0])
        self.cur = [ranks[0][1], ranks[2][1], ranks[3][1]]
        self.top5 = [ranks[0][1], ranks[2][1], ranks[3][1], ranks[4][1], ranks[5][1]]

    def cos(self, a, b):
        return dot(a, b)/(norm(a)*norm(b))
    
    def update_output(self):
        self.suggestions = []
        for user in self.top5:
            user_pins = self.users[user]
            for i in range(1):
                self.suggestions.append(random.choice(user_pins))
        for i in range(3):
            user = random.choice(list(self.users))
            user_pins = self.users[user]
            self.suggestions.append(random.choice(user_pins))

    def update_from_images(self, liked, disliked):
        # for i in liked:
        #     self.point += self.lw*self.suggestions[i-1][4]
        # for i in disliked:
        #     self.point -= self.dw*self.suggestions[i-1][4]
        data = [self.suggestions[i-1][4] for i in liked]
        np.mean(data, axis=0)
        self.point = data
        self.get_closest_users()
        self.update_output()
        self.get_closest_products()

    def get_closest_products(self):
        ranks = []
        count = 0
        for user in self.users:
            for pin in self.users[user]:
                if len(pin[2]):
                    if 'amazon' in pin[2]:
                        count += 1
                        ranks.append((self.cos(self.point, pin[4]), pin))
        ranks.sort(key=lambda x: -x[0])
        self.recoms = [ranks[0][1], ranks[2][1], ranks[3][1], ranks[4][1]]

app = Flask(__name__)
state = State()

state.reset()
state.get_closest_users()
state.update_output()
state.get_closest_products()

@app.context_processor
def inject_state():
    return dict(state=state)

@app.route('/process_form', methods=['POST'])
def process_form():
    list1 = [int(x) for x in request.form['list1'].split()]
    list2 = [int(x) for x in request.form['list2'].split()]
    state.update_from_images(list1, list2)
    # Do something with the two lists of numbers
    return "Lists processed successfully"

@app.route('/')
def home():
    """Landing page."""
    return render_template(
        'index.html'
    )

@app.route('/show_user')
def show_user():
    user =  random.choice(list(users))
    user_sample = []
    
    return render_template(
        'user.html'
    )

app.run()