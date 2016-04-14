from app import app
from flask import render_template, request, abort
import pandas
import glob
from flask.ext.sqlalchemy import SQLAlchemy
import random

db = SQLAlchemy(app)

pandas.set_option("display.max_colwidth", -1)
meta = pandas.read_table("/data/calendar/meta/user.dat")
users = list(meta.user[meta.tag == 1])
accounts = {"dafunk": 0, "mk": 1, "insung": 2, "rukeon": 3, "hanseok": 4, "guest": 5}
files_1 = glob.glob("/data/calendar/logs/train/1/*")
files_0 = glob.glob("/data/calendar/logs/train/0/*")

class Tag(db.Model):
        id = db.Column(db.Integer, primary_key=True)
        user_id = db.Column(db.String, unique=True )
        tag = db.Column(db.Integer)
        ans = db.Column(db.String)

        def __init__(self, user_id, tag, ans ):
                self.user_id = user_id
                self.tag = tag
                self.ans = ans

        def __repr__(self):
                return "/data/calendar/logs/train/" + str(self.tag) + "/" + self.user_id

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///today.db'
db.create_all()

def read_log(path):
        log = pandas.read_table(path, header=None)
        log = log.drop(log.columns[[3, 4]], axis=1)
        log[1] = log[1].apply(lambda x:str(x)[8:10] + ":" + str(x)[10:12] + ":" + str(x)[12:14])
        log.columns = ["type", "time", "text"]
        log = log[["time", "type", "text"]]
        return(log)

@app.route("/viewer/calendar/<string:id>/<string:model>")
def view(id, model):
        if id not in accounts:
                abort(400)
        page = request.args.get("page")
        if page is None:
                page = 0
        page = int(page)
        if page >= len(users):
                abort(400)
        #index = accounts[id] * 100 + page
        index = page
        user = users[int(index)]
        print(" - user[" + str(index) + "]:" + user)
        log = readLog("1/" + user)
        response = int(meta[meta["user"] == user].tag)
        return(render_template("view.html", id=id, model=model, page=page, user=user, response=response, table=log.to_html(header=False, index=False)))

@app.route('/show', methods=["GET", "POST"])
def show():
        users = Tag.query.all()
        if request.method == "POST":
                data = request.json
                tag_data = Tag(data["user"], data["tag"], data["ans"])
                db.session.add(tag_data)
                db.session.commit()
                return("OK")
        print(len(users))
        result_dict = [u.__dict__ for u in users]
        tagged_list = []
        for item in result_dict:
                tagged_list.append("/data/calendar/logs/train/" +str(item["tag"]) + "/" + item["user_id"])
        files = files_0
        candidate = set(files) - set(tagged_list)
        print(len(candidate))
        path = random.choice(list(candidate))
        log = read_log(path)
        user = path[(path.rfind("/") + 1):len(path)]
        response = path[(path.rfind("/")-1):(path.rfind("/"))]
        return(render_template("check.html", user=user, response=response , table=log.to_html(header=False, index=False)))

@app.route('/busy', methods=["GET", "POST"])
def busy():
        users = Tag.query.all()
        if request.method == "POST":
                data = request.json
                tag_data = Tag(data["user"], data["tag"], data["ans"])
                db.session.add(tag_data)
                db.session.commit()
                return("OK")
        print(len(users))
        result_dict = [u.__dict__ for u in users]
        tagged_list = []
        for item in result_dict:
                tagged_list.append("/data/calendar/logs/train/" +str(item["tag"]) + "/" + item["user_id"])
        files = files_1
        candidate = set(files) - set(tagged_list)
        print(len(candidate))
        path = random.choice(list(candidate))
        log = read_log(path)
        user = path[(path.rfind("/") + 1):len(path)]
        response = path[(path.rfind("/")-1):(path.rfind("/"))]
        return(render_template("busy.html", user=user, response=response , table=log.to_html(header=False, index=False)))


@app.route('/result')
def result():
        results = Tag.query.all()
        result_dict = [u.__dict__ for u in results]
        result_tag_0 = []
        result_tag_1 = []
        for item in result_dict:
                if item["tag"] == 0:
                        result_tag_0.append((item["user_id"], item["ans"]))
                if item["tag"] == 1:
                        result_tag_1.append((item["user_id"], item["ans"]))
        total = len(result_tag_0) + len(result_tag_1)
        result_0_T = [x for x in result_tag_0 if x[1] == 'T']
        result_1_T = [x for x in result_tag_1 if x[1] == 'T']
        # meta[meta["user"] == user].tag
        black = [20151006, 20151007, 20151008, 20151012]
        red = [20151009, 20151010, 20151011]
        num_0_black = [x for x in result_tag_0 if meta[meta["user"] == x[0]].date.item() in black]
        num_0_red = [x for x in result_tag_0 if meta[meta["user"] == x[0]].date.item() in red]
        num_1_black = [x for x in result_tag_1 if meta[meta["user"] == x[0]].date.item() in black]
        num_1_red = [x for x in result_tag_1 if meta[meta["user"] == x[0]].date.item() in red]

        t_0_black = [x for x in num_0_black if x[1] == 'T']
        t_0_red = [x for x in num_0_red if x[1] == 'T']
        t_1_black = [x for x in num_1_black if x[1] == 'T']
        t_1_red = [x for x in num_1_red if x[1] == 'T']
        print(len(t_0_black)/len(num_0_black))
        print(len(t_0_red)/len(num_0_red))
        print(len(t_1_black)/len(num_1_black))
        print(len(t_1_red)/len(num_1_red))
        # print(len(num_0_black))
        # print(len(num_0_red))
        # print(len(num_1_black))
        # print(len(num_1_red))
        # print(len(result_tag_0))
        # print(len(result_tag_1))
        if len(result_tag_0) == 0:
                result_0_ratio = 0
        else:
                result_0_ratio = len(result_0_T)/len(result_tag_0)
        if len(result_tag_1) == 0:
                result_1_ratio = 0
        else:
                result_1_ratio = len(result_1_T)/len(result_tag_1)

        return(render_template('result.html', results=results, result1=result_1_ratio, result0=result_0_ratio, tag_1_black=len(num_1_black) , tag_1_red= len(num_1_red) , tag_0_black= len(num_0_black) , tag_0_red= len(num_0_red) ))
                                                                                        
