from app import app
from flask import render_template, request, abort
import pandas
import glob
from sqlalchemy import create_engine

pandas.set_option("display.max_colwidth", -1)
meta = pandas.read_table("/data/calendar/meta/user.dat")
users = list(meta.user[meta.tag == 1])
accounts = {"dafunk": 0, "mk": 1, "insung": 2, "rukeon": 3, "hanseok": 4, "guest": 5}

def readLog(file, dir="/data/calendar/logs/train/"):
        log = pandas.read_table(dir + file, header=None)
        log = log.drop(log.columns[[3, 4]], axis=1)
        log[1] = log[1].apply(lambda x: str(x)[8:10] + ":" + str(x)[10:12] + ":" + str(x)[12:14])
        log.columns = ["type", "time", "text"]
        log = log[["time", "type", "text"]]
        return(log)

@app.route("/check/<string:id>")
def check(id):
        if id not in accounts:
                abort(400)
        page = request.args.get("page")
        if page is None:
                page = 0
        page = int(page)
        if page >= len(users):
                abort(400)
        index = page
        user = users[int(index)]
        log = readLog(user + "." + "curr" + ".log")
        response = int(meta[meta["user"] == user].tag)
        return(render_template("check.html", id=id, page=page, user=user, response=response, table=log.to_html(header=False, index=False)))
