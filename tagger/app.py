from flask import Flask, render_template, request, url_for, redirect
import pandas as pd
import io
import glob
import random


app = Flask(__name__)
wd = "/data/calendar/"
meta = pd.read_table(wd + "meta/cal.dat")
users1 = set(meta[meta.tag == 1].user.apply(lambda x: wd + "logs/train/" + x + ".curr.log").values)
train = set(glob.glob(wd + "logs/train/*.curr.log"))

linecount = pd.read_table(wd + "meta/count_lines.out", header=None, sep=" ")
whitelist = set([wd + "logs/train/" + p for p in linecount[(linecount[0] > 29) & (linecount[0] < 200)][1].tolist()])
pool = users1 & train & whitelist


def read_log(path):
	log = pd.read_table(path, header=None)
	log.columns = ["type", "timestamp", "text", "pattern", "kkma"]
	return(log[["type", "timestamp", "text", "pattern"]])


def get_next():
	tagged = set([wd + "logs/train/" + p[(p.rfind("/") + 1):p.find(".")] + ".curr.log" for p in glob.glob(wd + "tagged/d0/*.log")])
	candidates = pool - tagged
	path = random.choice(list(candidates))
	user = path[(path.rfind("/") + 1):path.find(".")]
	log = read_log(path)
	return(user, log)


@app.route('/tagger', methods=["GET", "POST"])
def do():
	user = request.args.get("user")
	if (request.method == "POST") & (user is not None):
		command = request.args.get("command")
		selected = request.json
		output = read_log(wd + "logs/train/" + user + ".curr.log")
		output["tag"] = 0
		tag = 1
		if int(command) < 1:
			tag = -1
		output.loc[output.index.isin(selected), "tag"] = tag
		output = output[["type", "timestamp", "tag", "text", "pattern"]]
		output.to_csv(wd + "tagged/" + command + "/" + user, sep="\t", header=False, index=False)
		return("OK")
	try:
		next_user, next_log = get_next()
		next_log.timestamp = next_log.timestamp.apply(lambda x: str(x)[8:10] + ":" + str(x)[10:12] + ":" + str(x)[12:14])
		data = next_log.reset_index().to_json(orient="records", force_ascii=False)
		return(render_template("index.html", user=next_user, data=data))
	except IndexError:
		abort(400)


if __name__ == "__main__":
	app.run(host="0.0.0.0", debug=True)

