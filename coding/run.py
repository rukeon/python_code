#!/scatter/anaconda3/envs/web/bin/python
from app import app

app.run(host="0.0.0.0", port=9000, debug=True)
