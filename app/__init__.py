from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_wtf import CSRFProtect
from flask_mail import Mail
from itsdangerous import URLSafeTimedSerializer
from flask_socketio import SocketIO

app = Flask(__name__)
app.config.from_object('config.Config')

db = SQLAlchemy(app)
csrf = CSRFProtect(app)
mail = Mail(app)

socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

serializer = URLSafeTimedSerializer(app.config['SECRET_KEY'])

from app import routes