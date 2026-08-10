from app import db
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    batches = db.relationship('Batch', backref='user', lazy=True)
    
    def set_password(self, password):
        self.password = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password, password)

class Batch(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    batch_name = db.Column(db.String(100), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    created_date = db.Column(db.DateTime, default=datetime.utcnow)
    quantity = db.Column(db.Integer)
    status = db.Column(db.String(50))
    inventory = db.relationship('Inventory', backref='batch', lazy=True)

class Inventory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    batch_id = db.Column(db.Integer, db.ForeignKey('batch.id'), nullable=False)
    male_count = db.Column(db.Integer, default=0)
    female_count = db.Column(db.Integer, default=0)
    total_count = db.Column(db.Integer)
    last_updated = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class ActivityLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    action = db.Column(db.String(255))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

class Sale(db.Model):
    __tablename__ = "sale"
    id = db.Column(db.Integer, primary_key=True)
    batch_id = db.Column(db.Integer, db.ForeignKey('batch.id'), nullable=False)
    quantity_sold = db.Column(db.Integer, nullable=False)
    amount = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime) 
    
    

class MonitoringRecord(db.Model):
    __tablename__ = "monitoring_record"

    id = db.Column(db.Integer, primary_key=True)
    ph_level = db.Column(db.Float)
    ph_status = db.Column(db.String(50))
    water_level = db.Column(db.Float)
    water_status = db.Column(db.String(50))
    arduino_status = db.Column(db.String(50))
    recorded_at = db.Column(db.DateTime, default=datetime.utcnow)

class ClassificationLog(db.Model):
    __tablename__ = "classification_log"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False)
    image_filename = db.Column(db.String(255), nullable=True)
    gender = db.Column(db.String(20), nullable=True)
    growth = db.Column(db.String(50), nullable=True)
    gender_confidence = db.Column(db.Float, nullable=True)
    growth_confidence = db.Column(db.Float, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
class SensorDevice(db.Model):
    __tablename__ = "sensor_device"

    id = db.Column(db.Integer, primary_key=True)
    device_name = db.Column(db.String(100))
    device_type = db.Column(db.String(50))
    location = db.Column(db.String(100))
    status = db.Column(db.String(20))
    last_seen = db.Column(db.DateTime)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
class SensorReading(db.Model):
    __tablename__ = "sensor_reading"
    # reading = SensorReading(...)
# db.session.add(reading)

    id = db.Column(db.Integer, primary_key=True)
    device_id = db.Column(db.Integer)
    ph_value = db.Column(db.Float)
    water_level = db.Column(db.Float)
    temperature = db.Column(db.Float)
    humidity = db.Column(db.Float)
    recorded_at = db.Column(db.DateTime, default=datetime.utcnow)
    
class WaterLevelLog(db.Model):
    __tablename__ = "water_level_log"

    id = db.Column(db.Integer, primary_key=True)
    level_value = db.Column(db.Float)
    status = db.Column(db.String(50))
    recorded_at = db.Column(db.DateTime, default=datetime.utcnow)
    
class MonitoringAlert(db.Model):
    __tablename__ = "monitoring_alert"

    id = db.Column(db.Integer, primary_key=True)
    sensor_type = db.Column(db.String(50))
    value = db.Column(db.Float)
    message = db.Column(db.String(255))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
