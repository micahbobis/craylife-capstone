# app/utils.py
import os
import json

from app import db
from app.models import MonitoringRecord

# EXACT SAME FILE used by arduino_reader.py
ARDUINO_JSON_FILE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "arduino_data.json")
)


def get_arduino_data():
    """Read Arduino data from JSON file and calculate pH/water status + alerts."""
    try:
        print("[get_arduino_data] READING FROM:", ARDUINO_JSON_FILE)

        with open(ARDUINO_JSON_FILE, "r") as f:
            data = json.load(f)

        print("[get_arduino_data] LOADED:", data)
        
        temperature_value = data.get("Temperature")

        data["Temperature Status"] = (
            get_temperature_status(
                temperature_value
            )
)

        ph_value = data.get("pH Level")
        ph_status = "Unknown"

        water_value = data.get("Water Level")
        water_status = "Unknown"

        try:
            ph_float = float(ph_value)

            if ph_float <= 2.99:
                ph_status = "STRONG ACID"
            elif ph_float <= 5.99:
                ph_status = "WEAK ACID"
            elif ph_float <= 6.99:
                ph_status = "SLIGHT ACID"
            elif ph_float <= 7.99:
                ph_status = "NEUTRAL"
            elif ph_float <= 8.99:
                ph_status = "SLIGHT ALKALINE"
            elif ph_float <= 11.99:
                ph_status = "WEAK BASE"
            else:
                ph_status = "STRONG BASE"

        except (ValueError, TypeError):
            ph_status = "Unknown"

        try:
            water_float = float(water_value)

            if water_float < 300:
                water_status = "LOW"
            elif water_float < 600:
                water_status = "MEDIUM"
            else:
                water_status = "HIGH"

        except (ValueError, TypeError):
            water_text = str(water_value).strip().upper()

            if water_text == "LOW":
                water_status = "LOW"
            elif water_text in ["MED", "MEDIUM"]:
                water_status = "MEDIUM"
            elif water_text == "HIGH":
                water_status = "HIGH"
            else:
                water_status = "Unknown"

        data["pH Status"] = ph_status
        data["Water Status"] = water_status

        # --- ALERT LOGIC ---
        alert_level = "NORMAL"
        alert_message = "Water conditions are stable."
        recommendation = "No action needed."

        water_quality = str(data.get("Water Quality", "Unknown")).strip().upper()
        sensor_status = str(data.get("Status", "Not connected")).strip()

        try:
            turbidity_ntu = float(data.get("Turbidity NTU", 0))
        except (ValueError, TypeError):
            turbidity_ntu = None

        if sensor_status.lower() in ["not connected", "disconnected"]:
            alert_level = "CRITICAL"
            alert_message = "Sensor connection lost."
            recommendation = "Check Arduino connection, cable, and power supply."

        elif water_quality == "DIRTY" or (turbidity_ntu is not None and turbidity_ntu >= 300):
            alert_level = "CRITICAL"
            alert_message = "Water is dirty."
            recommendation = "Change water immediately and inspect tank cleanliness."

        elif water_quality == "CLOUDY" or (turbidity_ntu is not None and 150 <= turbidity_ntu < 300):
            alert_level = "WARNING"
            alert_message = "Water is becoming cloudy."
            recommendation = "Monitor closely and prepare for cleaning or partial water change."

        elif ph_status in ["STRONG ACID", "STRONG BASE"]:
            alert_level = "CRITICAL"
            alert_message = "Unsafe pH level detected."
            recommendation = "Adjust water condition immediately."

        elif ph_status in ["WEAK ACID", "WEAK BASE", "SLIGHT ACID", "SLIGHT ALKALINE"]:
            alert_level = "WARNING"
            alert_message = "pH is outside the ideal range."
            recommendation = "Monitor pH and prepare partial water change if needed."

        elif water_status == "LOW":
            alert_level = "CRITICAL"
            alert_message = "Water level is low."
            recommendation = "Refill water immediately and check for leaks."

        elif water_status == "MEDIUM":
            alert_level = "WARNING"
            alert_message = "Water level is moderate."
            recommendation = "Monitor water level and prepare refill."

        data["Alert Level"] = alert_level
        data["Alert Message"] = alert_message
        data["Recommendation"] = recommendation

        return data

    except FileNotFoundError:
        print("[get_arduino_data] FILE NOT FOUND:", ARDUINO_JSON_FILE)
        return {
            "pH Level": "—",
            "Water Level": "—",
            "Water Quality": "—",
            "Turbidity NTU": "—",
            "Status": "Not connected",
            "pH Status": "Unknown",
            "Water Status": "Unknown",
            "Alert Level": "WARNING",
            "Alert Message": "No live sensor data available.",
            "Recommendation": "Check Arduino reader and sensor connection."
        }

    except Exception as e:
        print("[ERROR] get_arduino_data:", e)
        return {
            "pH Level": "—",
            "Water Level": "—",
            "Water Quality": "—",
            "Turbidity NTU": "—",
            "Status": "Not connected",
            "pH Status": "Unknown",
            "Water Status": "Unknown",
            "Alert Level": "WARNING",
            "Alert Message": "No live sensor data available.",
            "Recommendation": "Check Arduino reader and sensor connection."
        }


def save_monitoring_record(data):
    try:
        ph_raw = data.get("pH Level")
        water_raw = data.get("Water Level")

        try:
            ph_level = float(ph_raw)
        except (TypeError, ValueError):
            ph_level = None

        try:
            water_level = float(water_raw)
        except (TypeError, ValueError):
            water_level = None

        record = MonitoringRecord(
            ph_level=ph_level,
            ph_status=data.get("pH Status", "Unknown"),
            water_level=water_level,
            water_status=data.get("Water Status", "Unknown"),
            arduino_status=data.get("Status", "Unknown")
        )

        db.session.add(record)
        db.session.commit()
        print("[OK] Monitoring record saved")
        return True

    except Exception as e:
        db.session.rollback()
        print("[ERROR] save_monitoring_record:", e)
        return False
    
def get_temperature_status(value):
    try:
        temp = float(value)

        cold_limit = float(
            os.environ.get(
                "TEMP_COLD_LIMIT",
                22
            )
        )

        warm_limit = float(
            os.environ.get(
                "TEMP_WARM_LIMIT",
                30
            )
        )

        if temp < cold_limit:
            return "COLD"

        elif temp > warm_limit:
            return "WARM"

        return "NORMAL"

    except (TypeError, ValueError):
        return "UNKNOWN"