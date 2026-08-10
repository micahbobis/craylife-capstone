
import os
import time
import traceback
from datetime import datetime

from app import app, socketio, db
from app.utils import get_arduino_data
from app.db_save import save_monitoring_if_needed
from app.models import MonitoringRecord

print("DB URI:", app.config.get("SQLALCHEMY_DATABASE_URI"))

# A JSON file older than this is treated as stale/disconnected.
STALE_AFTER_SECONDS = 15

# Values that mean a sensor has no usable reading.
MISSING_VALUES = {
    "",
    "-",
    "—",
    "none",
    "null",
    "nan",
    "unknown",
    "not connected",
    "disconnected",
    "error",
}


def is_missing_sensor_value(value):
    """Return True when a sensor value is missing or represents a disconnection."""
    if value is None:
        return True

    return str(value).strip().lower() in MISSING_VALUES


def is_board_disconnected(status):
    """Return True when the board status is not a valid connected state."""
    if status is None:
        return True

    normalized = str(status).strip().lower()

    return normalized in {
        "",
        "unknown",
        "not connected",
        "disconnected",
        "error",
        "offline",
    } or normalized.startswith("not connected") or normalized.startswith("error")


def disconnected_payload(message="Sensor connection lost."):
    """Build a safe payload that clears old/stale readings."""
    return {
        "Status": "Disconnected",
        "pH Level": "—",
        "pH Status": "Disconnected",
        "Water Level": "—",
        "Water Status": "Disconnected",
        "Water Quality": "—",
        "Turbidity NTU": "—",
        "pH Sensor Status": "Disconnected",
        "Water Level Sensor Status": "Disconnected",
        "Turbidity Sensor Status": "Disconnected",
        "Alert Level": "CRITICAL",
        "Alert Message": message,
        "Recommendation": (
            "Check the board, sensor wires, power supply, "
            "and network or serial connection."
        ),
    }


def apply_individual_sensor_statuses(data):
    """
    Add an individual connection status for each sensor and clear values from
    sensors that appear disconnected.
    """
    merged = dict(data or {})

    board_status = merged.get("Status", "Not connected")

    if is_board_disconnected(board_status):
        return disconnected_payload("The sensor board is disconnected.")

    merged["Status"] = "Connected"

    ph_disconnected = is_missing_sensor_value(merged.get("pH Level"))
    water_disconnected = is_missing_sensor_value(merged.get("Water Level"))

    # A turbidity sensor is considered disconnected when both its numerical
    # reading and interpreted water-quality value are unavailable.
    turbidity_disconnected = (
        is_missing_sensor_value(merged.get("Turbidity NTU"))
        and is_missing_sensor_value(merged.get("Water Quality"))
    )

    merged["pH Sensor Status"] = (
        "Disconnected" if ph_disconnected else "Connected"
    )
    merged["Water Level Sensor Status"] = (
        "Disconnected" if water_disconnected else "Connected"
    )
    merged["Turbidity Sensor Status"] = (
        "Disconnected" if turbidity_disconnected else "Connected"
    )

    if ph_disconnected:
        merged["pH Level"] = "—"
        merged["pH Status"] = "Disconnected"

    if water_disconnected:
        merged["Water Level"] = "—"
        merged["Water Status"] = "Disconnected"

    if turbidity_disconnected:
        merged["Turbidity NTU"] = "—"
        merged["Water Quality"] = "—"

    disconnected_sensors = []

    if ph_disconnected:
        disconnected_sensors.append("pH sensor")
    if water_disconnected:
        disconnected_sensors.append("water-level sensor")
    if turbidity_disconnected:
        disconnected_sensors.append("turbidity sensor")

    if disconnected_sensors:
        merged["Alert Level"] = "CRITICAL"
        merged["Alert Message"] = (
            "Disconnected: " + ", ".join(disconnected_sensors) + "."
        )
        merged["Recommendation"] = (
            "Check the indicated sensor wire, power, and signal connection."
        )

    return merged


def safe_float(value):
    """Convert a sensor value to float without raising an exception."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def sensor_broadcast_loop():
    print("[sensor_broadcast_loop] STARTED")

    # Same root JSON file used by app/utils.py.
    json_path = os.path.abspath(
        os.path.join(app.root_path, "..", "arduino_data.json")
    )

    # db_save.py expects the keys last_save and last_ph.
    save_state = {"last_save": None, "last_ph": None}

    while True:
        try:
            if not os.path.exists(json_path):
                print("[sensor_broadcast_loop] JSON file not found:", json_path)
                merged = disconnected_payload("Sensor data file was not found.")
            else:
                file_age_seconds = time.time() - os.path.getmtime(json_path)

                if file_age_seconds > STALE_AFTER_SECONDS:
                    print(
                        "[sensor_broadcast_loop] STALE JSON:",
                        round(file_age_seconds, 2),
                        "seconds old",
                    )
                    merged = disconnected_payload(
                        "No recent sensor update was received."
                    )
                else:
                    current_data = get_arduino_data() or {}
                    print("RAW DATA:", current_data)
                    merged = apply_individual_sensor_statuses(current_data)

            print("MERGED DATA:", merged)
            print(
                "[sensor_broadcast_loop] tick, merged keys:",
                list(merged.keys()) if merged else None,
            )

            ph_float = safe_float(merged.get("pH Level"))
            water_float = safe_float(merged.get("Water Level"))

            ph_status = str(merged.get("pH Status") or "Unknown")
            water_status = str(merged.get("Water Status") or "Unknown")
            arduino_status = str(merged.get("Status") or "Unknown")

            try:
                # Save only through the throttled helper to avoid duplicate
                # MonitoringRecord rows every five seconds.
                with app.app_context():
                    save_monitoring_if_needed(
                        state=save_state,
                        ph_level=ph_float,
                        ph_status=ph_status,
                        arduino_status=arduino_status,
                        water_level=water_float,
                        water_status=water_status,
                    )

                    # Save a critical alert only when a specific sensor is
                    # reported disconnected. This block does not create a
                    # WaterLevelLog every five seconds.
                    disconnected_names = []

                    if merged.get("pH Sensor Status") == "Disconnected":
                        disconnected_names.append("pH sensor")
                    if merged.get("Water Level Sensor Status") == "Disconnected":
                        disconnected_names.append("water-level sensor")
                    if merged.get("Turbidity Sensor Status") == "Disconnected":
                        disconnected_names.append("turbidity sensor")

                    if disconnected_names:
                        from app.models import MonitoringAlert

                        # Avoid inserting the same alert on every loop. Only add
                        # one if the latest alert has a different message.
                        alert_message = (
                            "Disconnected: " + ", ".join(disconnected_names)
                        )

                        latest_alert = (
                            MonitoringAlert.query
                            .order_by(MonitoringAlert.created_at.desc())
                            .first()
                        )

                        if not latest_alert or latest_alert.message != alert_message:
                            db.session.add(
                                MonitoringAlert(
                                    sensor_type="connection",
                                    value=None,
                                    message=alert_message,
                                )
                            )
                            db.session.commit()

            except Exception as error:
                db.session.rollback()
                print("DB monitoring save failed:", error)

            merged["_tick"] = int(time.time() * 1000)
            socketio.emit("sensor_update", merged)

        except Exception:
            traceback.print_exc()

        socketio.sleep(5)


if __name__ == "__main__":
    socketio.start_background_task(sensor_broadcast_loop)

    port = int(os.environ.get("PORT", 5000))

    socketio.run(
        app,
        host="0.0.0.0",
        port=port,
        debug=False,
        use_reloader=False,
    )