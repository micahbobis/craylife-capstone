# app/arduino_service.py
import time
import serial
from threading import Lock, Thread

_lock = Lock()
_latest = {"pH Level": "—", "Status": "Not connected", "pH Status": "Unknown", "ts": None}
_started = False
_arduino = None

def ph_status_from_value(ph_value):
    try:
        ph = float(ph_value)
    except Exception:
        return "Unknown"

    if ph <= 2.99: return "STRONG ACID"
    if ph <= 5.99: return "WEAK ACID"
    if ph <= 6.99: return "SLIGHT ACID"
    if ph <= 7.99: return "NEUTRAL"
    if ph <= 8.99: return "SLIGHT ALKALINE"
    if ph <= 11.99: return "WEAK BASE"
    return "STRONG BASE"

def start_reader(port="COM5", baud=9600):
    """Start Arduino serial reader once (safe vs debug reloader)."""
    global _started, _arduino
    if _started:
        return
    _started = True

    try:
        _arduino = serial.Serial(port, baud, timeout=1)
        time.sleep(2)  # allow Arduino reset
        with _lock:
            _latest["Status"] = "Connected"
            _latest["ts"] = time.time()
    except Exception as e:
        with _lock:
            _latest["Status"] = f"Not connected ({e})"
            _latest["ts"] = time.time()
        _arduino = None
        return

    def loop():
        while True:
            try:
                line = _arduino.readline().decode(errors="ignore").strip()
                if line:
                    # expected: "<ph>,<optional status>"
                    parts = [p.strip() for p in line.split(",")]
                    ph_val = parts[0]

                    with _lock:
                        _latest["pH Level"] = ph_val
                        _latest["pH Status"] = ph_status_from_value(ph_val)
                        _latest["Status"] = "Connected"
                        _latest["ts"] = time.time()
                        try:
                             from app import socketio
                             socketio.emit("sensor_update", get_latest())
                        except Exception:
                            pass
                    
            except Exception as e:
                with _lock:
                    _latest["Status"] = f"Error reading ({e})"
                    _latest["ts"] = time.time()
                time.sleep(1)

    Thread(target=loop, daemon=True).start()

def get_latest():
    with _lock:
        return dict(_latest)