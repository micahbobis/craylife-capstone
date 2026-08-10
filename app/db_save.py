from datetime import datetime, timedelta
from app import db
from app.models import MonitoringRecord, ClassificationLog

# ===== SETTINGS (adjust if you want) =====
SAVE_EVERY_SECONDS = 60      # para di spam: save every 60 sec
PH_CHANGE_MIN = 0.10         # or save if pH change >= 0.10
KEEP_DAYS_MONITORING = 30    # auto-delete readings older than 30 days
KEEP_DAYS_CLASSIFY = 90      # auto-delete classification logs older than 90 days


def save_monitoring_if_needed(state, ph_level, ph_status, arduino_status, water_level=None, water_status=None):
    """
    state dict: keep last save time + last saved ph
    rule:
      - save every SAVE_EVERY_SECONDS
      - OR save if abs(ph - last_ph) >= PH_CHANGE_MIN
    """
    now = datetime.utcnow()
    last_save = state.get("last_save")
    last_ph = state.get("last_ph")

    should_save = False

    if last_save is None:
        should_save = True
    else:
        if (now - last_save).total_seconds() >= SAVE_EVERY_SECONDS:
            should_save = True

    if last_ph is None:
        should_save = True
    else:
        try:
            if abs(float(ph_level) - float(last_ph)) >= PH_CHANGE_MIN:
                should_save = True
        except Exception:
            pass

    if not should_save:
        return

    rec = MonitoringRecord(
    ph_level=ph_level,
    ph_status=ph_status,
    water_level=water_level,
    water_status=water_status,
    arduino_status=arduino_status,
    recorded_at=now
)
    db.session.add(rec)

    # retention cleanup
    cutoff = now - timedelta(days=KEEP_DAYS_MONITORING)
    db.session.query(MonitoringRecord).filter(MonitoringRecord.recorded_at < cutoff).delete(synchronize_session=False)

    db.session.commit()

    state["last_save"] = now
    state["last_ph"] = ph_level


def save_classification_log(user_id: int, image_filename: str, gender: str, growth: str, confidence: float):
    now = datetime.utcnow()

    log = ClassificationLog(
        user_id=user_id,
        image_filename=image_filename,
        gender=gender,
        growth=growth,
        confidence=confidence,
        created_at=now
    )
    db.session.add(log)

    cutoff = now - timedelta(days=KEEP_DAYS_CLASSIFY)
    db.session.query(ClassificationLog).filter(ClassificationLog.created_at < cutoff).delete(synchronize_session=False)

    db.session.commit()