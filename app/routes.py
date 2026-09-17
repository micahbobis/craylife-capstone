from flask import render_template, redirect, url_for, flash, request, session
from app import app, db, csrf, mail, serializer, socketio
from app.forms import LoginForm, RegistrationForm, BatchForm, RequestResetForm, ResetPasswordForm
from app.models import User, Batch, Inventory, Sale, ActivityLog, ClassificationLog, BatchGrowthRecord, MonitoringAlert, AdminReportSetting
from functools import wraps
from flask_mail import Message
import secrets
import threading
import os
import uuid
import io
from datetime import datetime, timedelta, timezone
import tensorflow as tf
import traceback
import json
import shutil
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
from app.utils import get_arduino_data
from werkzeug.security import check_password_hash, generate_password_hash
from flask import jsonify
from flask_socketio import join_room
from app.db_save import save_classification_log
from pillow_heif import register_heif_opener

register_heif_opener()
from app.batch_average_growth_service import (
    SAMPLE_TARGET,
    build_period_summaries,
    next_sample_position,
)


# =========================================================
# PHILIPPINE TIME DISPLAY
# =========================================================
try:
    from zoneinfo import ZoneInfo
    PH_TZ = ZoneInfo("Asia/Manila")
except Exception:
    # Safe fallback for Windows/Python installs without tzdata.
    # The Philippines uses UTC+8 year-round.
    PH_TZ = timezone(timedelta(hours=8), name="PHT")

UTC_TZ = timezone.utc


def to_ph_time(value):
    """Convert stored UTC datetime/string values to Philippine local time."""
    if value is None:
        return None

    # Some migrated MySQL rows can arrive as strings. Normalize them first.
    if isinstance(value, str):
        raw = value.strip()
        parsed = None

        for fmt in (
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M:%S.%f",
            "%Y-%m-%d",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%S.%f",
        ):
            try:
                parsed = datetime.strptime(raw, fmt)
                break
            except ValueError:
                continue

        if parsed is None:
            try:
                parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            except ValueError:
                return None

        value = parsed

    # SQLAlchemy DateTime rows are normally naive UTC datetimes.
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC_TZ)
        return value.astimezone(PH_TZ)

    # Accept plain date-like objects safely.
    if hasattr(value, "year") and hasattr(value, "month") and hasattr(value, "day"):
        value = datetime(value.year, value.month, value.day, tzinfo=UTC_TZ)
        return value.astimezone(PH_TZ)

    return None


def _format_ph_time(value, fmt="%b %d, %Y %I:%M %p", fallback="—"):
    """Safely format datetime/date/string values in Philippine time."""
    local_time = to_ph_time(value)
    return local_time.strftime(fmt) if local_time else fallback


@app.template_filter("ph_time")
def ph_time_filter(value, fmt="%b %d, %Y %I:%M %p"):
    return _format_ph_time(value, fmt, "—")

# =========================================================
# 14-DAY GROWTH TRACKING
# =========================================================
GROWTH_CHECK_INTERVAL_DAYS = 14


def _normalize_datetime(value):
    if value is None:
        return None

    if isinstance(value, datetime):
        return value

    if isinstance(value, str):
        raw = value.strip()

        for fmt in (
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M:%S.%f",
            "%Y-%m-%d",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%S.%f",
        ):
            try:
                return datetime.strptime(raw, fmt)
            except ValueError:
                continue

        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return None

    if hasattr(value, "year") and hasattr(value, "month") and hasattr(value, "day"):
        return datetime(value.year, value.month, value.day)

    return None


def _days_between(start_dt, end_dt):
    start_dt = _normalize_datetime(start_dt)
    end_dt = _normalize_datetime(end_dt)
    if not start_dt or not end_dt:
        return None
    return max(0, (end_dt.date() - start_dt.date()).days)


def _get_official_growth_references(observations):
    """
    Build the official growth-check timeline for one batch.

    Rules:
    - First observation = baseline / first official reference.
    - Uploads made before 14 days are saved in history only.
    - The first observation at least 14 days after the last
      official reference becomes the next official reference.
    """
    if not observations:
        return []

    def observation_sort_key(obs):
        dt = _normalize_datetime(obs.created_at)
        if dt is None:
            return datetime.min
        # Compare consistently even if an ISO string included timezone info.
        return dt.replace(tzinfo=None)

    sorted_observations = sorted(
        observations,
        key=observation_sort_key
    )

    official_references = []

    for observation in sorted_observations:

        if not observation.created_at:
            continue

        if not official_references:
            official_references.append(observation)
            continue

        last_reference = official_references[-1]

        elapsed_days = _days_between(
            last_reference.created_at,
            observation.created_at
        )

        if (
            elapsed_days is not None
            and elapsed_days >= GROWTH_CHECK_INTERVAL_DAYS
        ):
            official_references.append(observation)

    return official_references


@socketio.on("connect")
def join_user_growth_room():
    user_id = session.get("user_id")
    if user_id:
        join_room(f"user_{user_id}")

def send_email_verification(user, new_email):
    token = serializer.dumps(new_email, salt='email-change')

    verify_url = url_for(
        'verify_email_change',
        token=token,
        _external=True
    )

    msg = Message(
        subject="Verify your new email - Craylife",
        recipients=[new_email],
        body=f"""
Hello {user.username},

Please confirm your new email by clicking the link below:

{verify_url}

If you did not request this change, ignore this email.
"""
    )

    mail.send(msg)

def estimate_length_and_growth_stage(img_bytes: bytes) -> dict:
    """
    Estimate crayfish length (cm) from image using contours.
    Needs calibration PX_PER_CM based on your camera setup.
    """
    debug_lines = []
    try:
        import cv2
        import numpy as np

        PX_PER_CM = 35.0  

        arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return {"estimated_length_cm": "Unknown", "growth_stage": "Unknown", "debug": "cv2 decode failed"}

        h, w = img.shape[:2]
        max_w = 900
        if w > max_w:
            scale = max_w / float(w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
            debug_lines.append(f"Resized scale={scale:.3f}")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 0)

        th = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 2
        )

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)

        cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return {"estimated_length_cm": "Unknown", "growth_stage": "Unknown", "debug": "No contours detected"}

        c = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(c)
        debug_lines.append(f"largest_area={area:.1f}")
        if area < 1500:
            return {"estimated_length_cm": "Unknown", "growth_stage": "Unknown", "debug": "Contour too small"}

        rect = cv2.minAreaRect(c)
        (_, _), (rw, rh), angle = rect
        px_len = max(rw, rh)
        debug_lines.append(f"px_len={px_len:.2f}, angle={angle:.1f}")

        length_cm = px_len / float(PX_PER_CM)
        debug_lines.append(f"PX_PER_CM={PX_PER_CM:.2f} => length_cm={length_cm:.2f}")

        if length_cm < 4.0:
            stage = "Mid Juvenile"
        elif length_cm < 7.0:
            stage = "Juvenile"
        elif length_cm < 10.0:
            stage = "Sub Adult"
        else:
            stage = "Adult"

        return {
            "estimated_length_cm": f"{length_cm:.1f} cm",
            "growth_stage": stage,
            "debug": "\n".join(debug_lines),
        }

    except Exception as e:
        return {"estimated_length_cm": "Unknown", "growth_stage": "Unknown", "debug": f"Exception: {e}"}

# =========================================================
# UNDERWATER TEMPERATURE STATUS
# Adjust these thresholds later if your crayfish species
# requires a different preferred water-temperature range.
# =========================================================
TEMP_COLD_MAX = 22.0
TEMP_NORMAL_MAX = 28.0


def get_temperature_status(value):
    """
    Convert the raw underwater temperature into a simple
    dashboard status: COLD, NORMAL, HOT, or UNKNOWN.
    """
    try:
        temp = float(value)
    except (TypeError, ValueError):
        return "UNKNOWN"

    if temp <= TEMP_COLD_MAX:
        return "COLD"

    if temp <= TEMP_NORMAL_MAX:
        return "NORMAL"

    return "HOT"


def normalize_sensor_temperature(data):
    """
    Ensure temperature-related keys are always present
    in data returned to the dashboard.
    """
    payload = dict(data or {})

    raw_temperature = payload.get("Temperature")

    if raw_temperature in (None, "", "—", "Unknown"):
        payload["Temperature"] = "—"
        payload["Temperature Status"] = "UNKNOWN"
        payload["Temperature Sensor Status"] = "Disconnected"
        return payload

    payload["Temperature Status"] = get_temperature_status(
        raw_temperature
    )
    payload["Temperature Sensor Status"] = "Connected"

    return payload


# =========================================================
# SENSOR HEARTBEAT / CONNECTION STATUS
#
# The ESP32 sends data, then spends 5 seconds on LCD screen 1
# and 5 seconds on LCD screen 2. That means the next POST can
# arrive roughly every 10 seconds.
#
# Do NOT mark the hardware disconnected just because the
# sensor values did not change. Connected/Disconnected is based
# only on how recently the ESP32 wrote a fresh payload.
# =========================================================
SENSOR_HEARTBEAT_TIMEOUT_SECONDS = 20


def get_sensor_json_path():
    return os.path.abspath(
        os.path.join(
            app.root_path,
            "..",
            "arduino_data.json"
        )
    )


def get_live_sensor_data():
    """
    Read the most recently saved ESP32 payload.

    Connected:
        JSON file exists and was refreshed within the heartbeat
        timeout, regardless of whether the values changed.

    Disconnected:
        File is missing or no new ESP32 POST has refreshed it
        within SENSOR_HEARTBEAT_TIMEOUT_SECONDS.
    """
    json_path = get_sensor_json_path()

    disconnected_payload = {
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
        "Temperature": "—",
        "Temperature Status": "UNKNOWN",
        "Temperature Sensor Status": "Disconnected",
        "Alert Level": "CRITICAL",
        "Alert Message": "No recent sensor update was received.",
        "Recommendation": (
            "Check the board, sensor wires, power supply, "
            "and network connection."
        )
    }

    if not os.path.exists(json_path):
        return disconnected_payload

    try:
        file_age_seconds = max(
            0.0,
            datetime.now().timestamp()
            - os.path.getmtime(json_path)
        )

        if file_age_seconds > SENSOR_HEARTBEAT_TIMEOUT_SECONDS:
            return disconnected_payload

        with open(json_path, "r", encoding="utf-8") as sensor_file:
            payload = json.load(sensor_file)

        payload = normalize_sensor_temperature(payload)

        # A fresh file means the ESP32 is alive, even if every
        # measurement is identical to the previous reading.
        payload["Status"] = "Connected"

        return payload

    except Exception as error:
        print(
            "[get_live_sensor_data] ERROR:",
            error,
            flush=True
        )
        return disconnected_payload

import time



SENSOR_TIMEOUT_SECONDS = 12


def _log_monitoring_alert_once(sensor_type, value, message, cooldown_minutes=5):
    """
    Save a monitoring incident without creating a duplicate row every
    few seconds while the same condition remains active.
    """
    cutoff = datetime.utcnow() - timedelta(minutes=cooldown_minutes)

    recent_same_alert = (
        MonitoringAlert.query
        .filter(
            MonitoringAlert.sensor_type == sensor_type,
            MonitoringAlert.message == message,
            MonitoringAlert.created_at >= cutoff
        )
        .order_by(MonitoringAlert.created_at.desc())
        .first()
    )

    if recent_same_alert:
        return

    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        numeric_value = None

    db.session.add(
        MonitoringAlert(
            sensor_type=sensor_type,
            value=numeric_value,
            message=message,
            created_at=datetime.utcnow()
        )
    )

    try:
        db.session.commit()

        _send_sensor_issue_report(
            sensor_type=sensor_type,
            value=value,
            message=message
        )

    except Exception:
        db.session.rollback()


# =========================================================
# ADMIN REPORT NOTIFICATION HELPERS
# AIVEN / DATABASE-BACKED SETTINGS
# =========================================================

DAILY_SUMMARY_HOUR_PH = 8
DAILY_SUMMARY_MINUTE_PH = 0

_daily_summary_thread_started = False
_daily_summary_thread_lock = threading.Lock()


def _get_or_create_admin_report_setting(user_id, user_email=None):
    """
    Get one user's persistent notification settings from MySQL/Aiven.
    """
    setting = (
        AdminReportSetting.query
        .filter_by(user_id=user_id)
        .first()
    )

    if setting:
        return setting

    user = User.query.get(user_id)

    default_email = (
        user_email
        or (user.email if user else None)
        or app.config.get("MAIL_USERNAME")
        or ""
    )

    setting = AdminReportSetting(
        user_id=user_id,
        admin_email=default_email,
        sensor_alerts_enabled=True,
        growth_alerts_enabled=True,
        daily_summary_enabled=False
    )

    db.session.add(setting)

    try:
        db.session.commit()
        return setting
    except Exception:
        db.session.rollback()
        raise


def _get_admin_report_preferences(user_id=None, user_email=None):
    """
    Return database-backed admin-report preferences for one user.
    """
    if user_id is None:
        user_id = session.get("user_id")

    if not user_id:
        return {
            "admin_email": user_email or app.config.get("MAIL_USERNAME") or "",
            "sensor_alerts": "yes",
            "growth_alerts": "yes",
            "daily_summary": "no",
            "last_daily_summary_date": None
        }

    setting = _get_or_create_admin_report_setting(
        user_id=user_id,
        user_email=user_email
    )

    return {
        "admin_email": setting.admin_email or "",
        "sensor_alerts": (
            "yes"
            if setting.sensor_alerts_enabled
            else "no"
        ),
        "growth_alerts": (
            "yes"
            if setting.growth_alerts_enabled
            else "no"
        ),
        "daily_summary": (
            "yes"
            if setting.daily_summary_enabled
            else "no"
        ),
        "last_daily_summary_date": (
            setting.last_daily_summary_date.isoformat()
            if setting.last_daily_summary_date
            else None
        )
    }


def _save_admin_report_preferences(
    user_id,
    admin_email,
    sensor_alerts,
    growth_alerts,
    daily_summary
):
    """
    Persist Settings page notification preferences to Aiven/MySQL.
    """
    try:
        setting = _get_or_create_admin_report_setting(
            user_id=user_id,
            user_email=admin_email
        )

        setting.admin_email = admin_email
        setting.sensor_alerts_enabled = (
            sensor_alerts == "yes"
        )
        setting.growth_alerts_enabled = (
            growth_alerts == "yes"
        )
        setting.daily_summary_enabled = (
            daily_summary == "yes"
        )
        setting.updated_at = datetime.utcnow()

        db.session.commit()
        return True

    except Exception as error:
        db.session.rollback()

        print(
            "[ADMIN REPORT SETTINGS SAVE ERROR]",
            error,
            flush=True
        )

        return False


def _mark_daily_summary_sent(user_id, date_key):
    """
    Persist the most recent daily-summary date.
    """
    try:
        setting = (
            AdminReportSetting.query
            .filter_by(user_id=user_id)
            .first()
        )

        if not setting:
            return False

        setting.last_daily_summary_date = datetime.strptime(
            date_key,
            "%Y-%m-%d"
        ).date()

        setting.updated_at = datetime.utcnow()

        db.session.commit()
        return True

    except Exception as error:
        db.session.rollback()

        print(
            "[DAILY SUMMARY MARK ERROR]",
            error,
            flush=True
        )

        return False


def _send_admin_report_email(subject, body, recipient):
    """
    Send one Craylife admin email using the configured Flask-Mail sender.
    """
    if not recipient:
        print(
            "[ADMIN REPORT EMAIL] No recipient configured",
            flush=True
        )
        return False

    try:
        msg = Message(
            subject=subject,
            sender=app.config.get("MAIL_USERNAME"),
            recipients=[recipient],
            body=body
        )

        mail.send(msg)

        print(
            f"[ADMIN REPORT EMAIL] Sent to {recipient}: {subject}",
            flush=True
        )

        return True

    except Exception as error:
        print(
            "[ADMIN REPORT EMAIL ERROR]",
            error,
            flush=True
        )

        return False


def _get_enabled_sensor_recipients():
    """
    Return unique configured admin emails with sensor alerts enabled.
    """
    recipients = []

    settings_rows = (
        AdminReportSetting.query
        .filter_by(sensor_alerts_enabled=True)
        .all()
    )

    for setting in settings_rows:
        email = (setting.admin_email or "").strip()

        if email and email not in recipients:
            recipients.append(email)

    # Safe fallback before a user has opened Settings for the first time.
    if not recipients:
        fallback = app.config.get("MAIL_USERNAME")

        if fallback:
            recipients.append(fallback)

    return recipients


def _send_sensor_issue_report(sensor_type, value, message):
    """
    Email admins using the exact sensor issue recorded by the system.
    """
    live_sensor = get_live_sensor_data()

    value_text = "—"

    if value is not None:
        try:
            value_text = f"{float(value):.2f}"
        except (TypeError, ValueError):
            value_text = str(value)

    now_ph = to_ph_time(datetime.utcnow())

    subject = f"Craylife Sensor Alert - {sensor_type}"

    body = f"""CRAYLIFE SYSTEM REPORT

Date: {now_ph.strftime('%B %d, %Y')}
Time: {now_ph.strftime('%I:%M %p')}

STATUS
Attention Required

ISSUE DETECTED
Sensor: {sensor_type}
Current Reading: {value_text}
Issue: {message}

CURRENT SENSOR CONDITIONS
Board Connection: {live_sensor.get('Status', 'Unknown')}
pH Level: {live_sensor.get('pH Level', '—')}
Water Level: {live_sensor.get('Water Level', '—')}
Water Quality: {live_sensor.get('Water Quality', '—')}
Temperature: {live_sensor.get('Temperature', '—')}
Temperature Status: {live_sensor.get('Temperature Status', 'UNKNOWN')}

RECOMMENDED ACTION
Review the affected sensor or water condition and confirm whether
the issue continues.

This is an automated report generated by the Craylife Monitoring System.
"""

    sent_any = False

    for recipient in _get_enabled_sensor_recipients():
        if _send_admin_report_email(
            subject,
            body,
            recipient
        ):
            sent_any = True

    return sent_any


def _send_growth_due_report(
    batch,
    latest_observation,
    status,
    next_check_date
):
    """
    Send a batch growth alert using actual saved observations.
    """
    preferences = _get_admin_report_preferences(
        user_id=batch.user_id
    )

    if preferences.get("growth_alerts") != "yes":
        return False

    recipient = preferences.get("admin_email")

    if not recipient:
        return False

    latest_stage = (
        latest_observation.growth.replace("_", " ").title()
        if latest_observation and latest_observation.growth
        else "Unknown"
    )

    latest_length = (
        f"{latest_observation.estimated_length_cm:.1f} cm"
        if latest_observation
        and latest_observation.estimated_length_cm is not None
        else "Unknown"
    )

    now_ph = to_ph_time(datetime.utcnow())

    subject = f"Craylife Growth Report - {batch.batch_name}"

    body = f"""CRAYLIFE GROWTH REPORT

Date: {now_ph.strftime('%B %d, %Y')}
Time: {now_ph.strftime('%I:%M %p')}

Batch: {batch.batch_name}
Latest Stage: {latest_stage}
Latest Estimated Length: {latest_length}
Growth Status: {status}

Next Check:
{_format_ph_time(next_check_date, '%B %d, %Y', 'Not available')}

NEXT ACTION
Upload a new batch observation when the scheduled growth verification
is due so the system can complete the official comparison.

This is an automated report generated by the Craylife Monitoring System.
"""

    return _send_admin_report_email(
        subject,
        body,
        recipient
    )


def _build_daily_summary_for_user(user):
    """
    Build one daily report from the same actual data used by
    Home, Reports, MonitoringAlert, ActivityLog and ClassificationLog.
    """
    now_utc = datetime.utcnow()
    now_ph = to_ph_time(now_utc)
    since_utc = now_utc - timedelta(hours=24)

    live_sensor = get_live_sensor_data()

    recent_alerts = (
        MonitoringAlert.query
        .filter(
            MonitoringAlert.created_at >= since_utc
        )
        .order_by(
            MonitoringAlert.created_at.desc()
        )
        .limit(10)
        .all()
    )

    recent_activities = (
        ActivityLog.query
        .filter(
            ActivityLog.user_id == user.id,
            ActivityLog.timestamp >= since_utc
        )
        .order_by(
            ActivityLog.timestamp.desc()
        )
        .limit(10)
        .all()
    )

    batches = (
        Batch.query
        .filter_by(user_id=user.id)
        .order_by(Batch.created_date.asc())
        .all()
    )

    batch_lines = []
    due_count = 0

    for batch in batches:
        observations = (
            ClassificationLog.query
            .filter_by(
                user_id=user.id,
                batch_id=batch.id
            )
            .order_by(
                ClassificationLog.created_at.asc()
            )
            .all()
        )

        if not observations:
            batch_lines.append(
                f"- {batch.batch_name}: No growth observation yet"
            )
            continue

        official_refs = _get_official_growth_references(
            observations
        )

        last_verified = (
            official_refs[-1]
            if official_refs
            else observations[0]
        )

        latest = observations[-1]

        days_since_verified = (
            _days_between(
                last_verified.created_at,
                now_utc
            )
            or 0
        )

        if days_since_verified >= GROWTH_CHECK_INTERVAL_DAYS:
            status = "VERIFICATION DUE"
            due_count += 1
        elif latest.id != last_verified.id:
            status = "PROVISIONAL"
        elif len(official_refs) >= 2:
            previous = official_refs[-2]

            if (
                last_verified.estimated_length_cm is not None
                and previous.estimated_length_cm is not None
            ):
                change = (
                    last_verified.estimated_length_cm
                    - previous.estimated_length_cm
                )

                if change > 0:
                    status = "GROWTH DETECTED"
                elif change < 0:
                    status = "SIZE DECREASE"
                else:
                    status = "STABLE"
            else:
                status = "VERIFIED"
        else:
            status = "BASELINE"

        latest_stage = (
            latest.growth.replace("_", " ").title()
            if latest.growth
            else "Unknown"
        )

        latest_length = (
            f"{latest.estimated_length_cm:.1f} cm"
            if latest.estimated_length_cm is not None
            else "Unknown"
        )

        batch_lines.append(
            f"- {batch.batch_name}: "
            f"{latest_stage}, {latest_length}, {status}"
        )

    alert_lines = []

    for alert in recent_alerts:
        alert_time = to_ph_time(
            alert.created_at
        ).strftime("%I:%M %p")

        alert_lines.append(
            f"- {alert_time} | "
            f"{alert.sensor_type}: {alert.message}"
        )

    activity_lines = []

    for activity in recent_activities:
        activity_time = to_ph_time(
            activity.timestamp
        ).strftime("%I:%M %p")

        activity_lines.append(
            f"- {activity_time} | {activity.action}"
        )

    if not batch_lines:
        batch_lines = ["- No tracked batches."]

    if not alert_lines:
        alert_lines = ["- No monitoring alerts in the last 24 hours."]

    if not activity_lines:
        activity_lines = ["- No recent system activity."]

    body = f"""CRAYLIFE DAILY ADMIN REPORT

Date: {now_ph.strftime('%B %d, %Y')}
Generated: {now_ph.strftime('%I:%M %p')}

CURRENT SENSOR HEALTH
Board Connection: {live_sensor.get('Status', 'Unknown')}
pH Level: {live_sensor.get('pH Level', '—')}
Water Level: {live_sensor.get('Water Level', '—')}
Water Quality: {live_sensor.get('Water Quality', '—')}
Temperature: {live_sensor.get('Temperature', '—')}
Temperature Status: {live_sensor.get('Temperature Status', 'UNKNOWN')}

MONITORING ALERTS - LAST 24 HOURS
{chr(10).join(alert_lines)}

BATCH GROWTH STATUS
Tracked Batches: {len(batches)}
Verification Due: {due_count}
{chr(10).join(batch_lines)}

RECENT SYSTEM ACTIVITY - LAST 24 HOURS
{chr(10).join(activity_lines)}

This is the once-daily automated admin summary generated by
the Craylife Monitoring System.
"""

    return body


def _daily_summary_worker():
    """
    Send at most ONE daily summary per enabled user per
    Philippine calendar day.

    Render caveat:
    This worker runs while the Render web service is awake.
    If Render sleeps, it sends today's unsent summary after wake-up.
    """
    while True:
        try:
            with app.app_context():
                now_ph = datetime.now(PH_TZ)
                today = now_ph.date()

                scheduled_time_reached = (
                    now_ph.hour > DAILY_SUMMARY_HOUR_PH
                    or (
                        now_ph.hour == DAILY_SUMMARY_HOUR_PH
                        and now_ph.minute >= DAILY_SUMMARY_MINUTE_PH
                    )
                )

                if scheduled_time_reached:
                    enabled_settings = (
                        AdminReportSetting.query
                        .filter_by(
                            daily_summary_enabled=True
                        )
                        .all()
                    )

                    for setting in enabled_settings:
                        if (
                            setting.last_daily_summary_date
                            == today
                        ):
                            continue

                        user = User.query.get(
                            setting.user_id
                        )

                        if not user:
                            continue

                        recipient = (
                            setting.admin_email
                            or user.email
                        )

                        body = _build_daily_summary_for_user(
                            user
                        )

                        subject = (
                            "Craylife Daily Admin Report - "
                            + now_ph.strftime("%B %d, %Y")
                        )

                        if _send_admin_report_email(
                            subject,
                            body,
                            recipient
                        ):
                            _mark_daily_summary_sent(
                                user.id,
                                today.strftime("%Y-%m-%d")
                            )

        except Exception as error:
            print(
                "[DAILY SUMMARY WORKER ERROR]",
                error,
                flush=True
            )

        time.sleep(60)


def _ensure_daily_summary_thread():
    global _daily_summary_thread_started

    if _daily_summary_thread_started:
        return

    with _daily_summary_thread_lock:
        if _daily_summary_thread_started:
            return

        thread = threading.Thread(
            target=_daily_summary_worker,
            daemon=True,
            name="craylife-daily-summary"
        )

        thread.start()

        _daily_summary_thread_started = True

        print(
            "[DAILY SUMMARY] Scheduler started "
            "(08:00 AM Philippine time, once per day)",
            flush=True
        )

@app.route("/arduino-data")
def arduino_data():

    json_path = os.path.abspath(
        os.path.join(
            app.root_path,
            "..",
            "arduino_data.json"
        )
    )

    disconnected = {
        "Status": "Disconnected",

        "pH Level": "—",
        "pH Status": "Disconnected",

        "Water Level": "—",
        "Water Status": "Disconnected",

        "Water Quality": "—",
        "Turbidity NTU": "—",

        "Temperature": "—",
        "Temperature Status": "UNKNOWN",

        "pH Sensor Status": "Disconnected",
        "Water Level Sensor Status": "Disconnected",
        "Turbidity Sensor Status": "Disconnected",
        "Temperature Sensor Status": "Disconnected",

        "Alert Level": "CRITICAL",

        "Alert Message":
            "Sensor hardware is disconnected.",

        "Recommendation":
            "Check the ESP32 power, USB connection, "
            "sensor wiring, and Wi-Fi connection."
    }

    # ==========================================
    # NO SENSOR FILE YET
    # ==========================================
    if not os.path.exists(json_path):

        _log_monitoring_alert_once(
            "Board",
            None,
            "Sensor hardware is disconnected."
        )

        return jsonify(
            disconnected
        )


    # ==========================================
    # CHECK LAST ESP32 HEARTBEAT
    # ==========================================
    try:

        age_seconds = (
            time.time()
            - os.path.getmtime(json_path)
        )

    except OSError:

        _log_monitoring_alert_once(
            "Board",
            None,
            "Sensor hardware is disconnected."
        )

        return jsonify(
            disconnected
        )


    # ESP32 sends every ~5 seconds.
    # Allow two missed updates + small network delay.
    if age_seconds > SENSOR_TIMEOUT_SECONDS:

        _log_monitoring_alert_once(
            "Board",
            None,
            "Sensor hardware is disconnected."
        )

        return jsonify(
            disconnected
        )


    # ==========================================
    # HARDWARE IS ACTIVE
    # ==========================================
    try:

        with open(
            json_path,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

    except Exception:

        _log_monitoring_alert_once(
            "Board",
            None,
            "Sensor hardware is disconnected."
        )

        return jsonify(
            disconnected
        )


    data = normalize_sensor_temperature(
        data
    )

    data["Status"] = "Connected"

    return jsonify(data)


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in first', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function


@app.before_request
def log_request():
    _ensure_daily_summary_thread()

    print(
        f"[REQUEST] {request.method} {request.path}",
        flush=True
    )


@app.route("/esp32-data", methods=["POST"])
@csrf.exempt
def sensor_update_api():
    try:
        data = request.get_json(silent=True) or {}

        ph = data.get("ph") or data.get("pH Level")
        water_level = data.get("water_level") or data.get("Water Level")
        water_quality = data.get("water_quality") or data.get("Water Quality")
        turbidity_ntu = data.get("turbidity_ntu") or data.get("Turbidity NTU")
        temperature = data.get("temperature")

        if temperature is None:
            temperature = data.get("Temperature")

        temperature_status = get_temperature_status(
            temperature
        )

        temperature_sensor_status = (
            "Connected"
            if temperature not in (None, "", "—", "Unknown")
            else "Disconnected"
        )

        payload = {
            "pH Level": str(ph) if ph is not None else "—",
            "pH Voltage": str(data.get("pH Voltage", "—")),
            "Water Level": str(water_level).upper() if water_level is not None else "—",
            "Water Quality": str(water_quality).upper() if water_quality is not None else "—",
            "Turbidity NTU": str(turbidity_ntu) if turbidity_ntu is not None else "—",
            "Turbidity Voltage": str(data.get("Turbidity Voltage", "—")),
            "Turbidity ADC": str(data.get("Turbidity ADC", "—")),
            "Temperature": str(temperature) if temperature is not None else "—",
            "Temperature Status": temperature_status,
            "Temperature Sensor Status": temperature_sensor_status,
            "Status": "Connected"
        }

        # ------------------------------------------------------
        # REPORTABLE SENSOR CONDITIONS
        # Keep these records for the Reports page.
        # ------------------------------------------------------
        try:
            ph_numeric = float(ph) if ph is not None else None
        except (TypeError, ValueError):
            ph_numeric = None

        if ph_numeric is not None and not (6.5 <= ph_numeric <= 8.5):
            _log_monitoring_alert_once(
                "pH",
                ph_numeric,
                f"pH out of recommended range: {ph_numeric:.2f}"
            )

        try:
            temp_numeric = (
                float(temperature)
                if temperature is not None
                else None
            )
        except (TypeError, ValueError):
            temp_numeric = None

        if temp_numeric is not None and temperature_status in {"COLD", "HOT"}:
            _log_monitoring_alert_once(
                "Temperature",
                temp_numeric,
                f"Underwater temperature status is {temperature_status}."
            )

        if str(water_level or "").strip().upper() in {
            "LOW",
            "CRITICAL",
            "EMPTY"
        }:
            _log_monitoring_alert_once(
                "Water Level",
                None,
                f"Water level status is {str(water_level).upper()}."
            )

        if str(water_quality or "").strip().upper() in {
            "POOR",
            "DIRTY",
            "HIGH",
            "CRITICAL"
        }:
            _log_monitoring_alert_once(
                "Turbidity",
                turbidity_ntu,
                f"Water quality status is {str(water_quality).upper()}."
            )

        json_path = get_sensor_json_path()

        temp_file = json_path + ".tmp"
        with open(temp_file, "w") as f:
            json.dump(payload, f, indent=4)
        os.replace(temp_file, json_path)

        print("[/esp32-data] RECEIVED:", data)
        print("[/esp32-data] JSON UPDATED:", payload)

        return jsonify({
            "success": True,
            "message": "Sensor data received",
            "data": payload
        }), 200

    except Exception as e:
        print("[/esp32-data] ERROR:", e)
        return jsonify({
            "success": False,
            "message": str(e)
        }), 500
        
@app.route('/')
@app.route('/home')
def home():
    user_id = session.get('user_id')
    from app.models import BatchGrowthRecord
    batches = (
        Batch.query.filter_by(user_id=user_id).all()
        if user_id
        else []
    )

    sensor_data_raw = get_live_sensor_data()

    ph_value = sensor_data_raw.get("pH Level", "Unknown")
    ph_status = "Unknown"

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
            ph_status = "SLIGHT ALK"
        elif ph_float <= 11.99:
            ph_status = "WEAK BASE"
        else:
            ph_status = "STRONG BASE"

    except (ValueError, TypeError):
        ph_status = "Unknown"

    water_level = sensor_data_raw.get("Water Level", "Unknown")
    
    temperature = sensor_data_raw.get(
        "Temperature",
        "—"
    )

    temperature_status = sensor_data_raw.get(
        "Temperature Status",
        get_temperature_status(temperature)
    )

    sensor_data = [
        {
            "name": "pH Level",
            "value": ph_value,
            "unit": "pH"
        },
        {
            "name": "pH Status",
            "value": ph_status,
            "unit": ""
        },
        {
            "name": "Water Level",
            "value": water_level,
            "unit": "%"
        },
        {
            "name": "Arduino Status",
            "value": sensor_data_raw.get(
                "Status",
                "Not connected"
            ),
            "unit": ""
        },
        {
            "name": "Water Quality",
            "value": sensor_data_raw.get(
                "Water Quality",
                "Unknown"
            ),
            "unit": ""
        },
        {
            "name": "Turbidity NTU",
            "value": sensor_data_raw.get(
                "Turbidity NTU",
                "Unknown"
            ),
            "unit": "NTU"
            
        },
        
        {
            "name": "Underwater Temperature",
            "value": temperature,
            "unit": "°C"
        },
        {
            "name": "Temperature Status",
            "value": temperature_status,
            "unit": ""
        },
    ]

    # =========================================================
    # GROWTH MONITORING
    #
    # IMPORTANT:
    # Weekly Batch Growth is derived from the REAL saved
    # ClassificationLog observations of each batch.
    # No random/manual BatchGrowthRecord values are used here.
    # =========================================================
    growth_logs = []
    latest_growth = None
    growth_chart_labels = []
    growth_chart_values = []
    batch_growth_summaries = []

    if user_id:
        # -----------------------------------------------------
        # Overall recent growth classifications
        # -----------------------------------------------------
        growth_logs = (
            ClassificationLog.query
            .filter_by(user_id=user_id)
            .filter(ClassificationLog.growth.isnot(None))
            .order_by(ClassificationLog.created_at.desc())
            .limit(10)
            .all()
        )

        if growth_logs:
            latest_growth = growth_logs[0]

            stage_order = {
                "mid_juvenile": 1,
                "juvenile": 2,
                "mid_adult": 3,
                "sub_adult": 3,
                "adult": 4,
            }

            for log in reversed(growth_logs):
                label = _format_ph_time(
                    log.created_at,
                    "%b %d %I:%M %p",
                    "Unknown"
                )

                growth_chart_labels.append(
                    label
                )

                growth_key = (
                    log.growth or ""
                ).lower()

                growth_chart_values.append(
                    stage_order.get(
                        growth_key,
                        0
                    )
                )

        # -----------------------------------------------------
        # LIVE 14-DAY BATCH GROWTH MONITORING
        #
        # Realtime meaning here:
        # - water/sensor values are truly live from the ESP32
        # - growth is a LIVE PROJECTION between official image checks
        # - actual growth is verified by eligible observations that are
        #   at least 14 days apart
        # -----------------------------------------------------
        user_batches = (
            Batch.query
            .filter_by(user_id=user_id)
            .order_by(Batch.created_date.asc())
            .all()
        )

        now_for_growth = datetime.utcnow()

        for batch in user_batches:
            observations = (
                ClassificationLog.query
                .filter_by(
                    user_id=user_id,
                    batch_id=batch.id
                )
                .order_by(
                    ClassificationLog.created_at.asc()
                )
                .all()
            )

            if not observations:
                continue

            latest_observation = observations[-1]

            official_references = (
                _get_official_growth_references(
                    observations
                )
            )

            if not official_references:
                continue

            last_verified = official_references[-1]
            previous_verified = (
                official_references[-2]
                if len(official_references) >= 2
                else None
            )

            days_since_verified = (
                _days_between(
                    last_verified.created_at,
                    now_for_growth
                )
                or 0
            )

            cycle_day = min(
                days_since_verified,
                GROWTH_CHECK_INTERVAL_DAYS
            )

            days_remaining = max(
                0,
                GROWTH_CHECK_INTERVAL_DAYS
                - days_since_verified
            )

            cycle_progress_percent = round(
                min(
                    100.0,
                    (
                        cycle_day
                        / GROWTH_CHECK_INTERVAL_DAYS
                    ) * 100.0
                ),
                1
            )

            last_verified_dt = _normalize_datetime(last_verified.created_at)
            next_verification_date = (
                last_verified_dt
                + timedelta(days=GROWTH_CHECK_INTERVAL_DAYS)
                if last_verified_dt
                else None
            )

            verification_due = (
                days_since_verified
                >= GROWTH_CHECK_INTERVAL_DAYS
            )

            last_verified_length = (
                last_verified.estimated_length_cm
            )

            # ---------------------------------------------
            # Growth-rate projection
            # ---------------------------------------------
            daily_growth_rate_cm = None
            predicted_current_length_cm = None
            projected_change_cm = None

            if (
                previous_verified
                and previous_verified.estimated_length_cm is not None
                and last_verified_length is not None
            ):
                verified_interval_days = (
                    _days_between(
                        previous_verified.created_at,
                        last_verified.created_at
                    )
                    or 0
                )

                if verified_interval_days > 0:
                    daily_growth_rate_cm = (
                        last_verified_length
                        - previous_verified.estimated_length_cm
                    ) / verified_interval_days

                    # Do not extrapolate farther than one 14-day cycle.
                    projection_days = min(
                        days_since_verified,
                        GROWTH_CHECK_INTERVAL_DAYS
                    )

                    predicted_current_length_cm = round(
                        last_verified_length
                        + (
                            daily_growth_rate_cm
                            * projection_days
                        ),
                        2
                    )

                    projected_change_cm = round(
                        predicted_current_length_cm
                        - last_verified_length,
                        2
                    )

            # ---------------------------------------------
            # Provisional upload information
            # ---------------------------------------------
            latest_is_official = (
                latest_observation.id
                == last_verified.id
            )

            provisional_length_cm = (
                latest_observation.estimated_length_cm
                if not latest_is_official
                else None
            )

            provisional_date = (
                latest_observation.created_at
                if not latest_is_official
                else None
            )

            if verification_due:
                live_growth_status = (
                    'VERIFICATION DUE'
                )
            elif not latest_is_official:
                live_growth_status = (
                    'PROVISIONAL OBSERVATION'
                )
            elif len(official_references) >= 2:
                live_growth_status = (
                    'LIVE PROJECTION'
                )
            else:
                live_growth_status = (
                    'BASELINE RECORDED'
                )

            # ---------------------------------------------
            # Mini chart: verified measurements + projection
            # ---------------------------------------------
            chart_labels = []
            chart_actual = []
            chart_projection = []

            for ref in official_references:
                chart_labels.append(
                    _format_ph_time(
                        ref.created_at,
                        '%b %d',
                        'Unknown'
                    )
                )

                chart_actual.append(
                    round(ref.estimated_length_cm, 2)
                    if ref.estimated_length_cm is not None
                    else None
                )

                chart_projection.append(None)

            if (
                predicted_current_length_cm is not None
                and last_verified_length is not None
            ):
                # Start the dashed prediction from the last verified point.
                if chart_projection:
                    chart_projection[-1] = round(
                        last_verified_length,
                        2
                    )

                chart_labels.append('Today')
                chart_actual.append(None)
                chart_projection.append(
                    predicted_current_length_cm
                )

            batch_growth_summaries.append({
                'batch_id': batch.id,
                'batch_name': batch.batch_name,
                'batch_quantity': batch.quantity,

                'observation_count': len(observations),
                'official_check_count': len(official_references),

                'growth_stage': latest_observation.growth,
                'growth_confidence': latest_observation.growth_confidence,

                'last_verified_length_cm': last_verified_length,
                'last_verified_date': last_verified.created_at,

                'predicted_current_length_cm': predicted_current_length_cm,
                'projected_change_cm': projected_change_cm,
                'daily_growth_rate_cm': (
                    round(daily_growth_rate_cm, 4)
                    if daily_growth_rate_cm is not None
                    else None
                ),

                'provisional_length_cm': provisional_length_cm,
                'provisional_date': provisional_date,

                'cycle_day': cycle_day,
                'cycle_total_days': GROWTH_CHECK_INTERVAL_DAYS,
                'cycle_progress_percent': cycle_progress_percent,
                'days_remaining': days_remaining,
                'next_verification_date': next_verification_date,
                'verification_due': verification_due,

                'growth_status': live_growth_status,

                'chart_labels': chart_labels,
                'chart_actual': chart_actual,
                'chart_projection': chart_projection,
            })

    return render_template(
        'dashboard/home.html',
        batches=batches,
        sensor_data=sensor_data,
        latest_growth=latest_growth,
        growth_logs=growth_logs,
        growth_chart_labels=growth_chart_labels,
        growth_chart_values=growth_chart_values,
        batch_growth_summaries=batch_growth_summaries
    )
    
    
@app.route('/password_reset', methods=['GET', 'POST'])
def password_reset():

    form = RequestResetForm()   

    if form.validate_on_submit():

        user = User.query.filter_by(email=form.email.data).first()

        if user:

            token = serializer.dumps(user.email, salt="password-reset")

            reset_url = url_for(
                "reset_password_token",
                token=token,
                _external=True
            )

            msg = Message(
                subject="Reset your Craylife password",
                recipients=["micahavrill14@gmail.com"],
                body=f"""
Hello {user.username},

A password reset was requested.

Click the link below to reset your password:

{reset_url}

If you did not request this, ignore this email.
"""
            )

            mail.send(msg)

        flash("If the email exists, a reset link has been sent.", "info")

    return render_template("auth/reset_request.html", form=form)   # ← ADD form


@app.route('/settings', methods=['GET', 'POST'])
@login_required
def settings():
    user_id = session.get('user_id')
    user = User.query.get(user_id) if user_id else None

    if not user:
        flash('User account could not be found.', 'danger')
        return redirect(url_for('login'))

    default_settings = _get_admin_report_preferences(
        user_id=user.id,
        user_email=user.email
    )

    if request.method == 'POST':
        form_action = request.form.get('form_action', '').strip()

        if form_action == 'account':
            username = request.form.get('username', '').strip()
            new_email = request.form.get('email', '').strip()

            if not username or not new_email:
                flash('Username and email are required.', 'danger')
                return redirect(url_for('settings'))

            user.username = username

            if new_email != user.email:
                try:
                    token = serializer.dumps(new_email, salt='email-change')
                    verify_url = url_for(
                        'verify_email_change',
                        token=token,
                        _external=True
                    )

                    msg = Message(
                        subject="Verify your new email - Craylife",
                        recipients=[new_email],
                        body=f"""
Hello {user.username},

You requested to change your email to:

{new_email}

Please verify it by clicking the link below:

{verify_url}

If you did not request this change, ignore this message.
"""
                    )

                    mail.send(msg)
                    db.session.commit()

                    flash(
                        'Account information updated. '
                        'A verification email was sent to the new address.',
                        'success'
                    )

                except Exception as error:
                    db.session.rollback()
                    print("[SETTINGS EMAIL UPDATE ERROR]", error, flush=True)
                    flash(
                        'The account update could not be completed because '
                        'the verification email could not be sent.',
                        'danger'
                    )

                return redirect(url_for('settings'))

            db.session.commit()
            flash('Account information updated successfully.', 'success')
            return redirect(url_for('settings'))

        if form_action == 'password_request':
            current_password = request.form.get('current_password', '')
            new_password = request.form.get('new_password', '')
            confirm_password = request.form.get('confirm_password', '')

            if not user.check_password(current_password):
                flash('Current password is incorrect.', 'danger')
                return redirect(url_for('settings'))

            if len(new_password) < 8:
                flash(
                    'New password must contain at least 8 characters.',
                    'danger'
                )
                return redirect(url_for('settings'))

            if new_password != confirm_password:
                flash('New passwords do not match.', 'danger')
                return redirect(url_for('settings'))

            verification_code = f"{secrets.randbelow(1000000):06d}"

            session['password_change_code_hash'] = generate_password_hash(
                verification_code
            )
            session['pending_password_hash'] = generate_password_hash(
                new_password
            )
            session['password_change_expires_at'] = (
                datetime.utcnow() + timedelta(minutes=10)
            ).timestamp()

            try:
                msg = Message(
                    subject="Your Craylife password verification code",
                    recipients=[user.email],
                    body=f"""
Hello {user.username},

We received a request to change your Craylife password.

Your verification code is:

{verification_code}

This code expires in 10 minutes.

If you did not request this password change, you can ignore this email.
"""
                )
                mail.send(msg)

            except Exception as error:
                print("[PASSWORD CODE EMAIL ERROR]", error, flush=True)

                session.pop('password_change_code_hash', None)
                session.pop('pending_password_hash', None)
                session.pop('password_change_expires_at', None)

                flash(
                    'The verification code could not be sent. '
                    'Your password was not changed.',
                    'danger'
                )
                return redirect(url_for('settings'))

            flash(
                f'A 6-digit verification code was sent to {user.email}. '
                'Enter it below to complete the password change.',
                'success'
            )

            return redirect(url_for('settings', verify_password='1'))

        if form_action == 'password_verify':
            entered_code = request.form.get(
                'verification_code',
                ''
            ).strip()

            code_hash = session.get('password_change_code_hash')
            pending_password_hash = session.get('pending_password_hash')
            expires_at = session.get('password_change_expires_at')

            if not code_hash or not pending_password_hash or not expires_at:
                flash(
                    'There is no pending password change. '
                    'Please request a new verification code.',
                    'danger'
                )
                return redirect(url_for('settings'))

            if datetime.utcnow().timestamp() > float(expires_at):
                session.pop('password_change_code_hash', None)
                session.pop('pending_password_hash', None)
                session.pop('password_change_expires_at', None)

                flash(
                    'The verification code has expired. '
                    'Please request a new one.',
                    'danger'
                )
                return redirect(url_for('settings'))

            if not check_password_hash(code_hash, entered_code):
                flash('The verification code is incorrect.', 'danger')
                return redirect(url_for('settings', verify_password='1'))

            try:
                user.password = pending_password_hash
                db.session.commit()

                session.pop('password_change_code_hash', None)
                session.pop('pending_password_hash', None)
                session.pop('password_change_expires_at', None)

                flash('Password updated successfully.', 'success')

            except Exception as error:
                db.session.rollback()
                print("[PASSWORD VERIFY UPDATE ERROR]", error, flush=True)
                flash(
                    'Password could not be updated. Please try again.',
                    'danger'
                )

            return redirect(url_for('settings'))

        if form_action == 'password_cancel':
            session.pop('password_change_code_hash', None)
            session.pop('pending_password_hash', None)
            session.pop('password_change_expires_at', None)

            flash('Pending password change cancelled.', 'info')
            return redirect(url_for('settings'))

        if form_action == 'preferences':
            admin_email = request.form.get(
                'admin_email',
                ''
            ).strip()

            sensor_alerts = request.form.get(
                'sensor_alerts',
                'yes'
            )

            growth_alerts = request.form.get(
                'growth_alerts',
                'yes'
            )

            daily_summary = request.form.get(
                'daily_summary',
                'no'
            )

            if not admin_email:
                admin_email = user.email

            if _save_admin_report_preferences(
                user_id=user.id,
                admin_email=admin_email,
                sensor_alerts=sensor_alerts,
                growth_alerts=growth_alerts,
                daily_summary=daily_summary
            ):
                flash(
                    'Admin report notification settings saved.',
                    'success'
                )
            else:
                flash(
                    'Report settings could not be saved.',
                    'danger'
                )

            return redirect(url_for('settings'))

        flash('Unknown settings action.', 'warning')
        return redirect(url_for('settings'))

    password_verification_pending = bool(
        session.get('password_change_code_hash')
        and session.get('pending_password_hash')
        and session.get('password_change_expires_at')
    )

    return render_template(
        'dashboard/settings.html',
        user=user,
        settings=default_settings,
        password_verification_pending=password_verification_pending
    )

@app.route('/activity', methods=['GET', 'POST'])
@login_required
def activity():
    if request.method == 'POST':
        description = request.form.get('description')

        if description:
            new_activity = ActivityLog(
                user_id=session.get('user_id') or 1,
                action=description
            )
            db.session.add(new_activity)
            db.session.commit()
            flash('Activity added successfully.', 'success')

        return redirect(url_for('activity'))

    page = request.args.get('page', 1, type=int)

    activities = ActivityLog.query.order_by(
        ActivityLog.timestamp.desc()
    ).paginate(page=page, per_page=5, error_out=False)

    return render_template('dashboard/activity.html', activities=activities)
    
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from flask import send_file
import io 

@app.route('/download-report')
@login_required
def download_report():

    user_id = session.get('user_id')

    batches = Batch.query.filter_by(user_id=user_id).all()
    activities = ActivityLog.query.filter_by(user_id=user_id).all()

    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)

    pdf.setFont("Helvetica", 14)
    pdf.drawString(200, 750, "Craylife System Report")

    y = 720

    pdf.setFont("Helvetica", 10)
    pdf.drawString(50, y, "Batch Summary")
    y -= 20

    for batch in batches:
        pdf.drawString(50, y, f"Batch: {batch.batch_name} | Quantity: {batch.quantity}")
        y -= 20

    y -= 20
    pdf.drawString(50, y, "Recent Activities")
    y -= 20

    for act in activities[:10]:
        local_act_time = to_ph_time(act.timestamp)
        local_act_text = local_act_time.strftime("%Y-%m-%d %I:%M %p") if local_act_time else "Unknown"
        pdf.drawString(50, y, f"{act.action} ({local_act_text})")
        y -= 20

        if y < 100:
            pdf.showPage()
            y = 750

    pdf.save()

    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name="craylife_report.pdf",
        mimetype="application/pdf"
    )
    
    
@app.route('/reports')
@login_required
def reports():
    user_id = session.get('user_id')

    alert_page = request.args.get('alert_page', 1, type=int)
    growth_page = request.args.get('growth_page', 1, type=int)
    per_page = 5

    # =========================================================
    # LIVE SENSOR REPORT
    # =========================================================
    live_sensor = get_live_sensor_data()

    sensor_rows = [
        {
            'name': 'Board Connection',
            'value': live_sensor.get('Status', 'Disconnected'),
            'status': (
                'NORMAL'
                if live_sensor.get('Status') == 'Connected'
                else 'CRITICAL'
            )
        },
        {
            'name': 'pH Level',
            'value': live_sensor.get('pH Level', '—'),
            'status': live_sensor.get('pH Status', 'Unknown')
        },
        {
            'name': 'Water Level',
            'value': live_sensor.get('Water Level', '—'),
            'status': live_sensor.get('Water Status', 'Unknown')
        },
        {
            'name': 'Water Quality',
            'value': live_sensor.get('Water Quality', '—'),
            'status': (
                'NORMAL'
                if str(live_sensor.get('Water Quality', '')).upper()
                in {'CLEAR', 'GOOD', 'NORMAL'}
                else str(live_sensor.get('Water Quality', 'Unknown')).upper()
            )
        },
        {
            'name': 'Temperature',
            'value': live_sensor.get('Temperature', '—'),
            'status': live_sensor.get('Temperature Status', 'UNKNOWN')
        }
    ]

    active_sensor_issues = sum(
        1
        for item in sensor_rows
        if str(item['status']).upper()
        not in {
            'NORMAL',
            'CONNECTED',
            'NEUTRAL',
            'GOOD',
            'CLEAR',
            'OK'
        }
    )

    # =========================================================
    # SENSOR ALERT HISTORY
    # =========================================================
    alerts_pagination = (
        MonitoringAlert.query
        .order_by(MonitoringAlert.created_at.desc())
        .paginate(
            page=alert_page,
            per_page=per_page,
            error_out=False
        )
    )

    # =========================================================
    # GROWTH REPORT
    # =========================================================
    batches = (
        Batch.query
        .filter_by(user_id=user_id)
        .order_by(Batch.created_date.desc())
        .all()
    )

    growth_reports = []
    verification_due_count = 0

    now_for_growth = datetime.utcnow()

    for batch in batches:
        observations = (
            ClassificationLog.query
            .filter_by(
                user_id=user_id,
                batch_id=batch.id
            )
            .order_by(ClassificationLog.created_at.asc())
            .all()
        )

        if not observations:
            growth_reports.append({
                'batch_id': batch.id,
                'batch_name': batch.batch_name,
                'latest_stage': 'No observation',
                'latest_length_cm': None,
                'observation_count': 0,
                'last_observation_date': None,
                'days_since_verified': None,
                'next_check_date': None,
                'status': 'NO DATA'
            })
            continue

        official_refs = _get_official_growth_references(observations)
        last_verified = official_refs[-1] if official_refs else observations[0]
        latest = observations[-1]

        days_since_verified = (
            _days_between(
                last_verified.created_at,
                now_for_growth
            )
            or 0
        )

        last_verified_dt = _normalize_datetime(last_verified.created_at)
        next_check_date = (
            last_verified_dt
            + timedelta(days=GROWTH_CHECK_INTERVAL_DAYS)
            if last_verified_dt
            else None
        )

        if days_since_verified >= GROWTH_CHECK_INTERVAL_DAYS:
            growth_status = 'VERIFICATION DUE'
            verification_due_count += 1

            alert_key = (
                f"growth_due_sent_{batch.id}_"
                f"{last_verified_dt.date() if last_verified_dt else 'unknown'}"
            )

            if not session.get(alert_key):
                if _send_growth_due_report(
                    batch=batch,
                    latest_observation=latest,
                    status=growth_status,
                    next_check_date=next_check_date
                ):
                    session[alert_key] = True
        elif latest.id != last_verified.id:
            growth_status = 'PROVISIONAL'
        elif len(official_refs) >= 2:
            previous_verified = official_refs[-2]

            if (
                last_verified.estimated_length_cm is not None
                and previous_verified.estimated_length_cm is not None
            ):
                change = (
                    last_verified.estimated_length_cm
                    - previous_verified.estimated_length_cm
                )

                if change > 0:
                    growth_status = 'GROWTH DETECTED'
                elif change < 0:
                    growth_status = 'SIZE DECREASE'
                else:
                    growth_status = 'STABLE'
            else:
                growth_status = 'VERIFIED'
        else:
            growth_status = 'BASELINE'

        growth_reports.append({
            'batch_id': batch.id,
            'batch_name': batch.batch_name,
            'latest_stage': (
                latest.growth.replace('_', ' ').title()
                if latest.growth
                else 'Unknown'
            ),
            'latest_length_cm': latest.estimated_length_cm,
            'observation_count': len(observations),
            'last_observation_date': latest.created_at,
            'days_since_verified': days_since_verified,
            'next_check_date': next_check_date,
            'status': growth_status
        })

    growth_total = len(growth_reports)
    growth_pages = max(
        1,
        (growth_total + per_page - 1) // per_page
    )
    growth_page = min(
        max(growth_page, 1),
        growth_pages
    )

    growth_start = (growth_page - 1) * per_page
    growth_end = growth_start + per_page
    growth_page_items = growth_reports[
        growth_start:growth_end
    ]

    recent_activities = (
        ActivityLog.query
        .filter_by(user_id=user_id)
        .order_by(ActivityLog.timestamp.desc())
        .limit(5)
        .all()
    )

    return render_template(
        'dashboard/reports.html',
        sensor_rows=sensor_rows,
        active_sensor_issues=active_sensor_issues,
        alerts=alerts_pagination,
        growth_reports=growth_page_items,
        growth_total=growth_total,
        growth_page=growth_page,
        growth_pages=growth_pages,
        verification_due_count=verification_due_count,
        tracked_batch_count=len(batches),
        recent_activities=recent_activities
    )

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()

    if request.method == 'POST':
        print("[LOGIN] POST received", flush=True)

    if form.validate_on_submit():
        try:
            user = User.query.filter_by(
                username=form.username.data
            ).first()

            print(
                f"[LOGIN] user_found={user is not None}",
                flush=True
            )

            if user and user.check_password(form.password.data):
                session.clear()
                session['user_id'] = user.id
                session['username'] = user.username
                session['user_email'] = user.email

                flash('Login successful!', 'success')

                print(
                    f"[LOGIN] success user_id={user.id}; redirecting to home",
                    flush=True
                )

                return redirect(url_for('home'))

            print("[LOGIN] invalid credentials", flush=True)
            flash('Invalid username or password', 'danger')

        except Exception:
            print("[LOGIN] unexpected error", flush=True)
            traceback.print_exc()
            db.session.rollback()

            flash(
                'A server error occurred while signing in. Please try again.',
                'danger'
            )

    elif request.method == 'POST':
        print(
            f"[LOGIN] validation errors={form.errors}",
            flush=True
        )

    return render_template(
        'auth/login.html',
        form=form
    )


@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegistrationForm()
    if form.validate_on_submit():
        user = User(username=form.username.data, email=form.email.data)
        user.set_password(form.password.data)
        db.session.add(user)
        db.session.commit()
        flash('Registration successful! You can now log in.', 'success')
        return redirect(url_for('login'))
    return render_template('auth/register.html', form=form)


@app.route('/dashboard')
@login_required
def dashboard():
    user_id = session.get('user_id')
    batches = Batch.query.filter_by(user_id=user_id).all()
    return render_template('dashboard/home.html', batches=batches)


@app.route('/batch/create', methods=['GET', 'POST'])
@login_required
def create_batch():
    user_id = session.get('user_id')

    if request.method == 'POST':

        batch_name = (
            request.form.get('batch_name', '')
            .strip()
        )

        quantity = request.form.get(
            'quantity',
            type=int
        )

        if not batch_name:
            flash(
                'Please enter a batch name.',
                'danger'
            )
            return render_template(
                'dashboard/batch_form.html'
            )

        if quantity is None or quantity < 0:
            flash(
                'Please enter a valid quantity.',
                'danger'
            )
            return render_template(
                'dashboard/batch_form.html'
            )

        existing_batch = (
            Batch.query
            .filter_by(
                user_id=user_id,
                batch_name=batch_name
            )
            .first()
        )

        if existing_batch:
            flash(
                'A batch with this name already exists.',
                'danger'
            )
            return render_template(
                'dashboard/batch_form.html'
            )

        # ==========================================
        # CREATE BATCH RECORD
        # ==========================================
        batch = Batch(
            batch_name=batch_name,
            user_id=user_id,
            quantity=quantity,
            status='Active'
        )

        db.session.add(batch)
        db.session.flush()

        # ==========================================
        # CREATE BATCH FOLDER
        #
        # Example:
        # static/uploads/batches/12/
        # ==========================================
        batch_folder = os.path.join(
            app.root_path,
            'static',
            'uploads',
            'batches',
            str(batch.id)
        )

        os.makedirs(
            batch_folder,
            exist_ok=True
        )

        # ==========================================
        # RECEIVE ALL FILES FROM SELECTED FOLDER
        # ==========================================
        uploaded_files = request.files.getlist(
            'batch_folder'
        )

        allowed_extensions = {
            'jpg',
            'jpeg',
            'png',
            'heic',
            'heif'
        }

        saved_count = 0

        for file in uploaded_files:

            if not file or not file.filename:
                continue

            original_name = (
                file.filename
                .replace('\\', '/')
                .split('/')[-1]
            )

            filename = secure_filename(
                original_name
            )

            if not filename:
                continue

            if '.' not in filename:
                continue

            extension = (
                filename
                .rsplit('.', 1)[1]
                .lower()
            )

            if extension not in allowed_extensions:
                continue

            if extension in {'heic', 'heif'}:
                try:
                    image = Image.open(file.stream).convert("RGB")

                    base_name = os.path.splitext(filename)[0]

                    unique_filename = (
                        f"{uuid.uuid4().hex}_"
                        f"{base_name}.jpg"
                    )

                    destination = os.path.join(
                        batch_folder,
                        unique_filename
                    )

                    image.save(
                        destination,
                        "JPEG",
                        quality=95
                    )

                    saved_count += 1

                except Exception as error:
                    print(
                        "[HEIC CONVERSION ERROR]",
                        filename,
                        error
                    )

                    continue

            else:
                unique_filename = (
                    f"{uuid.uuid4().hex}_"
                    f"{filename}"
                )

                destination = os.path.join(
                    batch_folder,
                    unique_filename
                )

                file.save(
                    destination
                )

                saved_count += 1

        db.session.commit()

        activity_log = ActivityLog(
            user_id=user_id,
            action=(
                f"Created batch "
                f"{batch.batch_name} "
                f"with {saved_count} "
                f"reference image(s)"
            ),
            timestamp=datetime.utcnow()
        )

        db.session.add(
            activity_log
        )

        db.session.commit()

        flash(
            f'Batch "{batch.batch_name}" created '
            f'with {saved_count} image(s).',
            'success'
        )

        # Go back to classifier so bagong batch
        # appears immediately in dropdown.
        return redirect(
            url_for('classify')
        )

    return render_template(
        'dashboard/batch_form.html'
    )



# =========================================================
# BATCH FOLDERS / MANAGEMENT
# =========================================================
@app.route('/batches')
@login_required
def batch_folders():
    user_id = session.get('user_id')

    batches = (
        Batch.query
        .filter_by(user_id=user_id)
        .order_by(Batch.created_date.desc())
        .all()
    )

    batch_items = []

    for batch in batches:
        observation_query = (
            ClassificationLog.query
            .filter_by(
                user_id=user_id,
                batch_id=batch.id
            )
        )

        observation_count = observation_query.count()

        latest_observation = (
            observation_query
            .order_by(
                ClassificationLog.created_at.desc()
            )
            .first()
        )

        batch_items.append({
            "batch": batch,
            "observation_count": observation_count,
            "latest_observation": latest_observation
        })

    return render_template(
        'dashboard/batches.html',
        batch_items=batch_items
    )


@app.route(
    '/batch/<int:batch_id>/edit',
    methods=['GET', 'POST']
)
@login_required
def edit_batch(batch_id):
    user_id = session.get('user_id')

    batch = (
        Batch.query
        .filter_by(
            id=batch_id,
            user_id=user_id
        )
        .first_or_404()
    )

    if request.method == 'POST':
        batch_name = (
            request.form.get(
                'batch_name',
                ''
            )
            .strip()
        )

        quantity = request.form.get(
            'quantity',
            type=int
        )

        if not batch_name:
            flash(
                'Batch name is required.',
                'danger'
            )

            return render_template(
                'dashboard/edit_batch.html',
                batch=batch
            )

        if quantity is None or quantity < 0:
            flash(
                'Please enter a valid quantity.',
                'danger'
            )

            return render_template(
                'dashboard/edit_batch.html',
                batch=batch
            )

        duplicate = (
            Batch.query
            .filter(
                Batch.user_id == user_id,
                Batch.batch_name == batch_name,
                Batch.id != batch.id
            )
            .first()
        )

        if duplicate:
            flash(
                'Another batch already uses that name.',
                'danger'
            )

            return render_template(
                'dashboard/edit_batch.html',
                batch=batch
            )

        old_name = batch.batch_name

        batch.batch_name = batch_name
        batch.quantity = quantity

        db.session.add(
            ActivityLog(
                user_id=user_id,
                action=(
                    f'Updated batch "{old_name}" '
                    f'to "{batch.batch_name}" '
                    f'with quantity {batch.quantity}'
                ),
                timestamp=datetime.utcnow()
            )
        )

        db.session.commit()

        flash(
            'Batch updated successfully.',
            'success'
        )

        return redirect(
            url_for(
                'batch_detail',
                batch_id=batch.id
            )
        )

    return render_template(
        'dashboard/edit_batch.html',
        batch=batch
    )


@app.route(
    '/batch/<int:batch_id>/delete',
    methods=['POST']
)
@login_required
def delete_batch_record(batch_id):
    user_id = session.get('user_id')

    batch = (
        Batch.query
        .filter_by(
            id=batch_id,
            user_id=user_id
        )
        .first_or_404()
    )

    batch_name = batch.batch_name

    try:
        # Remove linked image-classification observations first.
        ClassificationLog.query.filter_by(
            user_id=user_id,
            batch_id=batch.id
        ).delete(
            synchronize_session=False
        )

        # Remove manual growth records for this batch.
        BatchGrowthRecord.query.filter_by(
            batch_id=batch.id
        ).delete(
            synchronize_session=False
        )

        # Inventory may reference the batch.
        Inventory.query.filter_by(
            batch_id=batch.id
        ).delete(
            synchronize_session=False
        )

        # Remove the physical reference-image folder.
        batch_folder = os.path.join(
            app.root_path,
            'static',
            'uploads',
            'batches',
            str(batch.id)
        )

        if os.path.isdir(batch_folder):
            shutil.rmtree(
                batch_folder,
                ignore_errors=True
            )

        db.session.delete(batch)

        db.session.add(
            ActivityLog(
                user_id=user_id,
                action=f'Deleted batch "{batch_name}"',
                timestamp=datetime.utcnow()
            )
        )

        db.session.commit()

        flash(
            f'Batch "{batch_name}" deleted successfully.',
            'success'
        )

    except Exception as error:
        db.session.rollback()

        print(
            "[ERROR] Failed to delete batch:",
            error
        )

        flash(
            'Batch could not be deleted because it is still linked to other records.',
            'danger'
        )

    return redirect(
        url_for('batch_folders')
    )

@app.route('/inventory')
@login_required
def inventory():
    user_id = session.get('user_id')

    classification_logs = (
        ClassificationLog.query
        .filter_by(user_id=user_id)
        .order_by(ClassificationLog.created_at.desc())
        .all()
    )

    total_classifications = len(classification_logs)

    male_count = sum(
        1 for log in classification_logs
        if (log.gender or '').strip().lower() == 'male'
    )

    female_count = sum(
        1 for log in classification_logs
        if (log.gender or '').strip().lower() == 'female'
    )

    batch_classifications = sum(
        1 for log in classification_logs
        if log.batch_id is not None
    )

    quick_classifications = (
        total_classifications
        - batch_classifications
    )

    growth_stage_counts = {
        'mid_juvenile': 0,
        'juvenile': 0,
        'mid_adult': 0,
        'sub_adult': 0,
        'adult': 0
    }

    for log in classification_logs:
        key = (log.growth or '').strip().lower()

        if key in growth_stage_counts:
            growth_stage_counts[key] += 1

    recent_records = []

    for log in classification_logs[:12]:
        recent_records.append({
            'id': log.id,
            'image_filename': log.image_filename,
            'gender': log.gender or 'Unknown',
            'growth': (
                log.growth.replace('_', ' ').title()
                if log.growth
                else 'Unknown'
            ),
            'gender_confidence': log.gender_confidence,
            'growth_confidence': log.growth_confidence,
            'estimated_length_cm': log.estimated_length_cm,
            'created_at': log.created_at,
            'batch_name': (
                log.batch.batch_name
                if log.batch
                else None
            ),
            'mode': (
                'Batch Tracking'
                if log.batch_id is not None
                else 'Quick Classify'
            )
        })

    classification_summary = {
        'total_classifications': total_classifications,
        'male_count': male_count,
        'female_count': female_count,
        'batch_classifications': batch_classifications,
        'quick_classifications': quick_classifications,
        'growth_stage_counts': growth_stage_counts
    }

    return render_template(
        'dashboard/inventory.html',
        classification_summary=classification_summary,
        recent_records=recent_records
    )

@app.route('/item/edit/<int:item_id>', methods=['GET', 'POST'])
@login_required
def edit_item(item_id):
    item = Inventory.query.get_or_404(item_id)
    if request.method == 'POST':
        item.male_count = request.form.get('male_count', type=int, default=0)
        item.female_count = request.form.get('female_count', type=int, default=0)
        item.total_count = item.male_count + item.female_count
        db.session.commit()
        flash('Item updated successfully.', 'success')
        return redirect(url_for('inventory'))

    return render_template('dashboard/edit_item.html', item=item)


@app.route('/item/delete/<int:item_id>', methods=['POST'])
@login_required
def delete_item(item_id):
    item = Inventory.query.get_or_404(item_id)
    db.session.delete(item)
    db.session.commit()
    flash('Item deleted successfully.', 'success')
    return redirect(url_for('inventory'))

IMG_SIZE = 224

GROWTH_LABELS = ["adult", "juvenile", "mid_adult", "mid_juvenile"]
GENDER_LABELS = ["Female", "Male"]

MODEL_DIR = os.path.join(app.root_path, "models")

GENDER_MODEL_PATH = os.path.join(MODEL_DIR, "crayfish_gender_resnet50.h5")
GROWTH_MODEL_PATH = os.path.join(MODEL_DIR, "crayfish_growth_resnet50.h5")

print("=== MODEL DEBUG START ===")
print("MODEL_DIR:", MODEL_DIR)
print("FILES:", os.listdir(MODEL_DIR) if os.path.exists(MODEL_DIR) else "NO DIR")
print("GENDER_MODEL_PATH:", GENDER_MODEL_PATH)
print("GROWTH_MODEL_PATH:", GROWTH_MODEL_PATH)
print("Gender exists:", os.path.exists(GENDER_MODEL_PATH))
print("Growth exists:", os.path.exists(GROWTH_MODEL_PATH))
print("=== MODEL DEBUG END ===")


# =========================================================
# IMAGE PREPROCESSING
# SAME SA TRAINING SAMPLE MO:
# resize -> /255.0 -> expand dims
# =========================================================
def preprocess_image_from_bytes(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# =========================================================
# MAIN CLASSIFIER
# =========================================================
def classify_crayfish(img_bytes):
    try:
        img_array = preprocess_image_from_bytes(img_bytes)

        # =========================================
        # GENDER MODEL - LOAD, PREDICT, RELEASE
        # =========================================
        print("[MODEL] Loading gender model...", flush=True)

        gender_model = load_model(
            GENDER_MODEL_PATH,
            compile=False
        )

        gender_prediction = gender_model.predict(
            img_array,
            verbose=0
        )

        gender_raw_value = float(
            gender_prediction[0][0]
        )

        if gender_raw_value > 0.5:
            gender_label = "Male"
            gender_confidence = (
                gender_raw_value * 100.0
            )
        else:
            gender_label = "Female"
            gender_confidence = (
                (1.0 - gender_raw_value) * 100.0
            )

        del gender_model
        tf.keras.backend.clear_session()

        print("[MODEL] Gender model released", flush=True)


        # =========================================
        # GROWTH MODEL - LOAD, PREDICT, RELEASE
        # =========================================
        print("[MODEL] Loading growth model...", flush=True)

        growth_model = load_model(
            GROWTH_MODEL_PATH,
            compile=False
        )

        growth_prediction = growth_model.predict(
            img_array,
            verbose=0
        )

        growth_class_index = int(
            np.argmax(growth_prediction[0])
        )

        growth_label = GROWTH_LABELS[
            growth_class_index
        ]

        growth_confidence = (
            float(
                growth_prediction[0][
                    growth_class_index
                ]
            )
            * 100.0
        )

        growth_raw_probs = [
            round(float(x), 6)
            for x in growth_prediction[0].tolist()
        ]

        del growth_model
        tf.keras.backend.clear_session()

        print("[MODEL] Growth model released", flush=True)


        # =========================================
        # LENGTH / STAGE ESTIMATE
        # =========================================
        extra_growth = (
            estimate_length_and_growth_stage(
                img_bytes
            )
        )


        return {
            "gender": gender_label,

            "gender_confidence": round(
                gender_confidence,
                2
            ),

            "gender_raw_value": round(
                gender_raw_value,
                6
            ),

            "growth": growth_label,

            "growth_confidence": round(
                growth_confidence,
                2
            ),

            "growth_raw_probs": (
                growth_raw_probs
            ),

            "estimated_length_cm": (
                extra_growth.get(
                    "estimated_length_cm",
                    "Unknown"
                )
            ),

            "estimated_growth_stage": (
                extra_growth.get(
                    "growth_stage",
                    "Unknown"
                )
            ),

            "growth_debug": (
                extra_growth.get(
                    "debug",
                    ""
                )
            )
        }

    except Exception as error:
        traceback.print_exc()

        tf.keras.backend.clear_session()

        return {
            "error":
                f"Classification failed: {str(error)}"
        }

# =========================================================
# CLASSIFY ROUTE
# existing flow mo, inayos lang for complete model use
# =========================================================
@csrf.exempt
@app.route('/classify', methods=['GET', 'POST'])
@login_required
def classify():
    print("CLASSIFY PAGE LOADED")

    result = None
    uploaded_file_url = None
    user_id = session.get("user_id")
    
    preset_batch_id = request.args.get(
    "batch_id",
    type=int
    )

    preset_mode = request.args.get(
        "mode",
        "quick"
    )

    # Existing batches created by the logged-in user.
    batches = (
        Batch.query
        .filter_by(user_id=user_id)
        .order_by(Batch.created_date.asc())
        .all()
    )

    # Only put actual evaluated model accuracy here.
    gender_model_accuracy = None
    growth_model_accuracy = None

    def render_classify_page():
        return render_template(
            'dashboard/classify.html',
            result=result,
            uploaded_file_url=uploaded_file_url,
            gender_model_accuracy=gender_model_accuracy,
            growth_model_accuracy=growth_model_accuracy,
            batches=batches,
            preset_batch_id=preset_batch_id,
            preset_mode=preset_mode
        )

    if request.method == 'POST':

        # =====================================================
        # CLASSIFICATION MODE
        #
        # quick = normal classification only
        # batch = classification + batch growth tracking
        # =====================================================
        tracking_mode = request.form.get(
            "tracking_mode",
            "quick"
        )

        batch_id = request.form.get(
            "batch_id",
            type=int
        )

        selected_batch = None
        latest_batch_log = None
        observation_number = None
        week_number = None
        monitoring_period = None
        sample_number = None

        # =====================================================
        # BATCH TRACKING
        # Batch is required ONLY when batch mode is selected.
        # =====================================================
        if tracking_mode == "batch":

            if not batch_id:
                flash(
                    "Please choose an existing batch or create a new batch first.",
                    "danger"
                )
                return render_classify_page()

            selected_batch = (
                Batch.query
                .filter_by(
                    id=batch_id,
                    user_id=user_id
                )
                .first()
            )

            if not selected_batch:
                flash(
                    "Invalid batch selected.",
                    "danger"
                )
                return render_classify_page()

            existing_batch_observations = (
                ClassificationLog.query
                .filter_by(
                    user_id=user_id,
                    batch_id=selected_batch.id
                )
                .order_by(
                    ClassificationLog.created_at.asc()
                )
                .all()
            )

            latest_batch_log = (
                existing_batch_observations[-1]
                if existing_batch_observations
                else None
            )

            batch_observation_count = len(existing_batch_observations)

            observation_number = batch_observation_count + 1

            # Keep these fields for compatibility with the current template.
            monitoring_period = observation_number
            sample_number = observation_number
            week_number = observation_number

        # =====================================================
        # GET IMAGE
        # =====================================================
        file = request.files.get('file')

        if not file or not file.filename:
            flash(
                'No file selected.',
                'danger'
            )
            return render_classify_page()

        original_filename = secure_filename(
            file.filename
        )

        if not original_filename:
            flash(
                'Invalid file name.',
                'danger'
            )
            return render_classify_page()

        allowed_extensions = {
            'jpg',
            'jpeg',
            'png'
        }

        if '.' not in original_filename:
            flash(
                'Invalid image file. Please upload JPG, JPEG, or PNG.',
                'danger'
            )
            return render_classify_page()

        file_extension = (
            original_filename
            .rsplit('.', 1)[1]
            .lower()
        )

        if file_extension not in allowed_extensions:
            flash(
                'Invalid image format. Please upload JPG, JPEG, or PNG.',
                'danger'
            )
            return render_classify_page()

        img_bytes = file.read()

        if not img_bytes:
            flash(
                'The uploaded image is empty or invalid.',
                'danger'
            )
            return render_classify_page()

        # =====================================================
        # VERIFY IMAGE
        # =====================================================
        try:
            image = Image.open(
                io.BytesIO(img_bytes)
            )

            image.verify()

        except Exception:
            flash(
                'The uploaded file is not a valid image.',
                'danger'
            )
            return render_classify_page()

        # =====================================================
        # SAVE UPLOADED IMAGE
        # =====================================================
        try:
            unique_filename = (
                f"{uuid.uuid4().hex}_"
                f"{original_filename}"
            )

            upload_folder = os.path.join(
                app.root_path,
                'static',
                'uploads'
            )

            os.makedirs(
                upload_folder,
                exist_ok=True
            )

            save_path = os.path.join(
                upload_folder,
                unique_filename
            )

            with open(
                save_path,
                'wb'
            ) as uploaded_image:

                uploaded_image.write(
                    img_bytes
                )

            uploaded_file_url = url_for(
                'static',
                filename=f'uploads/{unique_filename}'
            )

        except Exception as error:

            print(
                "[ERROR] Image upload failed:",
                error
            )

            flash(
                f'Image upload failed: {str(error)}',
                'danger'
            )

            return render_classify_page()

        # =====================================================
        # RUN ML CLASSIFICATION
        # =====================================================
        try:

            print(
                "=== ROUTE CALLED classify ==="
            )

            result = classify_crayfish(
                img_bytes
            )

            print(
                "=== ROUTE RESULT ===",
                result
            )

        except Exception as error:

            traceback.print_exc()

            flash(
                f'Classification failed: {str(error)}',
                'danger'
            )

            result = None

            return render_classify_page()

        # =====================================================
        # VALIDATE MODEL RESULT
        # =====================================================
        if not isinstance(result, dict):

            flash(
                'The classification model returned an invalid result.',
                'danger'
            )

            result = None

            return render_classify_page()

        if result.get('error'):

            flash(
                result['error'],
                'danger'
            )

            result = None

            return render_classify_page()

        # =====================================================
        # EXTRACT MODEL RESULTS
        # =====================================================
        gender_result = result.get(
            'gender',
            'Unknown'
        )

        growth_result = result.get(
            'growth',
            'Unknown'
        )

        gender_confidence = result.get(
            'gender_confidence'
        )

        growth_confidence = result.get(
            'growth_confidence'
        )

        # =====================================================
        # SAFE FLOAT CONVERSION
        # =====================================================
        try:
            gender_confidence = float(
                gender_confidence
            )

        except (TypeError, ValueError):
            gender_confidence = 0.0

        try:
            growth_confidence = float(
                growth_confidence
            )

        except (TypeError, ValueError):
            growth_confidence = 0.0

        gender_confidence = max(
            0.0,
            min(
                100.0,
                gender_confidence
            )
        )

        growth_confidence = max(
            0.0,
            min(
                100.0,
                growth_confidence
            )
        )

        result['gender'] = gender_result
        result['growth'] = growth_result

        result['gender_confidence'] = round(
            gender_confidence,
            2
        )

        result['growth_confidence'] = round(
            growth_confidence,
            2
        )

        # =====================================================
        # ESTIMATED LENGTH / GROWTH STAGE
        # =====================================================
        result['estimated_length_cm'] = (
            result.get(
                'estimated_length_cm',
                'Unknown'
            )
        )

        result['estimated_growth_stage'] = (
            result.get(
                'estimated_growth_stage',
                'Unknown'
            )
        )

        estimated_length_raw = (
            result.get(
                'estimated_length_cm'
            )
        )

        estimated_length_value = None

        if estimated_length_raw:

            try:
                estimated_length_value = float(
                    str(
                        estimated_length_raw
                    )
                    .replace(
                        "cm",
                        ""
                    )
                    .strip()
                )

            except (
                TypeError,
                ValueError
            ):
                estimated_length_value = None

        # =====================================================
        # 14-DAY BATCH GROWTH COMPARISON
        # =====================================================
        previous_length = None
        length_change = None
        elapsed_days = None
        growth_tracking_status = None
        previous_growth_stage = None
        next_check_date = None
        days_remaining = None

        if selected_batch:
            current_time_for_tracking = datetime.utcnow()

            existing_batch_observations = (
                ClassificationLog.query
                .filter_by(
                    user_id=user_id,
                    batch_id=selected_batch.id
                )
                .order_by(
                    ClassificationLog.created_at.asc()
                )
                .all()
            )

            # FIRST UPLOAD = BASELINE
            if not existing_batch_observations:

                growth_tracking_status = "BASELINE RECORDED"

                next_check_date = (
                    current_time_for_tracking
                    + timedelta(
                        days=GROWTH_CHECK_INTERVAL_DAYS
                    )
                )

                days_remaining = (
                    GROWTH_CHECK_INTERVAL_DAYS
                )

            else:
                # Build the official timeline using ONLY the
                # already-saved observations.
                official_references = (
                    _get_official_growth_references(
                        existing_batch_observations
                    )
                )

                last_official_reference = (
                    official_references[-1]
                    if official_references
                    else existing_batch_observations[0]
                )

                elapsed_days = (
                    _days_between(
                        last_official_reference.created_at,
                        current_time_for_tracking
                    )
                    or 0
                )

                # ---------------------------------------------
                # NOT YET 14 DAYS:
                # Save this upload to history only.
                # ---------------------------------------------
                if elapsed_days < GROWTH_CHECK_INTERVAL_DAYS:

                    days_remaining = max(
                        0,
                        GROWTH_CHECK_INTERVAL_DAYS
                        - elapsed_days
                    )

                    next_check_date = (
                        last_official_reference.created_at
                        + timedelta(
                            days=GROWTH_CHECK_INTERVAL_DAYS
                        )
                        if last_official_reference.created_at
                        else None
                    )

                    growth_tracking_status = (
                        f"WAITING FOR 2-WEEK CHECK "
                        f"({days_remaining} DAY(S) REMAINING)"
                    )

                # ---------------------------------------------
                # 14 DAYS OR MORE:
                # Compare this new upload against the last
                # official reference.
                # ---------------------------------------------
                else:

                    previous_length = (
                        last_official_reference
                        .estimated_length_cm
                    )

                    previous_growth_stage = (
                        last_official_reference.growth
                    )

                    if (
                        previous_length is not None
                        and estimated_length_value is not None
                    ):

                        length_change = (
                            estimated_length_value
                            - previous_length
                        )

                        if length_change > 0.05:
                            growth_tracking_status = (
                                "GROWTH DETECTED"
                            )

                        elif length_change < -0.05:
                            growth_tracking_status = (
                                "SIZE DECREASE DETECTED"
                            )

                        else:
                            growth_tracking_status = (
                                "NO SIZE CHANGE"
                            )

                    else:
                        growth_tracking_status = (
                            "2-WEEK CHECK RECORDED"
                        )

                    next_check_date = (
                        current_time_for_tracking
                        + timedelta(
                            days=GROWTH_CHECK_INTERVAL_DAYS
                        )
                    )

                    days_remaining = (
                        GROWTH_CHECK_INTERVAL_DAYS
                    )

        # =====================================================
        # SEND TRACKING INFORMATION TO HTML
        # =====================================================
        result['tracking_mode'] = (
            tracking_mode
        )

        result['batch_id'] = (
            selected_batch.id
            if selected_batch
            else None
        )

        result['batch_name'] = (
            selected_batch.batch_name
            if selected_batch
            else None
        )

        result['observation_number'] = (
            observation_number
            if selected_batch
            else None
        )

        result['monitoring_period'] = (
            monitoring_period if selected_batch else None
        )
        result['sample_number'] = (
            sample_number if selected_batch else None
        )
        result['sample_target'] = (
            SAMPLE_TARGET if selected_batch else None
        )
        result['period_complete'] = bool(
            selected_batch
            and growth_tracking_status in {
                "GROWTH DETECTED",
                "SIZE DECREASE DETECTED",
                "NO SIZE CHANGE",
                "2-WEEK CHECK RECORDED"
            }
        )

        result['previous_length_cm'] = (
            previous_length
        )

        result['length_change_cm'] = (
            round(
                length_change,
                2
            )
            if length_change is not None
            else None
        )

        result['elapsed_days'] = (
            elapsed_days
        )

        result['growth_tracking_status'] = (
            growth_tracking_status
        )

        result['previous_growth_stage'] = (
            previous_growth_stage
        )

        result['next_check_date'] = (
            next_check_date.strftime("%b %d, %Y")
            if next_check_date
            else None
        )

        result['days_remaining'] = (
            days_remaining
        )

        # =====================================================
        # SAVE CLASSIFICATION TO DATABASE
        #
        # QUICK:
        #   batch_id = NULL
        #
        # BATCH:
        #   batch_id = selected batch
        #   week_number = observation number
        # =====================================================
        try:

            current_time = datetime.utcnow()

            classification_log = ClassificationLog(

                user_id=user_id,

                batch_id=(
                    selected_batch.id
                    if selected_batch
                    else None
                ),

                week_number=(
                    week_number
                    if selected_batch
                    else None
                ),

                image_filename=(
                    unique_filename
                ),

                gender=(
                    gender_result
                ),

                growth=(
                    growth_result
                ),

                gender_confidence=(
                    gender_confidence
                ),

                growth_confidence=(
                    growth_confidence
                ),

                estimated_length_cm=(
                    estimated_length_value
                ),

                created_at=(
                    current_time
                )
            )

            # =================================================
            # ACTIVITY LOG MESSAGE
            # =================================================
            if selected_batch:

                activity_text = (
                    f"Batch "
                    f"{selected_batch.batch_name} "
                    f"Observation "
                    f"{observation_number}: "
                    f"{growth_result} growth "
                    f"({growth_confidence:.2f}% confidence)"
                )

            else:

                activity_text = (
                    f"Quick classification: "
                    f"{gender_result} "
                    f"({gender_confidence:.2f}%) / "
                    f"{growth_result} "
                    f"({growth_confidence:.2f}%)"
                )

            activity_log = ActivityLog(
                user_id=user_id,
                action=activity_text,
                timestamp=current_time
            )

            db.session.add(
                classification_log
            )

            db.session.add(
                activity_log
            )

            db.session.commit()

            if selected_batch:

                socketio.emit(
                    "batch_growth_update",
                    {
                        "batch_id": selected_batch.id,
                        "monitoring_period": monitoring_period,
                        "sample_number": sample_number,
                        "sample_target": SAMPLE_TARGET,
                        "period_complete": (
                            growth_tracking_status in {
                                "GROWTH DETECTED",
                                "SIZE DECREASE DETECTED",
                                "NO SIZE CHANGE",
                                "2-WEEK CHECK RECORDED"
                            }
                        )
                    },
                    room=f"user_{user_id}"
                )

                print(
                    "[OK] Batch observation saved:",
                    selected_batch.batch_name,
                    "Observation",
                    observation_number
                )

                flash(
                    f'Observation {observation_number} was successfully added to '
                    f'{selected_batch.batch_name}.',
                    'success'
                )

                return redirect(
                    url_for(
                        'batch_detail',
                        batch_id=selected_batch.id
                    )
                )

            else:

                print(
                    "[OK] Quick classification saved"
                )

        except Exception as error:

            db.session.rollback()

            print(
                "[ERROR] Failed to save classification result:",
                error
            )

            flash(
                'Classification succeeded, but the result could not be saved to the database.',
                'warning'
            )

    return render_classify_page()

def verify_email_change(token):

    try:
        new_email = serializer.loads(token, salt='email-change', max_age=3600)
    except:
        flash("Verification link expired or invalid.", "danger")
        return redirect(url_for("settings"))

    user_id = session.get('user_id')
    user = User.query.get(user_id)

    if not user:
        flash("User not found.", "danger")
        return redirect(url_for("login"))

    user.email = new_email
    db.session.commit()

    flash("Email successfully verified and updated!", "success")
    return redirect(url_for("settings"))

@app.route("/test-email")
def test_email():

    msg = Message(
        subject="Test Email - Craylife",
        recipients=["micahavrill14@gmail.com"],
        body="This is a test email from Craylife system."
    )

    mail.send(msg)

    return "Email sent successfully!"

@app.route("/reset-password/<token>", methods=["GET", "POST"])
def reset_password_token(token):

    try:
        email = serializer.loads(token, salt="password-reset", max_age=3600)
    except:
        flash("Reset link expired.", "danger")
        return redirect(url_for("password_reset"))

    user = User.query.filter_by(email=email).first()

    if request.method == "POST":

        new_password = request.form.get("password")

        user.set_password(new_password)
        db.session.commit()

        flash("Password successfully reset.", "success")
        return redirect(url_for("login"))

    return render_template("auth/reset_password.html")

@app.route(
    "/batch/<int:batch_id>/growth/add",
    methods=["GET", "POST"]
)
@login_required
def add_batch_growth(batch_id):

    batch = Batch.query.get_or_404(batch_id)

    if batch.user_id != session.get("user_id"):
        flash(
            "Unauthorized batch access.",
            "danger"
        )
        return redirect(
            url_for("home")
        )

    if request.method == "POST":

        week_number = request.form.get(
            "week_number",
            type=int
        )

        average_length_cm = request.form.get(
            "average_length_cm",
            type=float
        )

        survivor_count = request.form.get(
            "survivor_count",
            type=int
        )

        normal_growth_count = request.form.get(
            "normal_growth_count",
            type=int,
            default=0
        )

        slow_growth_count = request.form.get(
            "slow_growth_count",
            type=int,
            default=0
        )

        growth_stage = request.form.get(
            "growth_stage"
        )

        notes = request.form.get(
            "notes"
        )

        if normal_growth_count > slow_growth_count:
            growth_status = "NORMAL"

        elif slow_growth_count > normal_growth_count:
            growth_status = "SLOW"

        else:
            growth_status = "STABLE"

        record = BatchGrowthRecord(
            batch_id=batch.id,
            week_number=week_number,
            average_length_cm=average_length_cm,
            survivor_count=survivor_count,
            normal_growth_count=normal_growth_count,
            slow_growth_count=slow_growth_count,
            growth_stage=growth_stage,
            growth_status=growth_status,
            notes=notes
        )

        db.session.add(record)
        db.session.commit()

        flash(
            "Weekly growth record saved.",
            "success"
        )

        return redirect(
            url_for("home")
        )

    return render_template(
        "dashboard/growth_record_form.html",
        batch=batch
    )
    

# =========================================================
# BATCH OBSERVATION CRUD
# =========================================================

@app.route(
    '/batch/<int:batch_id>/observation/<int:observation_id>/edit',
    methods=['GET', 'POST']
)
@login_required
def edit_batch_observation(batch_id, observation_id):
    user_id = session.get('user_id')

    batch = (
        Batch.query
        .filter_by(
            id=batch_id,
            user_id=user_id
        )
        .first_or_404()
    )

    observation = (
        ClassificationLog.query
        .filter_by(
            id=observation_id,
            batch_id=batch.id,
            user_id=user_id
        )
        .first_or_404()
    )

    if request.method == 'POST':
        growth = (
            request.form.get('growth', '')
            .strip()
            .lower()
        )

        estimated_length_cm = request.form.get(
            'estimated_length_cm',
            type=float
        )

        allowed_growth_stages = {
            'mid_juvenile',
            'juvenile',
            'mid_adult',
            'sub_adult',
            'adult'
        }

        if growth not in allowed_growth_stages:
            flash(
                'Please choose a valid growth stage.',
                'danger'
            )
            return render_template(
                'dashboard/edit_observation.html',
                batch=batch,
                observation=observation
            )

        if (
            estimated_length_cm is not None
            and estimated_length_cm < 0
        ):
            flash(
                'Estimated length cannot be negative.',
                'danger'
            )
            return render_template(
                'dashboard/edit_observation.html',
                batch=batch,
                observation=observation
            )

        observation.growth = growth
        observation.estimated_length_cm = (
            estimated_length_cm
        )

        db.session.add(
            ActivityLog(
                user_id=user_id,
                action=(
                    f'Updated observation {observation.id} '
                    f'for batch "{batch.batch_name}"'
                ),
                timestamp=datetime.utcnow()
            )
        )

        db.session.commit()

        flash(
            'Observation updated successfully.',
            'success'
        )

        return redirect(
            url_for(
                'batch_detail',
                batch_id=batch.id
            )
        )

    return render_template(
        'dashboard/edit_observation.html',
        batch=batch,
        observation=observation
    )


@app.route(
    '/batch/<int:batch_id>/observation/<int:observation_id>/delete',
    methods=['POST']
)
@login_required
def delete_batch_observation(batch_id, observation_id):
    user_id = session.get('user_id')

    batch = (
        Batch.query
        .filter_by(
            id=batch_id,
            user_id=user_id
        )
        .first_or_404()
    )

    observation = (
        ClassificationLog.query
        .filter_by(
            id=observation_id,
            batch_id=batch.id,
            user_id=user_id
        )
        .first_or_404()
    )

    image_filename = observation.image_filename

    try:
        db.session.delete(observation)
        db.session.flush()

        # Keep week_number / observation order clean after deletion.
        remaining_observations = (
            ClassificationLog.query
            .filter_by(
                user_id=user_id,
                batch_id=batch.id
            )
            .order_by(
                ClassificationLog.created_at.asc()
            )
            .all()
        )

        for index, item in enumerate(
            remaining_observations,
            start=1
        ):
            item.week_number = index

        db.session.add(
            ActivityLog(
                user_id=user_id,
                action=(
                    f'Deleted an observation from '
                    f'batch "{batch.batch_name}"'
                ),
                timestamp=datetime.utcnow()
            )
        )

        db.session.commit()

        # Remove only the classifier-upload copy, if it exists.
        # Batch reference images are stored in another folder
        # and are not touched here.
        if image_filename:
            image_path = os.path.join(
                app.root_path,
                'static',
                'uploads',
                image_filename
            )

            if os.path.isfile(image_path):
                try:
                    os.remove(image_path)
                except OSError as file_error:
                    print(
                        '[WARNING] Could not remove observation image:',
                        file_error
                    )

        flash(
            'Observation deleted successfully. '
            'Growth tracking was recalculated automatically.',
            'success'
        )

    except Exception as error:
        db.session.rollback()

        print(
            '[ERROR] Failed to delete observation:',
            error
        )

        flash(
            'Observation could not be deleted.',
            'danger'
        )

    return redirect(
        url_for(
            'batch_detail',
            batch_id=batch.id
        )
    )


@app.route('/batch/<int:batch_id>')
@login_required
def batch_detail(batch_id):
    user_id = session.get('user_id')

    batch = (
        Batch.query
        .filter_by(
            id=batch_id,
            user_id=user_id
        )
        .first_or_404()
    )

    observations = (
        ClassificationLog.query
        .filter_by(
            user_id=user_id,
            batch_id=batch.id
        )
        .order_by(
            ClassificationLog.created_at.asc()
        )
        .all()
    )

    first_observation = (
        observations[0]
        if observations
        else None
    )

    latest_observation = (
        observations[-1]
        if observations
        else None
    )

    total_growth_cm = None
    elapsed_days = None
    latest_change_cm = None
    latest_tracking_status = "NO OBSERVATIONS"

    next_check_date = None
    days_remaining = None
    previous_eligible_observation = None

    # =====================================================
    # 14-DAY GROWTH TRACKING
    # =====================================================

    if first_observation and latest_observation:

        if (
            first_observation.created_at
            and latest_observation.created_at
        ):
            elapsed_days = _days_between(
                first_observation.created_at,
                latest_observation.created_at
            )

        # -----------------------------------------
        # FIRST OBSERVATION = BASELINE
        # -----------------------------------------
        if len(observations) == 1:

            latest_tracking_status = (
                "BASELINE RECORDED"
            )

            days_remaining = (
                GROWTH_CHECK_INTERVAL_DAYS
            )

            if first_observation.created_at:

                next_check_date = (
                    first_observation.created_at
                    + timedelta(
                        days=GROWTH_CHECK_INTERVAL_DAYS
                    )
                )

        # -----------------------------------------
        # SECOND / FUTURE OBSERVATIONS
        # -----------------------------------------
        else:

            reference_time = (
                latest_observation.created_at
                or datetime.utcnow()
            )

            # Build official references only from observations
            # BEFORE the latest upload.
            previous_observations = observations[:-1]

            official_references = (
                _get_official_growth_references(
                    previous_observations
                )
            )

            last_official_reference = (
                official_references[-1]
                if official_references
                else first_observation
            )

            elapsed_from_reference = (
                _days_between(
                    last_official_reference.created_at,
                    reference_time
                )
                or 0
            )

            # -------------------------------------
            # LESS THAN 14 DAYS
            # -------------------------------------
            if elapsed_from_reference < GROWTH_CHECK_INTERVAL_DAYS:

                days_remaining = max(
                    0,
                    GROWTH_CHECK_INTERVAL_DAYS
                    - elapsed_from_reference
                )

                if last_official_reference.created_at:

                    next_check_date = (
                        last_official_reference.created_at
                        + timedelta(
                            days=GROWTH_CHECK_INTERVAL_DAYS
                        )
                    )

                latest_tracking_status = (
                    f"WAITING FOR 2-WEEK CHECK "
                    f"({days_remaining} DAY(S) REMAINING)"
                )

            # -------------------------------------
            # 14 DAYS OR MORE = OFFICIAL CHECK
            # -------------------------------------
            else:

                previous_eligible_observation = (
                    last_official_reference
                )

                previous_length = (
                    previous_eligible_observation
                    .estimated_length_cm
                )

                current_length = (
                    latest_observation
                    .estimated_length_cm
                )

                if (
                    previous_length is not None
                    and current_length is not None
                ):

                    latest_change_cm = round(
                        current_length
                        - previous_length,
                        2
                    )

                    if latest_change_cm > 0.05:

                        latest_tracking_status = (
                            "GROWTH DETECTED"
                        )

                    elif latest_change_cm < -0.05:

                        latest_tracking_status = (
                            "SIZE DECREASE DETECTED"
                        )

                    else:

                        latest_tracking_status = (
                            "NO SIZE CHANGE"
                        )

                else:

                    latest_tracking_status = (
                        "2-WEEK CHECK RECORDED"
                    )

                if latest_observation.created_at:

                    next_check_date = (
                        latest_observation.created_at
                        + timedelta(
                            days=GROWTH_CHECK_INTERVAL_DAYS
                        )
                    )

                days_remaining = (
                    GROWTH_CHECK_INTERVAL_DAYS
                )

        # -----------------------------------------
        # TOTAL CHANGE FROM FIRST TO LATEST
        # -----------------------------------------
        if (
            first_observation.estimated_length_cm
            is not None
            and
            latest_observation.estimated_length_cm
            is not None
        ):

            total_growth_cm = round(
                latest_observation.estimated_length_cm
                - first_observation.estimated_length_cm,
                2
            )

    # =====================================================
    # BATCH REFERENCE IMAGES
    # =====================================================

    reference_images = []

    batch_folder = os.path.join(
        app.root_path,
        'static',
        'uploads',
        'batches',
        str(batch.id)
    )

    allowed_extensions = {
        '.jpg',
        '.jpeg',
        '.png'
    }

    if os.path.isdir(batch_folder):

        for filename in sorted(
            os.listdir(batch_folder)
        ):

            extension = os.path.splitext(
                filename
            )[1].lower()

            if extension in allowed_extensions:

                reference_images.append(
                    url_for(
                        'static',
                        filename=(
                            f'uploads/batches/'
                            f'{batch.id}/'
                            f'{filename}'
                        )
                    )
                )

    # =====================================================
    # SEND TO BATCH DETAIL HTML
    # =====================================================

    return render_template(
        'dashboard/batch_detail.html',

        batch=batch,

        observations=observations,

        first_observation=first_observation,

        latest_observation=latest_observation,

        total_growth_cm=total_growth_cm,

        elapsed_days=elapsed_days,

        latest_change_cm=latest_change_cm,

        latest_tracking_status=(
            latest_tracking_status
        ),

        reference_images=reference_images,

        growth_check_interval_days=(
            GROWTH_CHECK_INTERVAL_DAYS
        ),

        next_check_date=next_check_date,

        days_remaining=days_remaining,

        previous_eligible_observation=(
            previous_eligible_observation
        )
    )
