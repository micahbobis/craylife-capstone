from flask import render_template, redirect, url_for, flash, request, session
from app import app, db, csrf, mail, serializer
from app.forms import LoginForm, RegistrationForm, BatchForm, RequestResetForm, ResetPasswordForm
from app.models import User, Batch, Inventory, Sale, ActivityLog, ClassificationLog
from functools import wraps
from flask_mail import Message
import os
import uuid
import io
from datetime import datetime
import tensorflow as tf
import traceback
import json
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
from app.utils import get_arduino_data
from werkzeug.security import check_password_hash, generate_password_hash
from flask import jsonify
from app.db_save import save_classification_log
from app.models import ClassificationLog
from app import app


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

@app.route("/arduino-data")
def arduino_data():
    return jsonify(get_arduino_data())

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in first', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function


@app.route("/esp32-data", methods=["POST"])
@csrf.exempt
def sensor_update_api():
    try:
        data = request.get_json(silent=True) or {}

        ph = data.get("ph") or data.get("pH Level")
        water_level = data.get("water_level") or data.get("Water Level")
        water_quality = data.get("water_quality") or data.get("Water Quality")
        turbidity_ntu = data.get("turbidity_ntu") or data.get("Turbidity NTU")

        payload = {
            "pH Level": str(ph) if ph is not None else "—",
            "pH Voltage": str(data.get("pH Voltage", "—")),
            "Water Level": str(water_level).upper() if water_level is not None else "—",
            "Water Quality": str(water_quality).upper() if water_quality is not None else "—",
            "Turbidity NTU": str(turbidity_ntu) if turbidity_ntu is not None else "—",
            "Turbidity Voltage": str(data.get("Turbidity Voltage", "—")),
            "Turbidity ADC": str(data.get("Turbidity ADC", "—")),
            "Status": "Connected"
        }

        json_path = os.path.abspath(
            os.path.join(app.root_path, "..", "arduino_data.json")
        )

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

    batches = (
        Batch.query.filter_by(user_id=user_id).all()
        if user_id
        else []
    )

    sensor_data_raw = get_arduino_data() or {}

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
    ]

    # Growth monitoring
    growth_logs = []
    latest_growth = None
    growth_chart_labels = []
    growth_chart_values = []

    if user_id:
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
                "adult": 4,
            }

            for log in reversed(growth_logs):
                if log.created_at:
                    label = log.created_at.strftime(
                        "%b %d %H:%M"
                    )
                else:
                    label = "Unknown"

                growth_chart_labels.append(label)

                growth_key = (log.growth or "").lower()
                growth_chart_values.append(
                    stage_order.get(growth_key, 0)
                )

    return render_template(
        'dashboard/home.html',
        batches=batches,
        sensor_data=sensor_data,
        latest_growth=latest_growth,
        growth_logs=growth_logs,
        growth_chart_labels=growth_chart_labels,
        growth_chart_values=growth_chart_values
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

    
    default_settings = {
        'setting1': 'no',
        'setting2': 0,
        'setting3': 'no',
        'setting4': 30
    }

    if request.method == 'POST':

        
        if 'username' in request.form and 'email' in request.form:

            user.username = request.form['username']
            new_email = request.form['email']

            
            if new_email != user.email:

                token = serializer.dumps(new_email, salt='email-change')

                verify_url = url_for(
                    'verify_email_change',
                    token=token,
                    _external=True
                )

                msg = Message(
                    subject="Verify your new email - Craylife",
                    recipients=["micahavrill14@gmail.com"],  # fixed email receiver
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

                flash('Verification email sent to micahavrill14@gmail.com.', 'info')

            else:
                flash('User info updated successfully!', 'success')

            db.session.commit()

        
        if 'current_password' in request.form and 'new_password' in request.form:

            current_password = request.form['current_password']
            new_password = request.form['new_password']
            confirm_password = request.form['confirm_password']

            if not user.check_password(current_password):
                flash('Current password is incorrect.', 'danger')

            elif new_password != confirm_password:
                flash('New passwords do not match.', 'danger')

            else:
                user.set_password(new_password)
                db.session.commit()

                msg = Message(
                    subject="Password changed - Craylife",
                    recipients=["micahavrill14@gmail.com"],
                    body=f"""
Hello {user.username},

Your Craylife password was successfully changed.

If this was not you, please reset your password immediately.
"""
                )

                mail.send(msg)

                flash('Password updated successfully! Email notification sent.', 'success')

        
        default_settings['setting1'] = request.form.get('setting1', default_settings['setting1'])
        default_settings['setting2'] = request.form.get('setting2', default_settings['setting2'])
        default_settings['setting3'] = request.form.get('setting3', default_settings['setting3'])
        default_settings['setting4'] = request.form.get('setting4', default_settings['setting4'])

        flash('System settings updated!', 'success')

        return redirect(url_for('settings'))

    return render_template('dashboard/settings.html', user=user, settings=default_settings)
 
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
        pdf.drawString(50, y, f"{act.action} ({act.timestamp})")
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

    total_batches = Batch.query.filter_by(user_id=user_id).count()

    batches = Batch.query.filter_by(user_id=user_id).all()
    inventory_items = []
    for batch in batches:
        inv = Inventory.query.filter_by(batch_id=batch.id).first()
        inventory_items.append({
            'id': batch.id,
            'name': batch.batch_name,
            'quantity': inv.total_count if inv else batch.quantity
        })

    activities = ActivityLog.query.order_by(ActivityLog.timestamp.desc()).limit(5).all()

    reports = []
    for item in inventory_items:
        reports.append({
            'id': item['id'],
            'type': item['name'],
            'count': item['quantity'],
        })

    classification_logs = ClassificationLog.query.order_by(
        ClassificationLog.created_at.desc()
    ).all()

    return render_template(
        'dashboard/reports.html',
        total_batches=total_batches,
        inventory_items=inventory_items,
        activities=activities,
        reports=reports,
        classification_logs=classification_logs
    )
 

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    print(f"Form validated: {form.validate_on_submit()}")
    print(f"Form errors: {form.errors}")

    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        print(f"User found: {user}")
        if user:
            print(f"Password check: {user.check_password(form.password.data)}")
        if user and user.check_password(form.password.data):
            session['user_id'] = user.id
            session['username'] = user.username
            flash('Login successful!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid username or password', 'danger')

    return render_template('auth/login.html', form=form)


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
    form = BatchForm()
    if form.validate_on_submit():
        batch = Batch(
            batch_name=form.batch_name.data,
            user_id=session.get('user_id'),
            quantity=int(form.quantity.data or 0)
        )
        db.session.add(batch)
        db.session.commit()

        # handle optional file upload
        file = form.batch_file.data
        if file:
            uploads_dir = os.path.join(app.root_path, 'static', 'uploads')
            os.makedirs(uploads_dir, exist_ok=True)
            filename = secure_filename(file.filename)
            file_path = os.path.join(uploads_dir, filename)
            file.save(file_path)

        flash('Batch created successfully!', 'success')
        return redirect(url_for('inventory'))

    return render_template('dashboard/batch_form.html', form=form)


@app.route('/inventory')
@login_required
def inventory():
    user_id = session.get('user_id')
    batches = Batch.query.filter_by(user_id=user_id).all()

    inventory_items = []
    for batch in batches:
        inv = Inventory.query.filter_by(batch_id=batch.id).first()
        inventory_items.append({
            'id': batch.id,
            'name': batch.batch_name,
            'quantity': inv.total_count if inv else batch.quantity
        })

    return render_template('dashboard/inventory.html', inventory_items=inventory_items)


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

gender_model = None
growth_model = None

try:
    gender_model = load_model(GENDER_MODEL_PATH, compile=False)
    print("[OK] Gender model loaded")
except Exception as e:
    print("[ERROR] Gender:", repr(e))

try:
    growth_model = load_model(GROWTH_MODEL_PATH, compile=False)
    print("[OK] Growth model loaded")
except Exception as e:
    print("[ERROR] Growth:", repr(e))
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
        if gender_model is None:
            return {"error": f"Gender model failed to load: {GENDER_MODEL_PATH}"}

        if growth_model is None:
            return {"error": f"Growth model failed to load: {GROWTH_MODEL_PATH}"}

        img_array = preprocess_image_from_bytes(img_bytes)

        # =========================
        # GENDER PREDICTION
        # binary threshold based sa sample mo:
        # if prediction[0][0] > 0.5 => Male
        # else => Female
        # =========================
        gender_prediction = gender_model.predict(img_array, verbose=0)
        gender_raw_value = float(gender_prediction[0][0])

        if gender_raw_value > 0.5:
         gender_label = GENDER_LABELS[1]
         gender_confidence = gender_raw_value * 100.0
        else:
         gender_label = GENDER_LABELS[0]
         gender_confidence = (1.0 - gender_raw_value) * 100.0

        # =========================
        # GROWTH PREDICTION
        # multiclass argmax based sa sample mo
        # =========================
        growth_prediction = growth_model.predict(img_array, verbose=0)
        growth_class_index = int(np.argmax(growth_prediction[0]))
        growth_label = GROWTH_LABELS[growth_class_index]
        growth_confidence = float(growth_prediction[0][growth_class_index]) * 100.0

        # Existing contour-based estimate mo
        extra_growth = estimate_length_and_growth_stage(img_bytes)

        return {
            "gender": gender_label,
            "gender_confidence": round(gender_confidence, 2),
            "gender_raw_value": round(gender_raw_value, 6),

            "growth": growth_label,
            "growth_confidence": round(growth_confidence, 2),
            "growth_raw_probs": [round(float(x), 6) for x in growth_prediction[0].tolist()],

            "estimated_length_cm": extra_growth.get("estimated_length_cm", "Unknown"),
            "estimated_growth_stage": extra_growth.get("growth_stage", "Unknown"),
            "growth_debug": extra_growth.get("debug", "")
        }

    except Exception as e:
        traceback.print_exc()
        return {"error": f"Classification failed: {str(e)}"}


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

    # Ilagay lang ang tunay na model accuracy mula sa evaluation/test dataset.
    # Huwag gamitin ang prediction confidence bilang model accuracy.
    gender_model_accuracy = None
    growth_model_accuracy = None

    def render_classify_page():
        return render_template(
            'dashboard/classify.html',
            result=result,
            uploaded_file_url=uploaded_file_url,
            gender_model_accuracy=gender_model_accuracy,
            growth_model_accuracy=growth_model_accuracy
        )

    if request.method == 'POST':
        file = request.files.get('file')

        # ===== VALIDATE FILE =====
        if not file or not file.filename:
            flash('No file selected.', 'danger')
            return render_classify_page()

        original_filename = secure_filename(file.filename)

        if not original_filename:
            flash('Invalid file name.', 'danger')
            return render_classify_page()

        allowed_extensions = {'jpg', 'jpeg', 'png'}

        if '.' not in original_filename:
            flash('Invalid image file. Please upload JPG, JPEG, or PNG.', 'danger')
            return render_classify_page()

        file_extension = original_filename.rsplit('.', 1)[1].lower()

        if file_extension not in allowed_extensions:
            flash('Invalid image format. Please upload JPG, JPEG, or PNG.', 'danger')
            return render_classify_page()

        img_bytes = file.read()

        if not img_bytes:
            flash('The uploaded image is empty or invalid.', 'danger')
            return render_classify_page()

        # ===== VERIFY THAT FILE IS AN IMAGE =====
        try:
            image = Image.open(io.BytesIO(img_bytes))
            image.verify()
        except Exception:
            flash('The uploaded file is not a valid image.', 'danger')
            return render_classify_page()

        # ===== SAVE UPLOADED IMAGE =====
        try:
            unique_filename = (
                f"{uuid.uuid4().hex}_{original_filename}"
            )

            upload_folder = os.path.join(
                app.root_path,
                'static',
                'uploads'
            )

            os.makedirs(upload_folder, exist_ok=True)

            save_path = os.path.join(
                upload_folder,
                unique_filename
            )

            with open(save_path, 'wb') as uploaded_image:
                uploaded_image.write(img_bytes)

            uploaded_file_url = url_for(
                'static',
                filename=f'uploads/{unique_filename}'
            )

        except Exception as error:
            print("[ERROR] Image upload failed:", error)

            flash(
                f'Image upload failed: {str(error)}',
                'danger'
            )

            return render_classify_page()

        # ===== RUN CLASSIFICATION =====
        try:
            print("=== ROUTE CALLED classify ===")

            result = classify_crayfish(img_bytes)

            print("=== ROUTE RESULT ===", result)

        except Exception as error:
            traceback.print_exc()

            flash(
                f'Classification failed: {str(error)}',
                'danger'
            )

            result = None
            return render_classify_page()

        if not isinstance(result, dict):
            flash(
                'The classification model returned an invalid result.',
                'danger'
            )

            result = None
            return render_classify_page()

        if result.get('error'):
            flash(result['error'], 'danger')
            result = None
            return render_classify_page()

        # ===== EXTRACT RESULTS =====
        gender_result = result.get('gender', 'Unknown')
        growth_result = result.get('growth', 'Unknown')

        gender_confidence = result.get('gender_confidence')
        growth_confidence = result.get('growth_confidence')

        # ===== SAFE FLOAT CONVERSION =====
        try:
            gender_confidence = float(gender_confidence)
        except (TypeError, ValueError):
            gender_confidence = 0.0

        try:
            growth_confidence = float(growth_confidence)
        except (TypeError, ValueError):
            growth_confidence = 0.0

        # Ensure percentages stay between 0 and 100.
        gender_confidence = max(
            0.0,
            min(100.0, gender_confidence)
        )

        growth_confidence = max(
            0.0,
            min(100.0, growth_confidence)
        )

        # ===== MALE/FEMALE SCALE OUT OF 100 =====
        if str(gender_result).strip().lower() == 'male':
            male_percentage = gender_confidence
            female_percentage = 100.0 - gender_confidence

        elif str(gender_result).strip().lower() == 'female':
            female_percentage = gender_confidence
            male_percentage = 100.0 - gender_confidence

        else:
            male_percentage = 0.0
            female_percentage = 0.0

        # Add formatted values directly to result.
        # The HTML can use:
        # result.male_percentage
        # result.female_percentage
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

        result['male_percentage'] = round(
            male_percentage,
            2
        )

        result['female_percentage'] = round(
            female_percentage,
            2
        )

        # Preserve the length and stage values returned by the classifier.
        result['estimated_length_cm'] = result.get(
            'estimated_length_cm',
            'Unknown'
        )

        result['estimated_growth_stage'] = result.get(
            'estimated_growth_stage',
            'Unknown'
        )

        # ===== SAVE CLASSIFICATION TO DATABASE =====
        try:
            user_id = session.get('user_id') or 1
            current_time = datetime.utcnow()

            classification_log = ClassificationLog(
                user_id=user_id,
                image_filename=unique_filename,
                gender=gender_result,
                growth=growth_result,
                gender_confidence=gender_confidence,
                growth_confidence=growth_confidence,
                created_at=current_time
            )

            activity_log = ActivityLog(
                user_id=user_id,
                action=(
                    f"Classified image: "
                    f"{gender_result} "
                    f"({gender_confidence:.2f}%) / "
                    f"{growth_result} "
                    f"({growth_confidence:.2f}%)"
                ),
                timestamp=current_time
            )

            db.session.add(classification_log)
            db.session.add(activity_log)
            db.session.commit()

            print(
                "[OK] Classification result and "
                "activity saved to DB"
            )

        except Exception as error:
            db.session.rollback()

            print(
                "[ERROR] Failed to save classification result:",
                error
            )

            flash(
                'Classification succeeded, but the result '
                'could not be saved to the database.',
                'warning'
            )

    return render_classify_page()
@app.route('/verify-email-change/<token>')

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