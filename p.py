from flask import Flask, render_template, request, redirect, url_for, jsonify, session
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
import google.generativeai as genai
import pandas as pd

app = Flask(__name__)

# Konfigurasi Gemini
genai.configure(api_key="AIzaSyCZ-yDxOS5oKbMwMjobXJwPPgmzF6qNMW0")  # Ganti dengan API key Gemini-mu
model = genai.GenerativeModel('gemini-2.0-flash-001')  # Ganti dengan model yang sesuai

# Konfigurasi database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///patients.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# model machine learning
import joblib
# load traine mode
model = joblib.load('knn_model.pkl')

# Model Pasien
class Patient(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    birthdate = db.Column(db.Date, nullable=False)
    gender = db.Column(db.String(10), nullable=False)
    height = db.Column(db.Float, nullable=False)
    weight = db.Column(db.Float, nullable=False)
    age = db.Column(db.Integer, nullable=False)
    bmi = db.Column(db.Float, nullable=False)
    glucose = db.Column(db.Float, nullable=False)
    cholesterol = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    __table_args__ = (
        db.UniqueConstraint('name', 'birthdate', name='_name_birthdate_uc'),
    )

# Inisialisasi database
with app.app_context():
    db.create_all()

# Session
app.secret_key = 'anan020006'

# Data sensor terbaru
sensor_data = {"glucose": 0}

@app.route('/')
def awal():
    return render_template('home.html')

@app.route('/data')
def index():
    return render_template('index.html')

# Endpoint untuk memperbarui data sensor dari ESP32
@app.route('/sensor', methods=['POST'])
def receive_data():
    try:
        data = request.get_json()
        glucose = data.get("glucose")

        if glucose is not None:
            # Simpan nilai sementara
            sensor_data['glucose'] = glucose

            # Cari pasien terakhir yang terdaftar
            latest_patient = Patient.query.order_by(Patient.id.desc()).first()
            if latest_patient:
                latest_patient.glucose = glucose
                latest_patient.timestamp = datetime.utcnow()
                db.session.commit()
                print(f"Data updated for {latest_patient.name}: Glucose={glucose}")
            else:
                print("No patients found in the database. Cannot update data.")

            return jsonify({"message": "Data stored successfully"}), 201

        return jsonify({"error": "Invalid data"}), 400
    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": "Server error"}), 500



# Endpoint untuk mengambil data sensor terbaru (digunakan oleh frontend AJAX)
@app.route('/get_latest_patient', methods=['GET'])
def get_latest_patient():
    """Mengambil data pasien terbaru dari database"""
    latest_patient = Patient.query.order_by(Patient.id.desc()).first()
    
    if latest_patient:
        return jsonify({
            "name": latest_patient.name,
            "gender": latest_patient.gender,
            "age": latest_patient.age,
            "bmi": round(latest_patient.bmi, 2),
            "glucose": latest_patient.glucose
        })
    
    return jsonify({"message": "No patient data available"}), 404


@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        name = request.form['name']
        gender = request.form['gender']
        birthdate = request.form['birthdate']
        height = float(request.form['height'])
        weight = float(request.form['weight'])

        # Hitung usia
        birthdate_obj = datetime.strptime(birthdate, '%Y-%m-%d')
        today = datetime.today()
        age = today.year - birthdate_obj.year - ((today.month, today.day) < (birthdate_obj.month, birthdate_obj.day))

        # Hitung BMI
        bmi = weight / ((height / 100) ** 2)

        # Ambil data pasien terbaru dari database
        latest_patient = Patient.query.order_by(Patient.id.desc()).first()
        glucose = latest_patient.glucose if latest_patient else 0

        # Cek apakah pasien sudah ada
        existing_patient = Patient.query.filter_by(name=name, birthdate=birthdate_obj.date()).first()

        if existing_patient:
            existing_patient.gender = gender
            existing_patient.height = height
            existing_patient.weight = weight
            existing_patient.age = age
            existing_patient.bmi = bmi
            existing_patient.glucose = glucose
            existing_patient.timestamp = datetime.utcnow()
        else:
            new_patient = Patient(
                name=name,
                birthdate=birthdate_obj.date(),
                gender=gender,
                height=height,
                weight=weight,
                age=age,
                bmi=bmi,
                glucose=glucose,
            )
            db.session.add(new_patient)

        # Simpan ke database
        try:
            db.session.commit()
        except:
            db.session.rollback()
            raise

        return render_template('result.html',
    name=name,
    gender=gender,
    age=age,
    bmi=round(bmi, 2),
    glucose=glucose
)


    return redirect(url_for('index'))
@app.route('/deteksi', methods=['POST'])
def deteksi():
    age = float(request.form.get("age", 0))
    bmi = float(request.form.get("bmi", 0))
    glucose = float(request.form.get("glucose", 0))

    # Gunakan DataFrame agar sesuai dengan model
    import pandas as pd
    features = pd.DataFrame([[glucose, age, bmi]], columns=["glucose", "age", "bmi"])
    prediction = model.predict(features)
    result = prediction[0]

    if result == 1:
        return redirect(url_for('hasilA'))
    else:
        return redirect(url_for('hasilB'))

@app.route('/hasilA')
def hasilA():
    return render_template('positif.html')

@app.route('/hasilB')
def hasilB():
    return render_template('negatif.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    # Ambil data glukosa terbaru dari sensor_data
    # glucose = sensor_data.get('glucose', 0)
    glucose = float(request.form.get("glucose", 0))
    # Prompt ke Gemini
    prompt = f"Kadar glukosa seseorang adalah {glucose} mg/dL. Jelaskan kondisi kesehatannya secara singkat jika di periksa dalam keadaan puasa, 2 jam makan, dan tes acak, berikan juga kesimpulan dan saran kepada pasien."
    response = model.generate_content(prompt)

    return jsonify({
        'explanation': response.text
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
