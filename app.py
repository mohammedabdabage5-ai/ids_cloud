
import os
import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template_string
from datetime import datetime

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), ".")

model = joblib.load(os.path.join(MODEL_PATH, 'model.pkl'))
le = joblib.load(os.path.join(MODEL_PATH, 'label_encoder.pkl'))
scaler = joblib.load(os.path.join(MODEL_PATH, 'scaler.pkl'))
feature_names = joblib.load(os.path.join(MODEL_PATH, 'feature_names.pkl'))

print(f"Model loaded! Classes: {list(le.classes_)}")

alerts = []

@app.route('/')
def home():
    return render_template_string("""
    <!DOCTYPE html>
    <html dir="rtl" lang="ar">
    <head>
        <meta charset="UTF-8">
        <title>IDS - نظام كشف التسلل</title>
        <style>
            body { font-family: Arial; background: #1e1e2e; color: #cdd6f4; padding: 20px; text-align: center; }
            h1 { color: #89b4fa; }
            .card { background: #313244; padding: 20px; border-radius: 10px; margin: 20px auto; max-width: 800px; }
            .status { font-size: 24px; color: #a6e3a1; margin: 20px 0; }
            table { width: 100%; border-collapse: collapse; margin-top: 20px; }
            th, td { padding: 10px; border-bottom: 1px solid #45475a; }
            th { background: #45475a; }
            button { padding: 10px 20px; background: #89b4fa; border: none; border-radius: 5px; cursor: pointer; font-size: 14px; }
        </style>
    </head>
    <body>
        <h1>🛡️ نظام كشف التسلل</h1>
        <div class="card">
            <div class="status">✅ السيرفر يعمل</div>
            <p>عدد التنبيهات: <span id="count">0</span></p>
            <button onclick="loadAlerts()">🔄 تحديث</button>
        </div>
        <div class="card">
            <h3>📋 آخر التنبيهات</h3>
            <div id="alerts">لا توجد تنبيهات</div>
        </div>
        <script>
            async function loadAlerts() {
                try {
                    const res = await fetch('/api/alerts');
                    const data = await res.json();
                    document.getElementById('count').textContent = data.length;
                    if (data.length > 0) {
                        let html = '<table><tr><th>الوقت</th><th>نوع الهجوم</th><th>المصدر</th></tr>';
                        data.slice(-20).reverse().forEach(a => {
                            html += `<tr><td>${a.timestamp}</td><td style="color:#f38ba8">${a.attack}</td><td>${a.source}</td></tr>`;
                        });
                        html += '</table>';
                        document.getElementById('alerts').innerHTML = html;
                    }
                } catch(e) { console.log(e); }
            }
            loadAlerts();
            setInterval(loadAlerts, 5000);
        </script>
    </body>
    </html>
    """)

@app.route('/api/alerts')
def get_alerts():
    return jsonify(alerts)

@app.route('/api/report', methods=['POST'])
def report():
    data = request.json
    data['timestamp'] = data.get('timestamp', datetime.now().strftime('%H:%M:%S'))
    alerts.append(data)
    print(f"New alert: {data}")
    return jsonify({'status': 'success'}), 201

@app.route('/api/classify', methods=['POST'])
def classify():
    data = request.json
    features = data.get('features', [])
    sample = np.array(features).reshape(1, -1)
    scaled = scaler.transform(sample)
    pred = model.predict(scaled)[0]
    label = le.inverse_transform([pred])[0]
    proba = model.predict_proba(scaled)[0]
    return jsonify({'label': str(label), 'confidence': float(max(proba) * 100)})

@app.route('/health')
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
