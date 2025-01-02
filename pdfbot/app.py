from flask import Flask, request, render_template, jsonify, session
from flask_session import Session
import os
from werkzeug.utils import secure_filename
from base_rag import rag_process
import base64
import io
import os
from datetime import datetime, timedelta


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 21 * 1024 * 1024  # 10 MB limit

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = './session_data'
app.config['SESSION_COOKIE_NAME'] = 'my_session_cookie'
os.makedirs(app.config['SESSION_FILE_DIR'], exist_ok=True)

Session(app)

@app.route('/')
def home():
    return render_template('index.html')

def cleanup_old_files():
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file_time = datetime.fromtimestamp(os.path.getmtime(filepath))
        if datetime.now() - file_time > timedelta(hours=1):  # Delete files older than 1 hour
            os.remove(filepath)
            
@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    print("==> PDF uploaded and saved")
    rag = rag_process(file_path)
    session['rag_model'] = rag
    session['pdf_path'] = file_path
    return jsonify({'message': 'PDF uploaded and processed successfully'})

@app.route('/status', methods=['GET'])
def status():
    if 'rag_model' in session:
        return jsonify({'status': 'ready'})
    return jsonify({'status': 'processing'})

@app.route('/ask', methods=['POST'])
def ask():
    question = request.json.get('question')
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    if 'rag_model' not in session:
        return jsonify({'error': 'No PDF uploaded'}), 400
    rag_model = session['rag_model']
    answer, bin_img, page_n = rag_model.execute(question)

    if bin_img:
        img_byte_arr = io.BytesIO()
        bin_img.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        bin_img_base64 = base64.b64encode(img_byte_arr).decode('utf-8')
    else:
        bin_img_base64 = None

    #bin_img_base64 = base64.b64encode(bin_img).decode('utf-8') if bin_img else None
    return jsonify({'answer': answer, 'image': bin_img_base64, 'page_numbers': page_n})

@app.route('/end_session', methods=['POST'])
def end_session():
    if 'pdf_path' in session:
        os.remove(session['pdf_path'])
        session.pop('pdf_path', None)
        session.pop('rag_model', None)
    cleanup_old_files()  # Clean up old files before new upload

    return jsonify({'message': 'Session ended successfully'})

if __name__ == '__main__':
    app.run(debug=True)
