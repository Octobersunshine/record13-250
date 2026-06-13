import os
import tempfile
from flask import Flask, request, jsonify
import librosa
import numpy as np

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'mp3', 'wav', 'ogg', 'flac', 'm4a'}
MAX_CONTENT_LENGTH = 50 * 1024 * 1024

app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def analyze_bpm(file_path):
    y, sr = librosa.load(file_path, sr=None)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    confidence = float(np.std(librosa.onset.onset_strength(y=y, sr=sr)))

    if isinstance(tempo, np.ndarray):
        tempo = float(tempo.item() if tempo.size == 1 else tempo[0])
    else:
        tempo = float(tempo)

    return {
        'bpm': round(tempo, 2),
        'beat_count': len(beat_times),
        'duration': round(float(librosa.get_duration(y=y, sr=sr)), 2),
        'sample_rate': int(sr),
        'confidence': round(confidence, 4)
    }


@app.route('/api/bpm', methods=['POST'])
def bpm_detect():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed. Supported: mp3, wav, ogg, flac, m4a'}), 400

    try:
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name

        try:
            result = analyze_bpm(tmp_path)
            return jsonify({
                'success': True,
                'filename': file.filename,
                'data': result
            })
        finally:
            os.unlink(tmp_path)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'service': 'BPM Detection Service'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
