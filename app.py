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


def analyze_bpm(file_path, num_estimates=5):
    y, sr = librosa.load(file_path, sr=None)
    duration = float(librosa.get_duration(y=y, sr=sr))

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    base_confidence = float(np.std(onset_env))

    tempo_estimates = []
    beat_counts = []

    start_bpm_range = [80, 100, 120]
    tightness_range = [100, 200, 300, 400, 500]

    for start_bpm in start_bpm_range:
        for tightness in tightness_range[:max(1, num_estimates // len(start_bpm_range) + 1)]:
            try:
                tempo, beat_frames = librosa.beat.beat_track(
                    y=y,
                    sr=sr,
                    onset_envelope=onset_env,
                    start_bpm=start_bpm,
                    tightness=tightness
                )
                if isinstance(tempo, np.ndarray):
                    tempo_val = float(tempo.item() if tempo.size == 1 else tempo[0])
                else:
                    tempo_val = float(tempo)

                if 40 <= tempo_val <= 250:
                    tempo_estimates.append(tempo_val)
                    beat_counts.append(len(beat_frames))
            except Exception:
                continue

    if len(tempo_estimates) == 0:
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, onset_envelope=onset_env)
        if isinstance(tempo, np.ndarray):
            final_tempo = float(tempo.item() if tempo.size == 1 else tempo[0])
        else:
            final_tempo = float(tempo)
        beat_count = len(beat_frames)
        tempo_std = 0.0
    else:
        final_tempo = float(np.median(tempo_estimates))
        tempo_std = float(np.std(tempo_estimates))
        beat_count = int(np.median(beat_counts)) if beat_counts else 0

    return {
        'bpm': round(final_tempo, 2),
        'beat_count': beat_count,
        'duration': round(duration, 2),
        'sample_rate': int(sr),
        'confidence': round(base_confidence, 4),
        'estimation_count': len(tempo_estimates) if tempo_estimates else 1,
        'bpm_std': round(tempo_std, 4)
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
