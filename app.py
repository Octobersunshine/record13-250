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


def detect_time_signature(y, sr, onset_env, beat_frames):
    if len(beat_frames) < 8:
        return {'time_signature': '4/4', 'confidence': 0.5, 'beat_strengths': []}

    beat_strengths = []
    for bf in beat_frames:
        start = max(0, bf - 2)
        end = min(len(onset_env), bf + 3)
        if end > start:
            beat_strengths.append(float(np.max(onset_env[start:end])))
        else:
            beat_strengths.append(float(onset_env[bf]) if bf < len(onset_env) else 0.0)

    beat_strengths = np.array(beat_strengths)
    if len(beat_strengths) == 0:
        return {'time_signature': '4/4', 'confidence': 0.5, 'beat_strengths': []}

    mean_strength = np.mean(beat_strengths)
    std_strength = np.std(beat_strengths)
    if std_strength < 0.01:
        return {
            'time_signature': '4/4',
            'confidence': 0.5,
            'beat_strengths': [round(float(s), 4) for s in beat_strengths[:10]]
        }

    norm_strengths = (beat_strengths - mean_strength) / std_strength

    def measure_downbeat_contrast(strengths, beats_per_measure):
        if len(strengths) < beats_per_measure * 2:
            return 0.0

        contrasts = []
        for start in range(beats_per_measure):
            downbeat_strengths = []
            other_strengths = []

            for i in range(start, len(strengths)):
                pos_in_measure = (i - start) % beats_per_measure
                if pos_in_measure == 0:
                    downbeat_strengths.append(strengths[i])
                else:
                    other_strengths.append(strengths[i])

            if len(downbeat_strengths) > 0 and len(other_strengths) > 0:
                mean_downbeat = np.mean(downbeat_strengths)
                mean_other = np.mean(other_strengths)
                contrast = mean_downbeat - mean_other
                contrasts.append(contrast)

        return float(max(contrasts)) if contrasts else 0.0

    def autocorrelation_peak(strengths, lag):
        if len(strengths) <= lag:
            return 0.0
        return float(np.correlate(strengths[:-lag], strengths[lag:])[0] / (len(strengths) - lag))

    contrast_3 = measure_downbeat_contrast(norm_strengths, 3)
    contrast_4 = measure_downbeat_contrast(norm_strengths, 4)

    ac_3 = autocorrelation_peak(norm_strengths, 3)
    ac_4 = autocorrelation_peak(norm_strengths, 4)

    score_3 = contrast_3 + ac_3 * 0.5
    score_4 = contrast_4 + ac_4 * 0.5

    total_score = abs(score_3) + abs(score_4)
    if total_score < 0.01:
        confidence = 0.5
        time_sig = '4/4'
    else:
        if score_4 >= score_3:
            time_sig = '4/4'
            confidence = score_4 / total_score
        else:
            time_sig = '3/4'
            confidence = score_3 / total_score

    confidence = max(0.5, min(0.99, 0.5 + confidence * 0.5))

    return {
        'time_signature': time_sig,
        'confidence': round(float(confidence), 4),
        'beat_strengths': [round(float(s), 4) for s in beat_strengths[:10]],
        'score_34': round(float(score_3), 4),
        'score_44': round(float(score_4), 4)
    }


def analyze_bpm(file_path, num_estimates=5):
    y, sr = librosa.load(file_path, sr=None)
    duration = float(librosa.get_duration(y=y, sr=sr))

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    base_confidence = float(np.std(onset_env))

    tempo_estimates = []
    beat_counts = []
    beat_frames_list = []

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
                    beat_frames_list.append(beat_frames)
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
        best_beat_frames = beat_frames
    else:
        final_tempo = float(np.median(tempo_estimates))
        tempo_std = float(np.std(tempo_estimates))
        beat_count = int(np.median(beat_counts)) if beat_counts else 0

        median_idx = np.argsort(tempo_estimates)[len(tempo_estimates) // 2]
        best_beat_frames = beat_frames_list[median_idx]

    time_sig_result = detect_time_signature(y, sr, onset_env, best_beat_frames)

    return {
        'bpm': round(final_tempo, 2),
        'beat_count': beat_count,
        'duration': round(duration, 2),
        'sample_rate': int(sr),
        'confidence': round(base_confidence, 4),
        'estimation_count': len(tempo_estimates) if tempo_estimates else 1,
        'bpm_std': round(tempo_std, 4),
        'time_signature': time_sig_result['time_signature'],
        'time_signature_confidence': time_sig_result['confidence']
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
