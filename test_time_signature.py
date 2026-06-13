import os
import tempfile
import numpy as np
import soundfile as sf
import requests
import json


def generate_rhythm_wav(bpm, time_signature, duration_sec=20, sample_rate=22050):
    beats_per_measure = int(time_signature.split('/')[0])
    samples = int(duration_sec * sample_rate)
    t = np.arange(samples) / sample_rate

    beat_freq = bpm / 60.0
    beat_period = 1.0 / beat_freq

    envelope = np.zeros(samples)

    for i in range(int(duration_sec * beat_freq) + 1):
        beat_center = i * beat_period
        beat_sample = int(beat_center * sample_rate)

        pos_in_measure = i % beats_per_measure

        if pos_in_measure == 0:
            width = int(0.08 * sample_rate)
            amplitude = 1.0
        elif pos_in_measure == 2 and beats_per_measure == 4:
            width = int(0.06 * sample_rate)
            amplitude = 0.7
        else:
            width = int(0.05 * sample_rate)
            amplitude = 0.5

        if beat_sample - width >= 0 and beat_sample + width < samples:
            beat_env = np.zeros(width * 2)
            for j in range(width * 2):
                x = (j - width) / width
                beat_env[j] = np.exp(-x * x * 4)
            beat_env = beat_env * amplitude
            envelope[beat_sample - width:beat_sample + width] = np.maximum(
                envelope[beat_sample - width:beat_sample + width],
                beat_env
            )

    carrier_freq = 220.0 + 220.0
    carrier = np.sin(2 * np.pi * carrier_freq * t)

    audio = carrier * envelope
    audio = audio * 0.5

    tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    tmp.close()
    sf.write(tmp.name, audio, sample_rate)
    return tmp.name


def test_time_signature(bpm, time_signature, label):
    print(f'=== Test: {label} ({time_signature} @ {bpm} BPM ===')

    wav_path = generate_rhythm_wav(bpm, time_signature)

    try:
        with open(wav_path, 'rb') as f:
            files = {'file': ('test.wav', f, 'audio/wav')}
            resp = requests.post('http://127.0.0.1:5000/api/bpm', files=files)

        print(f'Status: {resp.status_code}')
        result = resp.json()

        if resp.status_code == 200 and result.get('success'):
            data = result['data']
            detected_ts = data.get('time_signature', 'unknown')
            ts_confidence = data.get('time_signature_confidence', 0)
            detected_bpm = data.get('bpm', 0)

            print(f'Detected BPM: {detected_bpm:.2f} (expected: {bpm})')
            print(f'Detected Time Signature: {detected_ts} (expected: {time_signature})')
            print(f'Confidence: {ts_confidence:.4f}')

            bpm_correct = abs(detected_bpm - bpm) <= 5.0
            ts_correct = detected_ts == time_signature

            print(f'BPM: {"✓" if bpm_correct else "✗"}')
            print(f'Time Sig: {"✓" if ts_correct else "✗"}')

            return bpm_correct and ts_correct, data
        else:
            print(f'Error: {result}')
            return False, result
    finally:
        os.unlink(wav_path)


if __name__ == '__main__':
    print('Time Signature Detection Tests')
    print('=' * 60)
    print()

    tests = [
        (120, '4/4', '4/4 at 120 BPM'),
        (90, '4/4', '4/4 at 90 BPM'),
        (140, '4/4', '4/4 at 140 BPM'),
        (100, '3/4', '3/4 at 100 BPM'),
        (120, '3/4', '3/4 at 120 BPM'),
        (80, '3/4', '3/4 at 80 BPM'),
    ]

    passed = 0
    failed = 0
    results = []

    for bpm, ts, label in tests:
        success, data = test_time_signature(bpm, ts, label)
        results.append((label, success, data))
        if success:
            passed += 1
        else:
            failed += 1
        print()

    print('=' * 60)
    print(f'Results: {passed} passed, {failed} failed')
    print(f'Total: {len(tests)} tests')
    print()

    print('Summary:')
    for label, success, data in results:
        status = '✓' if success else '✗'
        if isinstance(data, dict) and 'time_signature' in data:
            ts = data.get('time_signature', '?')
            conf = data.get('time_signature_confidence', 0)
            bpm = data.get('bpm', 0)
            print(f'  {status} {label}: {ts} (conf: {conf:.2f}, bpm: {bpm:.1f})')
        else:
            print(f'  {status} {label}: ERROR')

    exit(0 if failed == 0 else 1)
