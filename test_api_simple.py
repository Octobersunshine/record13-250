import os
import tempfile
import numpy as np
import soundfile as sf
import requests
import json


def generate_test_wav(bpm, duration_sec=10, sample_rate=22050):
    samples = int(duration_sec * sample_rate)
    t = np.arange(samples) / sample_rate

    beat_freq = bpm / 60.0
    beat_period = 1.0 / beat_freq

    envelope = np.zeros(samples)
    for i in range(int(duration_sec * beat_freq) + 1):
        beat_center = i * beat_period
        beat_sample = int(beat_center * sample_rate)
        width = int(0.05 * sample_rate)
        if beat_sample - width >= 0 and beat_sample + width < samples:
            envelope[beat_sample - width:beat_sample + width] = 1.0

    carrier_freq = 440.0
    audio = np.sin(2 * np.pi * carrier_freq * t) * envelope
    audio = audio * 0.5

    tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    tmp.close()
    sf.write(tmp.name, audio, sample_rate)
    return tmp.name


def test_health():
    print('=== Test 1: Health Check ===')
    resp = requests.get('http://127.0.0.1:5000/health')
    print(f'Status: {resp.status_code}')
    print(f'Response: {json.dumps(resp.json(), indent=2)}')
    assert resp.status_code == 200
    assert resp.json()['status'] == 'ok'
    print('PASSED\n')
    return True


def test_no_file():
    print('=== Test 2: No File Uploaded ===')
    resp = requests.post('http://127.0.0.1:5000/api/bpm')
    print(f'Status: {resp.status_code}')
    print(f'Response: {json.dumps(resp.json(), indent=2)}')
    assert resp.status_code == 400
    assert 'error' in resp.json()
    print('PASSED\n')
    return True


def test_empty_filename():
    print('=== Test 3: Empty Filename ===')
    files = {'file': ('', b'', 'audio/wav')}
    resp = requests.post('http://127.0.0.1:5000/api/bpm', files=files)
    print(f'Status: {resp.status_code}')
    print(f'Response: {json.dumps(resp.json(), indent=2)}')
    assert resp.status_code == 400
    assert 'error' in resp.json()
    print('PASSED\n')
    return True


def test_invalid_file_type():
    print('=== Test 4: Invalid File Type ===')
    tmp = tempfile.NamedTemporaryFile(suffix='.txt', delete=False)
    tmp.write(b'This is not an audio file')
    tmp.close()

    try:
        with open(tmp.name, 'rb') as f:
            files = {'file': ('test.txt', f, 'text/plain')}
            resp = requests.post('http://127.0.0.1:5000/api/bpm', files=files)
        print(f'Status: {resp.status_code}')
        print(f'Response: {json.dumps(resp.json(), indent=2)}')
        assert resp.status_code == 400
        assert 'error' in resp.json()
        print('PASSED\n')
        return True
    finally:
        os.unlink(tmp.name)


def test_bpm_detection():
    print('=== Test 5: BPM Detection ===')
    test_bpm = 120.0
    wav_path = generate_test_wav(test_bpm)

    try:
        with open(wav_path, 'rb') as f:
            files = {'file': ('test.wav', f, 'audio/wav')}
            resp = requests.post('http://127.0.0.1:5000/api/bpm', files=files)

        print(f'Status: {resp.status_code}')
        result = resp.json()
        print(f'Response: {json.dumps(result, indent=2)}')

        assert resp.status_code == 200
        assert result['success'] is True
        assert 'data' in result
        assert 'bpm' in result['data']

        detected_bpm = result['data']['bpm']
        tolerance = 5.0
        print(f'Expected BPM: {test_bpm}')
        print(f'Detected BPM: {detected_bpm}')
        print(f'Difference: {abs(detected_bpm - test_bpm):.2f} BPM')
        assert abs(detected_bpm - test_bpm) <= tolerance
        print('PASSED\n')
        return True
    finally:
        os.unlink(wav_path)


if __name__ == '__main__':
    print('BPM Detection Service API Tests')
    print('=' * 50 + '\n')

    tests = [
        test_health,
        test_no_file,
        test_empty_filename,
        test_invalid_file_type,
        test_bpm_detection,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f'FAILED: {e}\n')
            failed += 1

    print('=' * 50)
    print(f'Results: {passed} passed, {failed} failed')
    print(f'Total: {len(tests)} tests')
    exit(0 if failed == 0 else 1)
