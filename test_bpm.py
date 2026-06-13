import os
import tempfile
import numpy as np
import soundfile as sf
import requests


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
    resp = requests.get('http://127.0.0.1:5000/health')
    print('Health check:', resp.status_code, resp.json())
    return resp.status_code == 200


def test_bpm_detect():
    test_bpm = 120.0
    wav_path = generate_test_wav(test_bpm)

    try:
        with open(wav_path, 'rb') as f:
            files = {'file': ('test.wav', f, 'audio/wav')}
            resp = requests.post('http://127.0.0.1:5000/api/bpm', files=files)
        
        print('BPM detect status:', resp.status_code)
        result = resp.json()
        print('BPM detect response:', result)
        
        if resp.status_code == 200 and result.get('success'):
            detected_bpm = result['data']['bpm']
            tolerance = 5.0
            passed = abs(detected_bpm - test_bpm) <= tolerance
            print(f'Expected BPM: {test_bpm}, Detected BPM: {detected_bpm}, Passed: {passed}')
            return passed
        return False
    finally:
        os.unlink(wav_path)


if __name__ == '__main__':
    print('Running BPM Detection Service Tests...')
    print('=' * 50)
    
    health_ok = test_health()
    print('Health check:', 'PASSED' if health_ok else 'FAILED')
    print()
    
    bpm_ok = test_bpm_detect()
    print('BPM detection:', 'PASSED' if bpm_ok else 'FAILED')
    print()
    
    print('=' * 50)
    all_passed = health_ok and bpm_ok
    print('Overall result:', 'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED')
    exit(0 if all_passed else 1)