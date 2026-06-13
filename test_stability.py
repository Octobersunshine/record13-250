import os
import tempfile
import numpy as np
import soundfile as sf
import requests


def generate_test_wav(bpm, duration_sec=15, sample_rate=22050):
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


def run_multiple_tests(test_bpm, num_runs=10):
    wav_path = generate_test_wav(test_bpm)
    results = []

    try:
        for i in range(num_runs):
            with open(wav_path, 'rb') as f:
                files = {'file': ('test.wav', f, 'audio/wav')}
                resp = requests.post('http://127.0.0.1:5000/api/bpm', files=files)

            if resp.status_code == 200:
                data = resp.json()['data']
                results.append({
                    'run': i + 1,
                    'bpm': data['bpm'],
                    'estimation_count': data.get('estimation_count', 1),
                    'bpm_std': data.get('bpm_std', 0)
                })
                print(f'Run {i+1:2d}: BPM = {data["bpm"]:6.2f} | '
                      f'Estimates = {data.get("estimation_count", 1):2d} | '
                      f'Std = {data.get("bpm_std", 0):6.2f}')
    finally:
        os.unlink(wav_path)

    return results


def analyze_stability(results, expected_bpm):
    bpms = [r['bpm'] for r in results]
    bpm_array = np.array(bpms)

    mean_bpm = float(np.mean(bpm_array))
    median_bpm = float(np.median(bpm_array))
    std_bpm = float(np.std(bpm_array))
    min_bpm = float(np.min(bpm_array))
    max_bpm = float(np.max(bpm_array))
    range_bpm = max_bpm - min_bpm

    tolerance = 5.0
    within_tolerance = np.sum(np.abs(bpm_array - expected_bpm) <= tolerance)
    tolerance_rate = within_tolerance / len(bpms) * 100

    print()
    print('=' * 60)
    print('STABILITY ANALYSIS REPORT')
    print('=' * 60)
    print(f'Total runs:          {len(bpms)}')
    print(f'Expected BPM:        {expected_bpm:.2f}')
    print()
    print(f'Mean BPM:            {mean_bpm:.2f}')
    print(f'Median BPM:          {median_bpm:.2f}')
    print(f'Std Deviation:       {std_bpm:.4f}')
    print(f'Min BPM:             {min_bpm:.2f}')
    print(f'Max BPM:             {max_bpm:.2f}')
    print(f'Range (Max-Min):     {range_bpm:.4f}')
    print()
    print(f'Tolerance:           ±{tolerance} BPM')
    print(f'Within tolerance:    {within_tolerance}/{len(bpms)} ({tolerance_rate:.1f}%)')
    print()

    stable = std_bpm < 1.0 and tolerance_rate >= 90
    print(f'Overall stability:   {"STABLE ✓" if stable else "NEEDS IMPROVEMENT ✗"}')
    print('=' * 60)

    return stable


if __name__ == '__main__':
    test_bpm = 120.0
    num_runs = 10

    print(f'BPM Detection Stability Test')
    print(f'Target BPM: {test_bpm} | Runs: {num_runs}')
    print('=' * 60)

    results = run_multiple_tests(test_bpm, num_runs)
    stable = analyze_stability(results, test_bpm)

    exit(0 if stable else 1)
