import argparse
import wave
from dataclasses import dataclass

import numpy as np
import sounddevice as sd


@dataclass
class AudioConfig:
    sample_rate: int = 48_000
    output_path: str = "binaural_samples.wav"
    peak_level: float = 0.95


SPEED_OF_SOUND = 343.0
EAR_DISTANCE_M = 0.18


def sine(freq_hz: float, duration_s: float, sr: int, phase: float = 0.0) -> np.ndarray:
    t = np.arange(int(duration_s * sr), dtype=np.float32) / sr
    return np.sin(2.0 * np.pi * freq_hz * t + phase).astype(np.float32)


def sine_n(freq_hz: float, n_samples: int, sr: int, phase: float = 0.0) -> np.ndarray:
    t = np.arange(n_samples, dtype=np.float32) / sr
    return np.sin(2.0 * np.pi * freq_hz * t + phase).astype(np.float32)


def triangle(freq_hz: float, duration_s: float, sr: int) -> np.ndarray:
    t = np.arange(int(duration_s * sr), dtype=np.float32) / sr
    x = (t * freq_hz) % 1.0
    tri = 4.0 * np.abs(x - 0.5) - 1.0
    return tri.astype(np.float32)


def triangle_n(freq_hz: float, n_samples: int, sr: int) -> np.ndarray:
    t = np.arange(n_samples, dtype=np.float32) / sr
    x = (t * freq_hz) % 1.0
    tri = 4.0 * np.abs(x - 0.5) - 1.0
    return tri.astype(np.float32)


def delay_signal(x: np.ndarray, delay_samples: float) -> np.ndarray:
    n = np.arange(x.size, dtype=np.float32)
    # Linear interpolation supports smooth sub-sample interaural delays.
    return np.interp(n - delay_samples, n, x, left=0.0, right=0.0).astype(np.float32)


def spatialize_static(mono: np.ndarray, azimuth_deg: float, sr: int) -> np.ndarray:
    theta = np.deg2rad(azimuth_deg)

    # Simple ITD approximation.
    itd_s = (EAR_DISTANCE_M * np.sin(theta)) / SPEED_OF_SOUND
    itd_samples = itd_s * sr

    if itd_samples >= 0:
        left = delay_signal(mono, itd_samples)
        right = mono.copy()
    else:
        left = mono.copy()
        right = delay_signal(mono, -itd_samples)

    # Simple ILD approximation up to +/- 6 dB.
    ild_db = 6.0 * np.sin(theta)
    right_gain = 10.0 ** (ild_db / 20.0)
    left_gain = 10.0 ** (-ild_db / 20.0)

    left *= left_gain
    right *= right_gain

    out = np.column_stack((left, right)).astype(np.float32)

    peak = np.max(np.abs(out)) + 1e-9
    if peak > 1.0:
        out /= peak
    return out


def spatialize_dynamic(mono: np.ndarray, azimuth_curve_deg: np.ndarray, sr: int, block: int = 256) -> np.ndarray:
    out = np.zeros((mono.size, 2), dtype=np.float32)

    for start in range(0, mono.size, block):
        end = min(start + block, mono.size)
        az = float(np.mean(azimuth_curve_deg[start:end]))
        out[start:end] = spatialize_static(mono[start:end], az, sr)

    return out


def adsr_env(total_samples: int, sr: int, a=0.01, d=0.08, s=0.7, r=0.08) -> np.ndarray:
    attack = int(a * sr)
    decay = int(d * sr)
    release = int(r * sr)
    sustain = max(total_samples - (attack + decay + release), 0)

    env = np.zeros(total_samples, dtype=np.float32)
    pos = 0

    if attack > 0:
        env[pos:pos + attack] = np.linspace(0.0, 1.0, attack, endpoint=False)
        pos += attack
    if decay > 0:
        env[pos:pos + decay] = np.linspace(1.0, s, decay, endpoint=False)
        pos += decay
    if sustain > 0:
        env[pos:pos + sustain] = s
        pos += sustain
    if release > 0 and pos < total_samples:
        env[pos:] = np.linspace(env[pos - 1] if pos > 0 else s, 0.0, total_samples - pos)

    return env


def scene_rotating_tone(sr: int, duration_s: float = 8.0) -> np.ndarray:
    n = int(sr * duration_s)
    t = np.arange(n, dtype=np.float32) / sr

    base = 0.45 * sine(220.0, duration_s, sr)
    wobble = 0.25 * sine(440.0, duration_s, sr, phase=np.pi / 3)
    mono = base + wobble

    azimuth = (t / duration_s) * 360.0
    return spatialize_dynamic(mono, azimuth, sr)


def scene_beeps_and_ticks(sr: int, duration_s: float = 6.0) -> np.ndarray:
    n = int(sr * duration_s)
    stereo = np.zeros((n, 2), dtype=np.float32)

    beat_hz = 2.0
    beat_interval = 1.0 / beat_hz
    beep_len_s = 0.18
    tick_len_s = 0.03

    k = 0
    while True:
        start_s = k * beat_interval
        if start_s >= duration_s:
            break
        start = int(start_s * sr)
        end = min(start + int(beep_len_s * sr), n)
        seg_len = end - start

        beep = 0.5 * sine_n(700.0 + 90.0 * (k % 3), seg_len, sr)
        beep *= adsr_env(seg_len, sr, a=0.005, d=0.03, s=0.55, r=0.05)

        az = -60.0 if (k % 2 == 0) else 60.0
        stereo[start:end] += spatialize_static(beep, az, sr)

        tick_start = start + int(0.08 * sr)
        tick_end = min(tick_start + int(tick_len_s * sr), n)
        if tick_end > tick_start:
            tick_len = tick_end - tick_start
            tick = 0.8 * triangle_n(3500.0, tick_len, sr)
            tick *= adsr_env(tick_len, sr, a=0.001, d=0.003, s=0.2, r=0.01)
            tick_az = -110.0 + (k * 40.0) % 220.0
            stereo[tick_start:tick_end] += spatialize_static(tick, tick_az, sr)

        k += 1

    return stereo


def scene_overlapping_layers(sr: int, duration_s: float = 9.0) -> np.ndarray:
    n = int(sr * duration_s)
    t = np.arange(n, dtype=np.float32) / sr

    drone1 = 0.25 * np.sin(2 * np.pi * 110.0 * t)
    drone2 = 0.20 * np.sin(2 * np.pi * 164.81 * t + 0.7)
    drone3 = 0.18 * np.sin(2 * np.pi * 246.94 * t + 1.1)

    amp_mod = 0.55 + 0.45 * np.sin(2 * np.pi * 0.15 * t)
    pulse = 0.24 * np.sin(2 * np.pi * 6.0 * t) * np.sin(2 * np.pi * 520.0 * t)

    layer_a = spatialize_static((drone1 + 0.7 * pulse).astype(np.float32), -35.0, sr)
    layer_b = spatialize_static((drone2 * amp_mod).astype(np.float32), 35.0, sr)
    layer_c = spatialize_static((drone3 * (1.0 - amp_mod * 0.5)).astype(np.float32), 140.0, sr)

    return layer_a + layer_b + layer_c


def scene_surround_motion(sr: int, duration_s: float = 10.0) -> np.ndarray:
    n = int(sr * duration_s)
    t = np.arange(n, dtype=np.float32) / sr

    noise = np.random.default_rng(42).normal(0.0, 1.0, n).astype(np.float32)
    # Window the noise into repeating bursts.
    bursts = (np.sin(2 * np.pi * 1.2 * t) > 0.55).astype(np.float32)
    bursts *= adsr_env(n, sr, a=0.02, d=0.1, s=0.8, r=0.2)

    moving_source = 0.22 * noise * bursts
    azimuth = 180.0 * np.sin(2 * np.pi * 0.12 * t) + 120.0 * np.sin(2 * np.pi * 0.05 * t)

    bed = 0.14 * sine(95.0, duration_s, sr)
    bed_stereo = spatialize_static(bed, -150.0, sr)

    return bed_stereo + spatialize_dynamic(moving_source, azimuth, sr)


def scene_tick_rain(sr: int, duration_s: float = 7.0) -> np.ndarray:
    n = int(sr * duration_s)
    stereo = np.zeros((n, 2), dtype=np.float32)
    rng = np.random.default_rng(7)

    for _ in range(220):
        center = rng.uniform(0.0, duration_s)
        length_s = rng.uniform(0.01, 0.035)
        start = int(center * sr)
        end = min(start + int(length_s * sr), n)
        if end <= start:
            continue

        seg_len = end - start
        freq = rng.uniform(1700.0, 5200.0)
        tick = 0.35 * sine_n(freq, seg_len, sr)
        tick *= adsr_env(seg_len, sr, a=0.001, d=0.004, s=0.1, r=0.01)

        az = rng.uniform(-170.0, 170.0)
        stereo[start:end] += spatialize_static(tick, az, sr)

    return stereo


def normalize(stereo: np.ndarray, peak_target: float = 0.95) -> np.ndarray:
    peak = np.max(np.abs(stereo)) + 1e-9
    return (stereo / peak * peak_target).astype(np.float32)


def silence(duration_s: float, sr: int) -> np.ndarray:
    return np.zeros((int(duration_s * sr), 2), dtype=np.float32)


def write_wav_16bit(path: str, stereo: np.ndarray, sr: int) -> None:
    pcm = np.clip(stereo, -1.0, 1.0)
    pcm_i16 = (pcm * 32767.0).astype(np.int16)

    with wave.open(path, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm_i16.tobytes())


def build_showcase(sr: int) -> np.ndarray:
    parts = [
        scene_rotating_tone(sr),
        silence(0.4, sr),
        scene_beeps_and_ticks(sr),
        silence(0.4, sr),
        scene_overlapping_layers(sr),
        silence(0.4, sr),
        scene_surround_motion(sr),
        silence(0.4, sr),
        scene_tick_rain(sr),
    ]
    return np.concatenate(parts, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate binaural sample scenes and write a stereo WAV file.")
    parser.add_argument("--output", default="binaural_samples.wav", help="Output WAV filename.")
    parser.add_argument("--sr", type=int, default=48_000, help="Sample rate.")
    parser.add_argument("--play", action="store_true", help="Preview playback through sounddevice.")
    args = parser.parse_args()

    cfg = AudioConfig(sample_rate=args.sr, output_path=args.output)

    stereo = build_showcase(cfg.sample_rate)
    stereo = normalize(stereo, cfg.peak_level)
    write_wav_16bit(cfg.output_path, stereo, cfg.sample_rate)

    duration = stereo.shape[0] / cfg.sample_rate
    print(f"Wrote {cfg.output_path} ({duration:.2f}s, {cfg.sample_rate} Hz, stereo)")

    if args.play:
        print("Playing preview via sounddevice...")
        sd.play(stereo, samplerate=cfg.sample_rate)
        sd.wait()


if __name__ == "__main__":
    main()
