# Binaural Audio Examples

A Python-based utility for generating spatialized binaural audio examples. This project uses Interaural Time Difference (ITD) and Interaural Level Difference (ILD) approximations to create a 3D audio experience through standard stereo headphones.

## Features

The generator includes several pre-configured scenes:
- **Rotating Tones**: A dual-frequency tone that orbits 360 degrees around the listener.
- **Beeps and Ticks**: Rhythmic elements that panned to specific static and moving positions.
- **Overlapping Layers**: Multiple tonal drones layered at different spatial coordinates.
- **Surround Motion**: Noise bursts that move dynamically across the rear and front soundstages.
- **Tick Rain**: A dense environment of over 200 randomized spatial "ticks" falling around the listener.

## Requirements

- Python 3.10+
- [NumPy](https://numpy.org/)
- [sounddevice](https://python-sounddevice.readthedocs.io/)

## Installation

```bash
pip install numpy sounddevice
```

## Usage

Generate the default binaural showcase file:

```bash
python generate_binaural_samples.py --output binaural_samples.wav
```

### Options
- `--output`: Specify the output filename (default: `binaural_samples.wav`).
- `--sr`: Set the sample rate (default: `48000`).
- `--play`: Preview the generated audio immediately through your default sound device.

## How it Works

The script converts mono signals into binaural stereo by:
1. **ITD**: Delaying one channel by several samples based on the speed of sound and average human ear distance.
2. **ILD**: Applying gain differentials to simulate head shadowing.
3. **Dynamic Panning**: Using block-based processing to move sounds in real-time across the azimuth.
