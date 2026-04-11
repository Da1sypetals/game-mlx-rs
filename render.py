#!/usr/bin/env python3
"""
End-to-end vocal-to-piano renderer.

Usage:
    python render.py <audio_file> [options]

Runs game-mlxrs to produce a JSON score, then renders it as a piano WAV using
FluidSynth with per-note tuning for arbitrary (non-semitone) pitch precision.
"""

import argparse
import ctypes
import json
import math
import os
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

GAME_MANIFEST = os.path.join(SCRIPT_DIR, "game-mlxrs", "Cargo.toml")
WEIGHTS = os.path.join(SCRIPT_DIR, "game-mlx", "weights", "GAME-1.0-large.safetensors")
CONFIG = os.path.join(SCRIPT_DIR, "GAME", "models", "GAME-1.0-large", "config.yaml")
SF2_PATH = os.path.join(SCRIPT_DIR, "FluidR3_GM.sf2")

SAMPLE_RATE = 44100
# MIDI note 0..127, piano = program 0
PIANO_PROGRAM = 0


def find_pitch_delta(notes: list) -> float:
    """Find optimal pitch offset delta (in semitones) that minimizes duration-weighted
    quantization error. Uses exact piecewise-linear analysis.

    For each present note with pitch p_i and duration w_i, the quantization error is
    d(p_i + delta) = |frac(p_i + delta + 0.5) - 0.5| where frac is fractional part.
    The total cost L(delta) = sum_i w_i * d(p_i + delta) is piecewise linear with period 1.
    Breakpoints occur where some p_i + delta is exactly at a half-integer (midpoint between
    two semitones). The minimum of L must occur at one of these breakpoints.
    """
    pitches = []
    weights = []
    for n in notes:
        if not n["presence"]:
            continue
        pitches.append(float(n["pitch"]))
        weights.append(float(n["offset"]) - float(n["onset"]))

    if not pitches:
        return 0.0

    pitches = np.array(pitches)
    weights = np.array(weights)

    # Candidate deltas: values where some note has zero quantization error
    # i.e. p_i + delta = round(p_i + delta), which means delta = k - p_i for integer k
    # Mapped to [-0.5, 0.5): delta = -frac(p_i) mapped to [-0.5, 0.5)
    frac_parts = pitches - np.floor(pitches)  # in [0, 1)
    candidates = -frac_parts  # in (-1, 0]
    # Map to [-0.5, 0.5)
    candidates = ((candidates + 0.5) % 1.0) - 0.5

    # Also include breakpoints (half-integer crossings) where derivative changes
    breakpoints = 0.5 - frac_parts
    breakpoints = ((breakpoints + 0.5) % 1.0) - 0.5

    all_candidates = np.concatenate([candidates, breakpoints, [0.0]])

    best_delta = 0.0
    best_cost = float("inf")
    for d in all_candidates:
        shifted = pitches + d
        errors = np.abs(shifted - np.round(shifted))
        cost = np.dot(weights, errors)
        if cost < best_cost:
            best_cost = cost
            best_delta = float(d)

    return best_delta


def load_fluidsynth_lib():
    homebrew_prefix = os.getenv("HOMEBREW_PREFIX", "/opt/homebrew")
    lib_path = os.path.join(homebrew_prefix, "lib", "libfluidsynth.dylib")
    return ctypes.CDLL(lib_path)


def make_settings(lib):
    lib.new_fluid_settings.restype = ctypes.c_void_p
    settings = lib.new_fluid_settings()
    assert settings, "failed to create fluid settings"

    lib.fluid_settings_setnum.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_double]
    lib.fluid_settings_setnum(settings, b"synth.sample-rate", float(SAMPLE_RATE))
    lib.fluid_settings_setint.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
    lib.fluid_settings_setint(settings, b"synth.reverb.active", 0)
    lib.fluid_settings_setint(settings, b"synth.chorus.active", 0)
    return settings


def make_synth(lib, settings):
    lib.new_fluid_synth.restype = ctypes.c_void_p
    lib.new_fluid_synth.argtypes = [ctypes.c_void_p]
    synth = lib.new_fluid_synth(settings)
    assert synth, "failed to create fluid synth"
    return synth


def load_sf2(lib, synth, sf2_path):
    lib.fluid_synth_sfload.restype = ctypes.c_int
    lib.fluid_synth_sfload.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
    sfid = lib.fluid_synth_sfload(synth, sf2_path.encode(), 1)
    assert sfid != -1, f"failed to load SF2: {sf2_path}"
    return sfid


def setup_tuning_api(lib):
    # fluid_synth_activate_key_tuning(synth, bank, prog, name, pitch[128 doubles], apply)
    lib.fluid_synth_activate_key_tuning.restype = ctypes.c_int
    lib.fluid_synth_activate_key_tuning.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
    ]
    # fluid_synth_activate_tuning(synth, chan, bank, prog, apply)
    lib.fluid_synth_activate_tuning.restype = ctypes.c_int
    lib.fluid_synth_activate_tuning.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    # fluid_synth_deactivate_tuning(synth, chan, apply)
    lib.fluid_synth_deactivate_tuning.restype = ctypes.c_int
    lib.fluid_synth_deactivate_tuning.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
    ]
    # fluid_synth_noteon(synth, chan, key, vel)
    lib.fluid_synth_noteon.restype = ctypes.c_int
    lib.fluid_synth_noteon.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    # fluid_synth_noteoff(synth, chan, key)
    lib.fluid_synth_noteoff.restype = ctypes.c_int
    lib.fluid_synth_noteoff.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    # fluid_synth_write_float(synth, len, lbuf, loff, lincr, rbuf, roff, rincr)
    lib.fluid_synth_write_float.restype = ctypes.c_int
    lib.fluid_synth_write_float.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
    ]
    # fluid_synth_program_change(synth, chan, prog)
    lib.fluid_synth_program_change.restype = ctypes.c_int
    lib.fluid_synth_program_change.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]


def build_standard_tuning():
    """Return an array of 128 doubles: standard 12-TET cents for each MIDI key (key * 100.0)."""
    arr = (ctypes.c_double * 128)()
    for k in range(128):
        arr[k] = k * 100.0
    return arr


def render_samples(lib, synth, n_samples: int) -> np.ndarray:
    """Render n_samples frames (stereo interleaved float32)."""
    left = (ctypes.c_float * n_samples)()
    right = (ctypes.c_float * n_samples)()
    lib.fluid_synth_write_float(synth, n_samples, left, 0, 1, right, 0, 1)
    l_arr = np.frombuffer(left, dtype=np.float32).copy()
    r_arr = np.frombuffer(right, dtype=np.float32).copy()
    return np.stack([l_arr, r_arr], axis=1)  # (n_samples, 2)


def render_score(
    lib, synth, notes: list, pitch_delta: float = 0.0, quantize: bool = False, release_time: float = 0.08
) -> np.ndarray:
    standard_tuning = build_standard_tuning()
    # Initialize standard tuning preset at bank=0, prog=0
    lib.fluid_synth_activate_key_tuning(synth, 0, 0, b"standard", standard_tuning, 1)

    total_duration = max(n["offset"] for n in notes) + release_time + 0.2
    total_samples = int(math.ceil(total_duration * SAMPLE_RATE))
    output = np.zeros((total_samples, 2), dtype=np.float32)

    cursor = 0  # rendered samples so far

    for note in notes:
        if not note["presence"]:
            continue

        pitch = float(note["pitch"]) + pitch_delta
        if quantize:
            pitch = round(pitch)
        onset_sample = int(note["onset"] * SAMPLE_RATE)
        offset_sample = int(note["offset"] * SAMPLE_RATE)
        # include a short release tail
        end_sample = int((note["offset"] + release_time) * SAMPLE_RATE)
        end_sample = min(end_sample, total_samples)

        base_key = int(round(pitch))
        base_key = max(0, min(127, base_key))
        note_cents = pitch * 100.0

        # Build custom tuning: same as standard but override base_key with exact cents
        tuning = build_standard_tuning()
        tuning[base_key] = note_cents
        lib.fluid_synth_activate_key_tuning(synth, 0, 1, b"note_tuning", tuning, 1)
        lib.fluid_synth_activate_tuning(synth, 0, 0, 1, 1)

        # Render silence up to note onset
        gap = onset_sample - cursor
        if gap > 0:
            pcm = render_samples(lib, synth, gap)
            end = min(cursor + gap, total_samples)
            output[cursor:end] += pcm[: end - cursor]
            cursor += gap

        # Render note on portion
        lib.fluid_synth_noteon(synth, 0, base_key, 100)
        note_len = offset_sample - cursor
        if note_len > 0:
            pcm = render_samples(lib, synth, note_len)
            end = min(cursor + note_len, total_samples)
            output[cursor:end] += pcm[: end - cursor]
            cursor += note_len

        # Note off + render release tail
        lib.fluid_synth_noteoff(synth, 0, base_key)
        tail_len = end_sample - cursor
        if tail_len > 0:
            pcm = render_samples(lib, synth, tail_len)
            end = min(cursor + tail_len, total_samples)
            output[cursor:end] += pcm[: end - cursor]
            cursor += tail_len

        lib.fluid_synth_deactivate_tuning(synth, 0, 1)

    # Render any remaining tail
    remaining = total_samples - cursor
    if remaining > 0:
        pcm = render_samples(lib, synth, remaining)
        output[cursor:] += pcm[:remaining]

    return output


def run_inference(audio_path: str, output_dir: str, language: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    env = os.environ.copy()
    env["RUST_LOG"] = "info"
    cmd = [
        "cargo",
        "run",
        "--release",
        "--manifest-path",
        GAME_MANIFEST,
        "--",
        "-w",
        WEIGHTS,
        "-c",
        CONFIG,
        "-l",
        language,
        "-o",
        output_dir,
        audio_path,
    ]
    result = subprocess.run(cmd, env=env, capture_output=False)
    result.check_returncode()

    stem = os.path.splitext(os.path.basename(audio_path))[0]
    json_path = os.path.join(output_dir, stem + ".json")
    assert os.path.exists(json_path), f"Expected JSON output not found: {json_path}"
    return json_path


def main():
    parser = argparse.ArgumentParser(description="Vocal audio → piano WAV via GAME model")
    parser.add_argument("audio", help="Input vocal audio file")
    parser.add_argument("--language", "-l", default="zh", help="Language code (default: zh)")
    parser.add_argument("--output", "-o", default=None, help="Output WAV path")
    parser.add_argument("--sf2", default=SF2_PATH, help="Path to SoundFont SF2 file")
    parser.add_argument("--score-dir", default=None, help="Directory to save JSON score (default: temp dir)")
    parser.add_argument(
        "--no-align", action="store_true", help="Disable singer pitch alignment (keep raw pitch)"
    )
    parser.add_argument(
        "-q", "--quantize", action="store_true", help="Quantize pitch to nearest semitone before rendering"
    )
    args = parser.parse_args()

    audio_path = os.path.abspath(args.audio)
    assert os.path.exists(audio_path), f"Audio file not found: {audio_path}"

    stem = os.path.splitext(os.path.basename(audio_path))[0]
    out_wav = args.output or os.path.join(SCRIPT_DIR, "output_rs", stem + "_piano.wav")

    sf2_path = args.sf2
    assert os.path.exists(sf2_path), (
        f"SF2 not found: {sf2_path}\n"
        f"Download with: curl -L 'https://github.com/pianobooster/fluid-soundfont/releases/download/v3.1/FluidR3_GM.sf2' -o {SF2_PATH}"
    )

    use_temp = args.score_dir is None
    score_dir = args.score_dir or tempfile.mkdtemp(prefix="game_score_")

    print(f"[1/3] Running GAME inference on {audio_path} ...")
    json_path = run_inference(audio_path, score_dir, args.language)
    print(f"      Score saved to {json_path}")

    with open(json_path) as f:
        score = json.load(f)
    notes = score["notes"]
    present = [n for n in notes if n["presence"]]
    print(f"[2/4] Loaded {len(present)} notes (of {len(notes)} total segments)")

    if args.no_align:
        pitch_delta = 0.0
        print("[3/4] Pitch alignment disabled (delta = 0)")
    else:
        pitch_delta = find_pitch_delta(notes)
        print(f"[3/4] Optimal pitch delta = {pitch_delta:+.4f} semitones ({pitch_delta * 100:+.1f} cents)")

    print("[4/4] Rendering piano audio ...")
    lib = load_fluidsynth_lib()
    settings = make_settings(lib)
    synth = make_synth(lib, settings)
    load_sf2(lib, synth, sf2_path)
    lib.fluid_synth_program_change.restype = ctypes.c_int
    lib.fluid_synth_program_change.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    setup_tuning_api(lib)
    lib.fluid_synth_program_change(synth, 0, PIANO_PROGRAM)

    audio = render_score(lib, synth, notes, pitch_delta=pitch_delta, quantize=args.quantize)

    # Cleanup FluidSynth
    lib.delete_fluid_synth.restype = None
    lib.delete_fluid_synth.argtypes = [ctypes.c_void_p]
    lib.delete_fluid_synth(synth)
    lib.delete_fluid_settings.restype = None
    lib.delete_fluid_settings.argtypes = [ctypes.c_void_p]
    lib.delete_fluid_settings(settings)

    os.makedirs(os.path.dirname(os.path.abspath(out_wav)), exist_ok=True)
    peak = np.abs(audio).max()
    if peak > 0:
        audio = audio * (0.89125 / peak)  # normalize to -1 dBFS
    sf.write(out_wav, audio, SAMPLE_RATE)
    print(f"      Saved to {out_wav}")

    if use_temp:
        import shutil

        shutil.rmtree(score_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
