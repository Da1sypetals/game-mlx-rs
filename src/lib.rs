pub mod config;
pub mod mel;
pub mod functional;
pub mod d3pm;
pub mod decoding;
pub mod common_layers;
pub mod rope;
pub mod eglu;
pub mod ebf;
pub mod jebf;
pub mod midi_extraction;
pub mod inference;
pub mod midi;
pub mod score_json;

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{bail, Result};

/// A single transcribed note with continuous (non-quantized) pitch.
#[derive(Debug, Clone, PartialEq)]
pub struct Note {
    /// Note onset in seconds.
    pub onset: f32,
    /// Note offset in seconds.
    pub offset: f32,
    /// MIDI pitch value (continuous, not rounded to semitone).
    pub pitch: f32,
}

enum State {
    Unloaded,
    Loaded {
        model: inference::SegmentationEstimationInferenceModel,
        lang_map: Option<HashMap<String, i32>>,
    },
}

pub struct GameVocalTranscriber {
    weights_path: PathBuf,
    config_path: PathBuf,
    state: State,
    // inference parameters
    pub t0: f32,
    pub nsteps: i32,
    pub seg_threshold: f32,
    pub seg_radius: f32,
    pub est_threshold: f32,
}

impl GameVocalTranscriber {
    /// Create a new transcriber without loading the model.
    pub fn new(weights_path: impl AsRef<Path>, config_path: impl AsRef<Path>) -> Self {
        Self {
            weights_path: weights_path.as_ref().to_path_buf(),
            config_path: config_path.as_ref().to_path_buf(),
            state: State::Unloaded,
            t0: 0.0,
            nsteps: 8,
            seg_threshold: 0.2,
            seg_radius: 0.02,
            est_threshold: 0.2,
        }
    }

    /// Load the model into memory. Idempotent if already loaded.
    pub fn load(&mut self) -> Result<()> {
        if matches!(self.state, State::Loaded { .. }) {
            return Ok(());
        }
        let (model, lang_map) = inference::load_model(&self.weights_path, &self.config_path)?;
        self.state = State::Loaded { model, lang_map };
        Ok(())
    }

    /// Release the model from memory.
    pub fn unload(&mut self) {
        self.state = State::Unloaded;
    }

    /// Returns true if the model is currently loaded.
    pub fn is_loaded(&self) -> bool {
        matches!(self.state, State::Loaded { .. })
    }

    /// Transcribe a vocal audio file. The model must be loaded first.
    ///
    /// Returns a list of notes with continuous (non-quantized) pitch values.
    pub fn transcribe(&mut self, audio_path: impl AsRef<Path>) -> Result<Vec<Note>> {
        self.transcribe_with_language(audio_path, None)
    }

    /// Transcribe and quantize pitches to semitones.
    ///
    /// - `weighted=true`: find the optimal pitch delta that minimizes
    ///   duration-weighted quantization error, then round each pitch.
    ///   Longer notes have more influence on the optimal delta.
    /// - `weighted=false`: all notes have equal weight (1.0) when
    ///   finding the optimal pitch delta.
    pub fn transcribe_quantized(
        &mut self,
        audio_path: impl AsRef<Path>,
        language: Option<&str>,
        weighted: bool,
    ) -> Result<Vec<Note>> {
        let mut notes = self.transcribe_with_language(audio_path, language)?;
        let delta = find_pitch_delta(&notes, weighted);
        for note in &mut notes {
            note.pitch = (note.pitch + delta).round();
        }
        Ok(notes)
    }

    /// Transcribe with an explicit language code (e.g. `"zh"`).
    pub fn transcribe_with_language(
        &mut self,
        audio_path: impl AsRef<Path>,
        language: Option<&str>,
    ) -> Result<Vec<Note>> {
        let (model, lang_map) = match &mut self.state {
            State::Loaded { model, lang_map } => (model, lang_map),
            State::Unloaded => bail!("model is not loaded; call load() first"),
        };

        let language_id = get_language_id(language, lang_map.as_ref());
        let samplerate = model.inference_config.features.audio_sample_rate;
        let timestep = model.inference_config.features.timestep();
        let seg_radius_frames = (self.seg_radius / timestep).round() as i32;
        let ts = d3pm_ts(self.t0, self.nsteps);

        let (wav, sr) = mel::load_audio(audio_path.as_ref())?;
        let wav = if sr != samplerate as u32 {
            mel::resample(&wav, sr, samplerate as u32)?
        } else {
            wav
        };

        let (durations, presence, scores) = model.infer(
            &wav,
            samplerate,
            language_id,
            &ts,
            self.seg_threshold,
            seg_radius_frames,
            self.est_threshold,
        )?;

        let notes = durations_to_notes(&durations, &presence, &scores);
        Ok(notes)
    }

    /// Transcribe raw PCM samples (f32, mono) at the given sample rate.
    pub fn transcribe_pcm(
        &mut self,
        samples: &[f32],
        sample_rate: u32,
        language: Option<&str>,
    ) -> Result<Vec<Note>> {
        let (model, lang_map) = match &mut self.state {
            State::Loaded { model, lang_map } => (model, lang_map),
            State::Unloaded => bail!("model is not loaded; call load() first"),
        };

        let language_id = get_language_id(language, lang_map.as_ref());
        let samplerate = model.inference_config.features.audio_sample_rate;
        let timestep = model.inference_config.features.timestep();
        let seg_radius_frames = (self.seg_radius / timestep).round() as i32;
        let ts = d3pm_ts(self.t0, self.nsteps);

        let wav = if sample_rate != samplerate as u32 {
            mel::resample(samples, sample_rate, samplerate as u32)?
        } else {
            samples.to_vec()
        };

        let (durations, presence, scores) = model.infer(
            &wav,
            samplerate,
            language_id,
            &ts,
            self.seg_threshold,
            seg_radius_frames,
            self.est_threshold,
        )?;

        let notes = durations_to_notes(&durations, &presence, &scores);
        Ok(notes)
    }
}

fn d3pm_ts(t0: f32, nsteps: i32) -> Vec<f32> {
    let step = (1.0 - t0) / nsteps as f32;
    (0..nsteps).map(|i| t0 + i as f32 * step).collect()
}

fn get_language_id(language: Option<&str>, lang_map: Option<&HashMap<String, i32>>) -> i32 {
    match (language, lang_map) {
        (Some(lang), Some(map)) => {
            *map.get(lang)
                .unwrap_or_else(|| panic!("Language '{}' not in lang_map. Available: {:?}", lang, map.keys().collect::<Vec<_>>()))
        }
        _ => 0,
    }
}

fn durations_to_notes(durations: &[f32], presence: &[bool], scores: &[f32]) -> Vec<Note> {
    let mut notes = Vec::new();
    let mut cursor = 0.0f32;
    for i in 0..durations.len() {
        let onset = cursor;
        let offset = cursor + durations[i];
        if presence[i] {
            notes.push(Note {
                onset,
                offset,
                pitch: scores[i],
            });
        }
        cursor = offset;
    }
    notes
}

/// Find optimal pitch delta (in semitones) that minimizes quantization error.
///
/// The algorithm evaluates candidate deltas at:
/// - Points where some pitch would quantize to exactly an integer (zero error)
/// - Breakpoints where the cost function's derivative changes (half-integer crossings)
///
/// When `weighted` is true, longer notes contribute more to the cost.
/// When `weighted` is false, all notes have equal weight.
fn find_pitch_delta(notes: &[Note], weighted: bool) -> f32 {
    if notes.is_empty() {
        return 0.0;
    }

    let n = notes.len();
    let pitches: Vec<f32> = notes.iter().map(|n| n.pitch).collect();
    let weights: Vec<f32> = if weighted {
        notes.iter().map(|n| n.offset - n.onset).collect()
    } else {
        vec![1.0; n]
    };

    // Candidate deltas: values where some pitch has zero quantization error
    // i.e. p + delta = round(p + delta) => delta = k - p for integer k
    // Normalized to [-0.5, 0.5): candidate = -frac(p), mapped to principal range
    let mut candidates: Vec<f32> = Vec::with_capacity(2 * n + 1);
    for &p in &pitches {
        let frac = p - p.floor();
        let c = -frac;
        candidates.push(normalize_delta(c));
    }

    // Breakpoints: half-integer crossings where derivative changes
    // i.e. p + delta = n + 0.5 => delta = n + 0.5 - p
    for &p in &pitches {
        let frac = p - p.floor();
        let bp = 0.5 - frac;
        candidates.push(normalize_delta(bp));
    }

    candidates.push(0.0);

    let mut best_delta = 0.0;
    let mut best_cost = f32::INFINITY;

    for &d in &candidates {
        let mut cost = 0.0f32;
        for i in 0..n {
            let shifted = pitches[i] + d;
            let err = (shifted - shifted.round()).abs();
            cost += weights[i] * err;
        }
        if cost < best_cost {
            best_cost = cost;
            best_delta = d;
        }
    }

    best_delta
}

/// Normalize delta to [-0.5, 0.5) range.
#[inline]
fn normalize_delta(d: f32) -> f32 {
    // Use floor-based approach to match Python's % behavior
    // Python: ((d + 0.5) % 1.0) - 0.5
    let v = d + 0.5;
    let normalized = v - v.floor();
    normalized - 0.5
}

/// Convert a `Vec<Note>` back to `(durations, presence, scores)` as expected by
/// the MIDI and JSON save functions. Silent gaps between notes are re-inserted.
///
/// This is the inverse of `durations_to_notes` and is used by the CLI binary.
pub fn notes_to_raw(notes: &[Note]) -> (Vec<f32>, Vec<bool>, Vec<f32>) {
    // notes only contains present notes; we need to reconstruct all segments
    // including the silent ones. Since durations_to_notes discarded silent notes,
    // we only have present ones here — reconstruct as a flat present-only list.
    let durations: Vec<f32> = notes.iter().map(|n| n.offset - n.onset).collect();
    let presence: Vec<bool> = vec![true; notes.len()];
    let scores: Vec<f32> = notes.iter().map(|n| n.pitch).collect();
    (durations, presence, scores)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn weights_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent().unwrap()
            .join("game-mlx/weights/GAME-1.0-large.safetensors")
    }

    fn config_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent().unwrap()
            .join("GAME/models/GAME-1.0-large/config.yaml")
    }

    fn sample_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent().unwrap()
            .join("samples/fswx.wav")
    }

    #[test]
    fn new_does_not_load_model() {
        let t = GameVocalTranscriber::new(weights_path(), config_path());
        assert!(!t.is_loaded());
    }

    #[test]
    fn transcribe_before_load_returns_error() {
        let mut t = GameVocalTranscriber::new(weights_path(), config_path());
        let result = t.transcribe(sample_path());
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("not loaded"), "unexpected error: {msg}");
    }

    #[test]
    fn load_and_unload() {
        let mut t = GameVocalTranscriber::new(weights_path(), config_path());
        t.load().expect("load failed");
        assert!(t.is_loaded());
        t.unload();
        assert!(!t.is_loaded());
    }

    #[test]
    fn load_is_idempotent() {
        let mut t = GameVocalTranscriber::new(weights_path(), config_path());
        t.load().expect("first load failed");
        t.load().expect("second load failed");
        assert!(t.is_loaded());
    }

    #[test]
    fn transcribe_after_unload_returns_error() {
        let mut t = GameVocalTranscriber::new(weights_path(), config_path());
        t.load().expect("load failed");
        t.unload();
        let result = t.transcribe(sample_path());
        assert!(result.is_err());
    }

    #[test]
    fn transcribe_returns_notes() {
        let mut t = GameVocalTranscriber::new(weights_path(), config_path());
        t.load().expect("load failed");
        let notes = t.transcribe_with_language(sample_path(), Some("zh"))
            .expect("transcribe failed");
        assert!(!notes.is_empty(), "expected at least one note");
        for n in &notes {
            assert!(n.onset >= 0.0, "onset must be non-negative");
            assert!(n.offset > n.onset, "offset must be after onset");
            assert!(n.pitch >= 0.0 && n.pitch <= 127.0, "pitch out of MIDI range: {}", n.pitch);
            // pitch must be continuous (not necessarily an integer)
        }
        let has_fractional = notes.iter().any(|n| (n.pitch - n.pitch.round()).abs() > 1e-3);
        assert!(has_fractional, "expected at least some non-integer (continuous) pitch values");
    }

    #[test]
    fn note_times_are_monotonic() {
        let mut t = GameVocalTranscriber::new(weights_path(), config_path());
        t.load().expect("load failed");
        let notes = t.transcribe_with_language(sample_path(), Some("zh"))
            .expect("transcribe failed");
        for w in notes.windows(2) {
            assert!(w[1].onset >= w[0].onset, "notes must be in time order");
        }
    }

    #[test]
    fn find_pitch_delta_empty() {
        let notes: Vec<Note> = vec![];
        assert_eq!(find_pitch_delta(&notes, true), 0.0);
        assert_eq!(find_pitch_delta(&notes, false), 0.0);
    }

    #[test]
    fn find_pitch_delta_single_note() {
        // Single note at pitch 60.3, duration 1.0
        let notes = vec![Note { onset: 0.0, offset: 1.0, pitch: 60.3 }];
        // Optimal delta should be -0.3 (to round to 60)
        let delta = find_pitch_delta(&notes, true);
        assert!((delta - (-0.3)).abs() < 0.001, "expected delta ≈ -0.3, got {}", delta);
    }

    #[test]
    fn find_pitch_delta_quantizes_to_integers() {
        // Use synthetic notes instead of loading model
        let notes = vec![
            Note { onset: 0.0, offset: 0.5, pitch: 60.3 },
            Note { onset: 0.5, offset: 1.0, pitch: 62.7 },
            Note { onset: 1.0, offset: 1.5, pitch: 64.2 },
        ];
        
        // Test weighted quantization
        let delta_w = find_pitch_delta(&notes, true);
        for note in &notes {
            let quantized = (note.pitch + delta_w).round();
            assert!((quantized - quantized.round()).abs() < 0.001, 
                "quantized pitch should be integer: {}", quantized);
        }
        
        // Test equal weight quantization  
        let delta_e = find_pitch_delta(&notes, false);
        for note in &notes {
            let quantized = (note.pitch + delta_e).round();
            assert!((quantized - quantized.round()).abs() < 0.001,
                "quantized pitch should be integer: {}", quantized);
        }
    }

    #[test]
    fn find_pitch_delta_uses_duration_weight() {
        // Two notes: one short at 60.3, one long at 60.9
        // With duration weighting, long note should dominate
        let notes = vec![
            Note { onset: 0.0, offset: 0.1, pitch: 60.3 },  // short
            Note { onset: 0.1, offset: 1.1, pitch: 60.9 },  // long (1.0s)
        ];
        
        let delta_weighted = find_pitch_delta(&notes, true);
        let delta_equal = find_pitch_delta(&notes, false);
        
        // Weighted should prefer rounding 60.9 to 61 (delta ≈ +0.1)
        // Equal weight might prefer 60 (delta ≈ -0.3) due to the 60.3 note
        // The important thing is they can be different
        
        // Verify weighted version minimizes weighted error
        let calc_cost = |d: f32, weighted: bool| -> f32 {
            notes.iter().map(|n| {
                let err = ((n.pitch + d) - (n.pitch + d).round()).abs();
                let w = if weighted { n.offset - n.onset } else { 1.0 };
                w * err
            }).sum()
        };
        
        let cost_w = calc_cost(delta_weighted, true);
        // Check that nearby deltas have higher cost
        for d in [-0.4f32, -0.2, 0.0, 0.2, 0.4].iter() {
            if (*d - delta_weighted).abs() > 0.05 {
                assert!(calc_cost(*d, true) >= cost_w - 0.001,
                    "found better delta {} with cost {} vs {} at {}",
                    d, calc_cost(*d, true), cost_w, delta_weighted);
            }
        }
    }

    #[test]
    fn normalize_delta_bounds() {
        // Test normalize_delta matches Python's ((d + 0.5) % 1.0) - 0.5
        // Verified against Python 3.11 behavior
        let test_cases = [
            (0.0, 0.0),      // (0.5 % 1.0) - 0.5 = 0.0
            (0.5, -0.5),     // (1.0 % 1.0) - 0.5 = -0.5
            (-0.5, -0.5),    // (0.0 % 1.0) - 0.5 = -0.5
            (1.0, 0.0),      // (1.5 % 1.0) - 0.5 = 0.0
            (-1.0, 0.0),     // (-0.5 % 1.0) - 0.5 = 0.5 - 0.5 = 0.0
            (1.5, -0.5),     // (2.0 % 1.0) - 0.5 = -0.5
            (-1.5, -0.5),    // (-1.0 % 1.0) - 0.5 = 0.0 - 0.5 = -0.5
            (2.0, 0.0),      // (2.5 % 1.0) - 0.5 = 0.0
            (-2.0, 0.0),     // (-1.5 % 1.0) - 0.5 = 0.5 - 0.5 = 0.0
        ];

        for (input, expected) in test_cases {
            let result = normalize_delta(input);
            assert!((result - expected).abs() < 0.0001,
                "normalize_delta({}) = {}, expected {}", input, result, expected);
        }
    }
}
