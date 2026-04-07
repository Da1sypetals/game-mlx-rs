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
}
