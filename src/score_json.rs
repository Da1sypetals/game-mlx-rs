use anyhow::Result;
use serde::Serialize;
use std::path::Path;

#[derive(Serialize)]
pub struct NoteEntry {
    pub onset: f32,
    pub offset: f32,
    pub pitch: f32,
    pub presence: bool,
}

#[derive(Serialize)]
pub struct ScoreJson {
    pub notes: Vec<NoteEntry>,
}

pub fn save_json(
    output_path: &Path,
    durations: &[f32],
    presence: &[bool],
    scores: &[f32],
) -> Result<()> {
    let mut notes = Vec::with_capacity(durations.len());
    let mut cursor: f32 = 0.0;

    for ((&dur, &pres), &score) in durations.iter().zip(presence.iter()).zip(scores.iter()) {
        let onset = cursor;
        let offset = cursor + dur;
        notes.push(NoteEntry {
            onset,
            offset,
            pitch: score,
            presence: pres,
        });
        cursor = offset;
    }

    let score_json = ScoreJson { notes };
    let file = std::fs::File::create(output_path)?;
    serde_json::to_writer_pretty(file, &score_json)?;
    Ok(())
}
