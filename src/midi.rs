use anyhow::Result;
use midly::num::{u4, u7, u15, u24, u28};
use midly::{Format, Header, MetaMessage, MidiMessage, Smf, Timing, TrackEvent, TrackEventKind};
use std::path::Path;

pub fn save_midi(
    output_path: &Path,
    durations: &[f32],
    presence: &[bool],
    scores: &[f32],
    tempo: f32,
) -> Result<()> {
    let ticks_per_beat: u16 = 480;
    let sec_per_beat = 60.0 / tempo;
    let ticks_per_sec = ticks_per_beat as f64 / sec_per_beat as f64;

    let us_per_beat = (60_000_000.0 / tempo as f64).round() as u32;

    let mut events: Vec<TrackEvent<'static>> = Vec::new();
    events.push(TrackEvent {
        delta: u28::new(0),
        kind: TrackEventKind::Meta(MetaMessage::Tempo(u24::new(us_per_beat))),
    });

    let mut last_tick: i64 = 0;
    let mut current_tick: i64 = 0;

    for ((&dur, &pres), &score) in durations.iter().zip(presence.iter()).zip(scores.iter()) {
        let note_ticks = {
            let t = (dur as f64 * ticks_per_sec).round() as i64;
            if t <= 0 { 1 } else { t }
        };

        if !pres {
            current_tick += note_ticks;
            continue;
        }

        let pitch = (score.round() as i32).clamp(0, 127) as u8;

        let delta_on = (current_tick - last_tick) as u32;
        events.push(TrackEvent {
            delta: u28::new(delta_on),
            kind: TrackEventKind::Midi {
                channel: u4::new(0),
                message: MidiMessage::NoteOn {
                    key: u7::new(pitch),
                    vel: u7::new(64),
                },
            },
        });

        events.push(TrackEvent {
            delta: u28::new(note_ticks as u32),
            kind: TrackEventKind::Midi {
                channel: u4::new(0),
                message: MidiMessage::NoteOff {
                    key: u7::new(pitch),
                    vel: u7::new(64),
                },
            },
        });

        last_tick = current_tick + note_ticks;
        current_tick += note_ticks;
    }

    events.push(TrackEvent {
        delta: u28::new(0),
        kind: TrackEventKind::Meta(MetaMessage::EndOfTrack),
    });

    let smf = Smf {
        header: Header {
            format: Format::SingleTrack,
            timing: Timing::Metrical(u15::new(ticks_per_beat)),
        },
        tracks: vec![events],
    };

    smf.save(output_path)?;
    Ok(())
}
