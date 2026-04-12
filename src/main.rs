use game_mlxrs::{GameVocalTranscriber, midi, score_json};

use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, ValueEnum};

#[derive(Debug, Clone, ValueEnum, PartialEq, Eq)]
enum OutputFormat {
    Json,
    Midi,
}

impl std::fmt::Display for OutputFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OutputFormat::Json => write!(f, "json"),
            OutputFormat::Midi => write!(f, "midi"),
        }
    }
}

#[derive(Parser, Debug)]
#[command(name = "game-mlxrs", about = "GAME MIDI extraction in Rust + MLX")]
struct Cli {
    /// Audio file or directory path
    path: PathBuf,

    /// Path to MLX .safetensors weights file
    #[arg(short = 'w', long)]
    weights: PathBuf,

    /// Path to GAME config.yaml
    #[arg(short = 'c', long)]
    config: PathBuf,

    /// Language code (e.g. zh)
    #[arg(short = 'l', long)]
    language: Option<String>,

    /// Output directory
    #[arg(short = 'o', long, default_value = "output")]
    output_dir: PathBuf,

    /// Output format(s): json, midi (can be specified multiple times)
    #[arg(short = 'f', long = "output-format", value_enum, default_values = ["json", "midi"])]
    output_formats: Vec<OutputFormat>,

    /// D3PM start T
    #[arg(long, default_value_t = 0.0)]
    t0: f32,

    /// D3PM steps
    #[arg(long, default_value_t = 8)]
    nsteps: i32,

    /// Segmentation boundary threshold
    #[arg(long, default_value_t = 0.2)]
    seg_threshold: f32,

    /// Segmentation boundary radius (seconds)
    #[arg(long, default_value_t = 0.02)]
    seg_radius: f32,

    /// Estimation score threshold
    #[arg(long, default_value_t = 0.2)]
    est_threshold: f32,

    /// MIDI tempo (BPM)
    #[arg(long, default_value_t = 120.0)]
    tempo: f32,

    /// Comma-separated input audio formats
    #[arg(long, default_value = "wav,flac,mp3,aac,ogg,m4a")]
    input_formats: String,

    /// Quantize pitches to semitones (uses duration-weighted optimization)
    #[arg(long)]
    quantize: bool,

    /// Use equal weighting for quantization (ignored if --quantize is not set)
    #[arg(long)]
    quantize_equal_weight: bool,
}

fn collect_audio_files(
    path: &std::path::Path,
    extensions: &std::collections::HashSet<String>,
) -> Vec<PathBuf> {
    if path.is_file() {
        return vec![path.to_path_buf()];
    }
    let mut files = Vec::new();
    for entry in walkdir::WalkDir::new(path)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        let p = entry.path();
        if p.is_file() {
            if let Some(ext) = p.extension() {
                if extensions.contains(&ext.to_string_lossy().to_lowercase()) {
                    files.push(p.to_path_buf());
                }
            }
        }
    }
    files
}

fn main() -> Result<()> {
    env_logger::init();
    let cli = Cli::parse();

    let extensions: std::collections::HashSet<String> = cli
        .input_formats
        .split(',')
        .map(|s| s.trim().to_lowercase())
        .collect();

    let audio_files = collect_audio_files(&cli.path, &extensions);
    if audio_files.is_empty() {
        anyhow::bail!("No audio files found at {:?}", cli.path);
    }

    std::fs::create_dir_all(&cli.output_dir)?;

    let mut transcriber = GameVocalTranscriber::new(&cli.weights, &cli.config);
    transcriber.t0 = cli.t0;
    transcriber.nsteps = cli.nsteps;
    transcriber.seg_threshold = cli.seg_threshold;
    transcriber.seg_radius = cli.seg_radius;
    transcriber.est_threshold = cli.est_threshold;
    transcriber.load()?;

    let language = cli.language.as_deref();

    for audio_path in &audio_files {
        log::info!("Processing {} ...", audio_path.display());

        let infer_start = std::time::Instant::now();
        let notes = if cli.quantize {
            let weighted = !cli.quantize_equal_weight;
            transcriber.transcribe_quantized(audio_path, language, weighted)?
        } else {
            transcriber.transcribe_with_language(audio_path, language)?
        };
        let infer_secs = infer_start.elapsed().as_secs_f64();

        if std::env::var_os("GAME_BENCHMARK").is_some() {
            eprintln!(
                "GAME_BENCHMARK\t{}\t{:.6}",
                audio_path.file_name().unwrap().to_string_lossy(),
                infer_secs
            );
        }

        // Convert notes back to raw (durations, presence, scores) for existing save functions
        let (durations, presence, scores) = game_mlxrs::notes_to_raw(&notes);

        if cli.output_formats.contains(&OutputFormat::Json) {
            let json_path = cli.output_dir.join(
                audio_path
                    .file_stem()
                    .unwrap()
                    .to_string_lossy()
                    .to_string()
                    + ".json",
            );
            score_json::save_json(&json_path, &durations, &presence, &scores)?;
            log::info!("  -> Saved to {}", json_path.display());
        }

        if cli.output_formats.contains(&OutputFormat::Midi) {
            let midi_path = cli.output_dir.join(
                audio_path
                    .file_stem()
                    .unwrap()
                    .to_string_lossy()
                    .to_string()
                    + ".mid",
            );
            midi::save_midi(&midi_path, &durations, &presence, &scores, cli.tempo)?;
            log::info!("  -> Saved to {}", midi_path.display());
        }
    }

    log::info!("Done.");
    Ok(())
}
