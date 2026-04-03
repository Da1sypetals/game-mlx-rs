use std::path::Path;

use anyhow::Result;
use mlx_rs::Array;
use mlx_rs::ops::indexing::{IndexOp, IntoStrideBy};

pub fn load_audio(path: &Path) -> Result<(Vec<f32>, u32)> {
    let ext = path
        .extension()
        .map(|e| e.to_string_lossy().to_lowercase())
        .unwrap_or_default();

    if ext == "wav" {
        let reader = hound::WavReader::open(path)?;
        let spec = reader.spec();
        let sr = spec.sample_rate;
        let samples: Vec<f32> = match spec.sample_format {
            hound::SampleFormat::Int => {
                let max_val = (1i64 << (spec.bits_per_sample - 1)) as f32;
                reader
                    .into_samples::<i32>()
                    .map(|s| s.unwrap() as f32 / max_val)
                    .collect()
            }
            hound::SampleFormat::Float => {
                reader.into_samples::<f32>().map(|s| s.unwrap()).collect()
            }
        };
        let mono = if spec.channels > 1 {
            samples
                .chunks(spec.channels as usize)
                .map(|ch| ch.iter().sum::<f32>() / spec.channels as f32)
                .collect()
        } else {
            samples
        };
        return Ok((mono, sr));
    }

    use symphonia::core::audio::SampleBuffer;
    use symphonia::core::codecs::DecoderOptions;
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;
    use symphonia::core::probe::Hint;

    let file = std::fs::File::open(path)?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());
    let mut hint = Hint::new();
    if !ext.is_empty() {
        hint.with_extension(&ext);
    }
    let probed = symphonia::default::get_probe().format(
        &hint,
        mss,
        &FormatOptions::default(),
        &MetadataOptions::default(),
    )?;
    let mut format = probed.format;
    let track = format
        .default_track()
        .ok_or_else(|| anyhow::anyhow!("no default track"))?;
    let codec_params = track.codec_params.clone();
    let sr = codec_params.sample_rate.unwrap_or(44100);
    let channels = codec_params.channels.map(|c| c.count()).unwrap_or(1);
    let track_id = track.id;

    let mut decoder = symphonia::default::get_codecs()
        .make(&codec_params, &DecoderOptions::default())?;

    let mut all_samples: Vec<f32> = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(symphonia::core::errors::Error::IoError(ref e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(e) => return Err(e.into()),
        };
        if packet.track_id() != track_id {
            continue;
        }
        let decoded = decoder.decode(&packet)?;
        let spec = *decoded.spec();
        let n_frames = decoded.capacity();
        let mut sample_buf = SampleBuffer::<f32>::new(n_frames as u64, spec);
        sample_buf.copy_interleaved_ref(decoded);
        all_samples.extend_from_slice(sample_buf.samples());
    }

    let mono = if channels > 1 {
        all_samples
            .chunks(channels)
            .map(|ch| ch.iter().sum::<f32>() / channels as f32)
            .collect()
    } else {
        all_samples
    };

    Ok((mono, sr))
}

pub fn resample(wav: &[f32], from_sr: u32, to_sr: u32) -> Result<Vec<f32>> {
    use rubato::{Fft, FixedSync, Resampler};
    use rubato::audioadapter_buffers::direct::SequentialSliceOfVecs;

    let chunk_size = 1024;
    let mut resampler = Fft::<f32>::new(
        from_sr as usize,
        to_sr as usize,
        chunk_size,
        1,
        1,
        FixedSync::Input,
    )?;

    let input_len = wav.len();
    let output_len = resampler.process_all_needed_output_len(input_len);

    let input_data = vec![wav.to_vec()];
    let input = SequentialSliceOfVecs::new(&input_data, 1, input_len)?;

    let mut output_data = vec![vec![0.0f32; output_len]; 1];
    let mut output = SequentialSliceOfVecs::new_mut(&mut output_data, 1, output_len)?;

    let (_nbr_in, nbr_out) = resampler.process_all_into_buffer(&input, &mut output, input_len, None)?;

    output_data[0].truncate(nbr_out);
    Ok(output_data.into_iter().next().unwrap())
}

fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0)
}

pub fn mel_filterbank(sr: i32, n_fft: i32, n_mels: i32, fmin: f32, fmax: f32) -> Result<Array> {
    let n_freqs = n_fft / 2 + 1;

    let fmin_mel = hz_to_mel(fmin);
    let fmax_mel = hz_to_mel(fmax);

    let mel_points: Vec<f32> = (0..n_mels + 2)
        .map(|i| fmin_mel + (fmax_mel - fmin_mel) * i as f32 / (n_mels + 1) as f32)
        .collect();

    let hz_points: Vec<f32> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();

    let fft_freqs: Vec<f32> = (0..n_freqs)
        .map(|i| i as f32 * sr as f32 / n_fft as f32)
        .collect();

    let mut weights = vec![0.0f32; (n_mels * n_freqs) as usize];

    for m in 0..n_mels as usize {
        let f_left = hz_points[m];
        let f_center = hz_points[m + 1];
        let f_right = hz_points[m + 2];

        for k in 0..n_freqs as usize {
            let f = fft_freqs[k];
            if f >= f_left && f <= f_center {
                weights[m * n_freqs as usize + k] =
                    (f - f_left) / (f_center - f_left + 1e-10);
            } else if f > f_center && f <= f_right {
                weights[m * n_freqs as usize + k] =
                    (f_right - f) / (f_right - f_center + 1e-10);
            }
        }
    }

    let enorm: Vec<f32> = (0..n_mels as usize)
        .map(|m| 2.0 / (hz_points[m + 2] - hz_points[m] + 1e-10))
        .collect();
    for m in 0..n_mels as usize {
        for k in 0..n_freqs as usize {
            weights[m * n_freqs as usize + k] *= enorm[m];
        }
    }

    Ok(Array::from_slice(&weights, &[n_mels, n_freqs]))
}

/// Reflect-pad a 1D array [L] along last dim.
fn reflect_pad_1d(y: &Array, pad_before: i32, pad_after: i32) -> Result<Array> {
    let len = y.dim(-1) as i32;
    let mut parts: Vec<Array> = Vec::new();

    if pad_before > 0 {
        // y[1:pad_before+1] reversed via stride_by(-1)
        let before = y.index((1..pad_before + 1).stride_by(1));
        let before = before.index((..).stride_by(-1));
        parts.push(before);
    }

    parts.push(y.clone());

    if pad_after > 0 {
        let start = len - pad_after - 1;
        let end = len - 1;
        let after = y.index((start..end).stride_by(1));
        let after = after.index((..).stride_by(-1));
        parts.push(after);
    }

    let refs: Vec<&Array> = parts.iter().collect();
    Ok(mlx_rs::ops::concatenate_axis(&refs, -1)?)
}

fn hann_window(size: i32) -> Result<Array> {
    let n: Vec<f32> = (0..size).map(|i| i as f32).collect();
    let n = Array::from_slice(&n, &[size]);
    let two_pi_over_nm1 = 2.0 * std::f32::consts::PI / (size - 1) as f32;
    let arg = &n * Array::from_f32(two_pi_over_nm1);
    let cos_val = mlx_rs::ops::cos(&arg)?;
    let window = Array::from_f32(0.5) - &cos_val * Array::from_f32(0.5);
    Ok(window)
}

/// STFT of a 1D signal: returns magnitude spectrogram [n_frames, n_fft/2+1]
fn stft_forward(
    y: &Array,
    n_fft: i32,
    hop_length: i32,
    win_length: i32,
    window: &Array,
) -> Result<Array> {
    let len = y.dim(-1) as i32;
    let n_frames = (len - win_length) / hop_length + 1;

    // Frame the signal using as_strided: [n_frames, win_length]
    let y_strided = mlx_rs::ops::as_strided(
        y,
        &[n_frames, win_length],
        &[hop_length as i64, 1],
        0,
    )?;

    // Apply window
    let windowed = &y_strided * window;

    // Zero-pad to n_fft if needed
    let padded = if win_length < n_fft {
        let pad_amount = n_fft - win_length;
        mlx_rs::ops::pad(
            &windowed,
            &[(0, 0), (0, pad_amount)][..],
            None::<Array>,
            None,
        )?
    } else {
        windowed
    };

    // rfft along last axis
    let spec = mlx_rs::fft::rfft(&padded, None::<i32>, Some(-1))?;

    // Compute magnitude: |spec| = sqrt(real^2 + imag^2)
    let real = spec.real()?;
    let imag = spec.imag()?;
    let magnitude = mlx_rs::ops::sqrt(&(&(&real * &real) + &(&imag * &imag)))?;

    Ok(magnitude)
}

pub struct StretchableMelSpectrogram {
    pub sample_rate: i32,
    pub n_mels: i32,
    pub n_fft: i32,
    pub win_size: i32,
    pub hop_length: i32,
    pub fmin: f32,
    pub fmax: f32,
    pub clip_val: f32,
    pub mel_basis: Array,
}

impl StretchableMelSpectrogram {
    pub fn new(
        sample_rate: i32,
        n_mels: i32,
        n_fft: i32,
        win_length: i32,
        hop_length: i32,
        fmin: f32,
        fmax: f32,
        clip_val: f32,
    ) -> Result<Self> {
        let mel_basis = mel_filterbank(sample_rate, n_fft, n_mels, fmin, fmax)?;
        Ok(Self {
            sample_rate,
            n_mels,
            n_fft,
            win_size: win_length,
            hop_length,
            fmin,
            fmax,
            clip_val,
            mel_basis,
        })
    }

    /// Forward pass: waveform [L] -> mel spectrogram [n_mels, T]
    pub fn forward(&self, y: &Array) -> Result<Array> {
        let win_length = self.win_size;
        let hop_length = self.hop_length;
        let n_fft = self.n_fft;

        let pad_before = (win_length - hop_length) / 2;
        let pad_after = (win_length - hop_length + 1) / 2;

        let y_1d = if y.ndim() == 2 {
            y.index(0)
        } else {
            y.clone()
        };

        let y_padded = reflect_pad_1d(&y_1d, pad_before, pad_after)?;

        let window = hann_window(win_length)?;
        let spec = stft_forward(&y_padded, n_fft, hop_length, win_length, &window)?;
        let spec_t = spec.transpose_axes(&[1, 0][..])?;

        let mel_spec = self.mel_basis.matmul(&spec_t)?;

        let clip_val_arr = Array::from_f32(self.clip_val);
        let clamped = mlx_rs::ops::maximum(&mel_spec, &clip_val_arr)?;
        let log_spec = mlx_rs::ops::log(&clamped)?;

        Ok(log_spec)
    }
}
