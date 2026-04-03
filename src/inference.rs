use std::collections::HashMap;
use std::path::Path;

use anyhow::Result;
use mlx_rs::Array;
use mlx_rs::Dtype;
use mlx_rs::module::{ModuleParameters, ModuleParametersExt};
use mlx_rs::ops::indexing::IndexOp;

use crate::config::{InferenceConfig, ModelConfig, RawConfig};
use crate::d3pm::{d3pm_time_schedule, remove_mutable_boundaries};
use crate::decoding::{decode_gaussian_blurred_probs, decode_soft_boundaries};
use crate::functional::{boundaries_to_regions, format_boundaries, regions_to_durations};
use crate::mel::StretchableMelSpectrogram;
use crate::midi_extraction::SegmentationEstimationModel;

pub struct SegmentationEstimationInferenceModel {
    pub model_config: ModelConfig,
    pub inference_config: InferenceConfig,
    pub timestep: f32,
    pub to_spectrogram: StretchableMelSpectrogram,
    pub model: SegmentationEstimationModel,
}

impl SegmentationEstimationInferenceModel {
    pub fn new(
        model_config: ModelConfig,
        inference_config: InferenceConfig,
    ) -> Result<Self> {
        let timestep = inference_config.features.timestep();
        let feat = &inference_config.features;
        let spec = &feat.spectrogram;

        let to_spectrogram = StretchableMelSpectrogram::new(
            feat.audio_sample_rate,
            spec.num_bins,
            feat.fft_size,
            feat.win_size,
            feat.hop_size,
            spec.fmin,
            spec.fmax,
            1e-5,
        )?;

        let model = SegmentationEstimationModel::new(&model_config)?;

        Ok(Self {
            model_config,
            inference_config,
            timestep,
            to_spectrogram,
            model,
        })
    }

    pub fn forward_encoder(
        &mut self,
        waveform: &Array,
        duration: &Array,
    ) -> Result<(Array, Array, Array)> {
        let b = waveform.dim(0) as i32;
        let mut specs = Vec::with_capacity(b as usize);
        for i in 0..b {
            let wi = waveform.index(i);
            let spec = self.to_spectrogram.forward(&wi)?;
            specs.push(spec);
        }

        let spec_refs: Vec<&Array> = specs.iter().collect();
        let spectrogram = if b == 1 {
            mlx_rs::ops::expand_dims(&specs.into_iter().next().unwrap(), 0)?
        } else {
            mlx_rs::ops::stack_axis(&spec_refs, 0)?
        };
        let spectrogram = spectrogram.transpose_axes(&[0, 2, 1][..])?;

        spectrogram.eval()?;

        let t_len = spectrogram.dim(1) as i32;
        let l = mlx_rs::ops::round(
            &(duration / &Array::from_f32(self.timestep)),
            None,
        )?.as_dtype(Dtype::Int32)?;

        let idx_vec: Vec<i32> = (0..t_len).collect();
        let idx = Array::from_slice(&idx_vec, &[1, t_len]);
        let l_exp = mlx_rs::ops::expand_dims(&l, -1)?;
        let mask = idx.lt(&l_exp)?;

        let (x_seg, x_est) = self.model.forward_encoder(&spectrogram, Some(&mask))?;
        Ok((x_seg, x_est, mask))
    }

    pub fn forward_and_decode_boundaries(
        &mut self,
        x_seg: &Array,
        known_boundaries: &Array,
        prev_boundaries: &Array,
        mask: &Array,
        threshold: f32,
        radius: i32,
        language: Option<&Array>,
        t: Option<&Array>,
    ) -> Result<Array> {
        let b = x_seg.dim(0) as i32;

        let boundaries;
        let t_exp;
        if self.model_config.mode == "d3pm" {
            let t_val = t.unwrap();
            t_exp = mlx_rs::ops::broadcast_to(
                &mlx_rs::ops::expand_dims(t_val, 0)?,
                &[b],
            )?;
            let p = d3pm_time_schedule(&t_exp)?;
            boundaries = remove_mutable_boundaries(prev_boundaries, known_boundaries, &p)?;
        } else {
            t_exp = Array::from_f32(0.0);
            boundaries = known_boundaries.clone();
        }

        let noise = boundaries_to_regions(&boundaries, Some(mask))?;
        let t_arg = if self.model_config.mode == "d3pm" {
            Some(&t_exp)
        } else {
            None
        };
        let (logits, _latent) = self.model.forward_segmentation(
            x_seg,
            &noise,
            t_arg,
            language,
            Some(mask),
        )?;

        let soft_boundaries = mlx_rs::ops::sigmoid(&logits)?;
        let decoded = decode_soft_boundaries(
            &soft_boundaries,
            Some(known_boundaries),
            Some(mask),
            threshold,
            radius,
        )?;
        Ok(decoded)
    }

    pub fn forward_and_decode_scores(
        &mut self,
        x_est: &Array,
        regions: &Array,
        t_mask: &Array,
        n_mask: &Array,
        threshold: f32,
    ) -> Result<(Array, Array)> {
        let logits = self.model.forward_estimation(x_est, regions, t_mask, n_mask)?;
        let probs = mlx_rs::ops::sigmoid(&logits)?;
        let (scores, presence) = decode_gaussian_blurred_probs(
            &probs,
            self.inference_config.midi_min,
            self.inference_config.midi_max,
            self.inference_config.midi_std * 3.0,
            threshold,
        )?;
        let presence = presence.logical_and(n_mask)?;
        let scores = &scores * &presence.as_dtype(Dtype::Float32)?;
        Ok((presence, scores))
    }

    pub fn forward_segmenter_main(
        &mut self,
        x_seg: &Array,
        known_boundaries: &Array,
        mask: &Array,
        threshold: f32,
        radius: i32,
        language: Option<&Array>,
        t: Option<&[f32]>,
    ) -> Result<(Array, Array, i32)> {
        let boundaries;
        if self.model_config.mode == "d3pm" {
            let ts = t.unwrap();
            let mut b = known_boundaries.clone();
            for &ti in ts {
                let t_arr = Array::from_f32(ti);
                b = self.forward_and_decode_boundaries(
                    x_seg,
                    known_boundaries,
                    &b,
                    mask,
                    threshold,
                    radius,
                    language,
                    Some(&t_arr),
                )?;
            }
            boundaries = b;
        } else {
            boundaries = self.forward_and_decode_boundaries(
                x_seg,
                known_boundaries,
                known_boundaries,
                mask,
                threshold,
                radius,
                language,
                None,
            )?;
        }

        let regions = boundaries_to_regions(&boundaries, Some(mask))?;
        let max_n_arr = regions.max(None)?;
        max_n_arr.eval()?;
        let max_n = max_n_arr.item::<i32>();
        Ok((boundaries, regions, max_n))
    }

    pub fn forward_segmenter(
        &mut self,
        x_seg: &Array,
        known_durations: &Array,
        mask: &Array,
        threshold: f32,
        radius: i32,
        language: Option<&Array>,
        t: Option<&[f32]>,
    ) -> Result<(Array, Array, i32)> {
        let t_len = x_seg.dim(1) as i32;
        let known_boundaries = format_boundaries(known_durations, t_len, self.timestep)?;
        let known_boundaries = known_boundaries.logical_and(mask)?;

        let (_boundaries, regions, max_n) = self.forward_segmenter_main(
            x_seg,
            &known_boundaries,
            mask,
            threshold,
            radius,
            language,
            t,
        )?;

        let durations = regions_to_durations(&regions, Some(max_n))?
            .as_dtype(Dtype::Float32)?
            * Array::from_f32(self.timestep);
        Ok((durations, regions, max_n))
    }

    pub fn forward_estimator(
        &mut self,
        x_est: &Array,
        regions: &Array,
        mask: &Array,
        max_n: i32,
        threshold: f32,
    ) -> Result<(Array, Array)> {
        let b = x_est.dim(0) as i32;
        let idx_vec: Vec<i32> = (0..max_n).collect();
        let idx = Array::from_slice(&idx_vec, &[1, max_n]);

        let max_idx = regions.max_axis(-1, Some(true))?;
        let n_mask = idx.lt(&max_idx)?;

        let (presence, scores) = self.forward_and_decode_scores(
            x_est,
            regions,
            mask,
            &n_mask,
            threshold,
        )?;
        Ok((presence, scores))
    }

    pub fn infer(
        &mut self,
        wav: &[f32],
        samplerate: i32,
        language_id: i32,
        ts: &[f32],
        boundary_threshold: f32,
        boundary_radius: i32,
        score_threshold: f32,
    ) -> Result<(Vec<f32>, Vec<bool>, Vec<f32>)> {
        let this = self;

        let dur_sec = wav.len() as f32 / samplerate as f32;

        let waveform = Array::from_slice(wav, &[1, wav.len() as i32]);
        let known_durations = Array::from_slice(&[dur_sec], &[1, 1]);
        let language = Array::from_slice(&[language_id], &[1]);

        let waveform_duration = known_durations.sum_axis(1, None)?;
        let (x_seg, x_est, mask) =
            this.forward_encoder(&waveform, &waveform_duration)?;

        let (durations, regions, max_n) = this.forward_segmenter(
            &x_seg,
            &known_durations,
            &mask,
            boundary_threshold,
            boundary_radius,
            Some(&language),
            Some(ts),
        )?;

        let (presence, scores) = this.forward_estimator(
            &x_est,
            &regions,
            &mask,
            max_n,
            score_threshold,
        )?;

        mlx_rs::transforms::eval([&durations, &presence, &scores])?;

        let durations_flat = durations.index(0);
        let presence_flat = presence.index(0);
        let scores_flat = scores.index(0);

        durations_flat.eval()?;
        presence_flat.eval()?;
        scores_flat.eval()?;

        let n = durations_flat.dim(0) as usize;
        let dur_vec: Vec<f32> = durations_flat.as_slice::<f32>().to_vec();
        let scores_vec: Vec<f32> = scores_flat.as_slice::<f32>().to_vec();
        let presence_vec: Vec<bool> = (0..n)
            .map(|i| presence_flat.index(i as i32).item::<bool>())
            .collect();

        Ok((dur_vec, presence_vec, scores_vec))
    }
}

fn remap_safetensors_key(key: &str) -> String {
    if let Some(rest) = key.strip_prefix("time_embedding.layers.0.") {
        return format!("time_linear1.{rest}");
    }
    if let Some(rest) = key.strip_prefix("time_embedding.layers.2.") {
        return format!("time_linear2.{rest}");
    }
    key.to_string()
}

pub fn load_model(
    weights_path: &Path,
    config_path: &Path,
) -> Result<(SegmentationEstimationInferenceModel, Option<HashMap<String, i32>>)> {
    let raw_config: RawConfig = crate::config::load_config(config_path)?;

    let lang_map = if raw_config.model.use_languages {
        let lang_map_path = config_path.parent().unwrap().join("lang_map.json");
        if lang_map_path.exists() {
            let contents = std::fs::read_to_string(&lang_map_path)?;
            Some(serde_json::from_str(&contents)?)
        } else {
            None
        }
    } else {
        None
    };

    let mut model = SegmentationEstimationInferenceModel::new(
        raw_config.model,
        raw_config.inference,
    )?;

    log::info!("Loading weights from {} ...", weights_path.display());
    let loaded = mlx_rs::Array::load_safetensors(weights_path)
        .map_err(|e| anyhow::anyhow!("Failed to load safetensors: {}", e))?;

    let mut params = model.model.parameters_mut().flatten();
    let mut loaded_count = 0usize;
    let mut missing_keys = Vec::new();
    for (raw_key, value) in &loaded {
        let key = remap_safetensors_key(raw_key);
        if let Some(param) = params.get_mut(&*key) {
            **param = value.clone();
            loaded_count += 1;
        } else {
            missing_keys.push(raw_key.clone());
        }
    }
    log::info!(
        "Loaded {}/{} parameters from safetensors.",
        loaded_count,
        loaded.len()
    );
    if !missing_keys.is_empty() {
        missing_keys.sort();
        log::warn!("Unmatched safetensors keys ({}):", missing_keys.len());
        for k in &missing_keys {
            log::warn!("  {}", k);
        }
    }

    let extra: Vec<_> = {
        let loaded_remapped: std::collections::HashSet<String> =
            loaded.keys().map(|k| remap_safetensors_key(k)).collect();
        params
            .keys()
            .filter(|k| !loaded_remapped.contains(k.as_ref()))
            .cloned()
            .collect()
    };
    if !extra.is_empty() {
        log::warn!("Model parameters not in safetensors ({}):", extra.len());
        for k in &extra {
            log::warn!("  {}", k);
        }
    }

    model.model.eval()
        .map_err(|e| anyhow::anyhow!("Failed to eval model: {}", e))?;
    log::info!("Weights loaded successfully.");

    Ok((model, lang_map))
}
