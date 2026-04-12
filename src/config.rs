use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct RawConfig {
    pub model: ModelConfig,
    pub inference: InferenceConfig,
}

#[derive(Debug, Deserialize)]
pub struct ModelConfig {
    pub mode: String,
    pub use_languages: bool,
    pub num_languages: i32,
    pub region_cycle_len: i32,
    pub in_dim: i32,
    pub embedding_dim: i32,
    pub estimator_out_dim: i32,
    pub encoder: BackboneSpec,
    pub segmenter: BackboneSpec,
    pub estimator: BackboneSpec,
}

#[derive(Debug, Deserialize)]
pub struct BackboneSpec {
    pub cls: String,
    pub kwargs: BackboneKwargs,
}

#[derive(Debug, Deserialize)]
pub struct BackboneKwargs {
    pub dim: i32,
    pub num_layers: i32,
    pub num_heads: i32,
    pub head_dim: i32,
    pub ffn_type: String,
    #[serde(default)]
    pub ffn_latent_drop: f32,
    #[serde(default)]
    pub ffn_out_drop: f32,
    #[serde(default = "default_true")]
    pub use_ls: bool,
    #[serde(default)]
    pub dropout_attn: f32,
    #[serde(default)]
    pub out_drop: f32,

    // EBF-specific
    #[serde(default = "default_kernel_31")]
    pub c_kernel_size: i32,
    #[serde(default = "default_kernel_31")]
    pub m_kernel_size: i32,
    #[serde(default)]
    pub c_out_drop: f32,
    #[serde(default)]
    pub c_latent_drop: f32,
    #[serde(default)]
    pub latent_layer_idx: Option<i32>,
    #[serde(default = "default_latent_out_dim")]
    pub latent_out_dim: i32,

    // JEBF-specific
    #[serde(default = "default_kernel_7")]
    pub c_kernel_size_pool: i32,
    #[serde(default = "default_kernel_5")]
    pub m_kernel_size_pool: i32,
    #[serde(default = "default_kernel_31")]
    pub c_kernel_size_x: i32,
    #[serde(default = "default_kernel_31")]
    pub m_kernel_size_x: i32,
    #[serde(default)]
    pub qk_norm: bool,
    #[serde(default = "default_attn_type")]
    pub attn_type: String,
    #[serde(default = "default_rope_mode")]
    pub rope_mode: String,
    #[serde(default)]
    pub use_region_bias: bool,
    #[serde(default = "default_one")]
    pub region_token_num: i32,
    #[serde(default = "default_pool_merge_mode")]
    pub pool_merge_mode: String,
}

fn default_true() -> bool {
    true
}
fn default_kernel_31() -> i32 {
    31
}
fn default_kernel_7() -> i32 {
    7
}
fn default_kernel_5() -> i32 {
    5
}
fn default_latent_out_dim() -> i32 {
    16
}
fn default_one() -> i32 {
    1
}
fn default_attn_type() -> String {
    "joint".to_string()
}
fn default_rope_mode() -> String {
    "mixed".to_string()
}
fn default_pool_merge_mode() -> String {
    "mean".to_string()
}

#[derive(Debug, Deserialize)]
pub struct InferenceConfig {
    pub features: FeaturesConfig,
    pub midi_min: f32,
    pub midi_max: f32,
    pub midi_num_bins: i32,
    pub midi_std: f32,
}

#[derive(Debug, Deserialize)]
pub struct FeaturesConfig {
    pub audio_sample_rate: i32,
    pub hop_size: i32,
    pub fft_size: i32,
    pub win_size: i32,
    pub spectrogram: SpectrogramConfig,
}

impl FeaturesConfig {
    pub fn timestep(&self) -> f32 {
        self.hop_size as f32 / self.audio_sample_rate as f32
    }
}

#[derive(Debug, Deserialize)]
pub struct SpectrogramConfig {
    #[serde(rename = "type")]
    pub spec_type: String,
    pub num_bins: i32,
    pub fmin: f32,
    pub fmax: f32,
}

pub fn load_config(path: &std::path::Path) -> anyhow::Result<RawConfig> {
    let contents = std::fs::read_to_string(path)?;
    let config: RawConfig = serde_yaml::from_str(&contents)?;
    Ok(config)
}
