use anyhow::Result;
use mlx_rs::module::Module;
use mlx_rs::Array;
use mlx_rs::nn;
use mlx_rs::builder::Builder;
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::macros::ModuleParameters;

use crate::common_layers::CyclicRegionEmbedding;
use crate::config::ModelConfig;
use crate::ebf::EBFBackbone;
use crate::jebf::JEBFBackbone;

#[derive(Debug, ModuleParameters)]
pub struct SegmentationEstimationModel {
    pub embedding_dim: i32,
    pub mode: String,
    pub use_language_embedding: bool,

    #[param]
    pub spectrogram_projection: nn::Linear,
    #[param]
    pub encoder: EBFBackbone,

    #[param]
    pub noise_embedding: CyclicRegionEmbedding,
    #[param]
    pub time_linear1: Option<nn::Linear>,
    #[param]
    pub time_linear2: Option<nn::Linear>,

    #[param]
    pub language_embedding: Option<nn::Embedding>,

    #[param]
    pub segmenter: EBFBackbone,

    #[param]
    pub region_embedding: CyclicRegionEmbedding,
    #[param]
    pub estimator: JEBFBackbone,
}

impl SegmentationEstimationModel {
    pub fn new(config: &ModelConfig) -> Result<Self> {
        let dim = config.embedding_dim;

        let spectrogram_projection =
            nn::LinearBuilder::new(config.in_dim, dim).build()?;

        let enc = &config.encoder.kwargs;
        let encoder = EBFBackbone::new(
            dim,
            2 * dim,
            false,
            enc.dim,
            enc.num_layers,
            enc.latent_layer_idx,
            enc.latent_out_dim,
            enc.num_heads,
            enc.head_dim,
            enc.c_kernel_size,
            enc.m_kernel_size,
            enc.use_ls,
            false,
            false,
        )?;

        let noise_embedding =
            CyclicRegionEmbedding::new(dim, config.region_cycle_len)?;

        let (time_linear1, time_linear2) = if config.mode == "d3pm" {
            (
                Some(nn::LinearBuilder::new(1, dim * 4).build()?),
                Some(nn::LinearBuilder::new(dim * 4, dim).build()?),
            )
        } else {
            (None, None)
        };

        let language_embedding = if config.use_languages {
            Some(
                nn::Embedding::new(config.num_languages + 1, dim)
                    .map_err(|e| anyhow::anyhow!("{}", e))?,
            )
        } else {
            None
        };

        let seg = &config.segmenter.kwargs;
        let segmenter = EBFBackbone::new(
            dim,
            1,
            true,
            seg.dim,
            seg.num_layers,
            seg.latent_layer_idx,
            seg.latent_out_dim,
            seg.num_heads,
            seg.head_dim,
            seg.c_kernel_size,
            seg.m_kernel_size,
            seg.use_ls,
            false,
            false,
        )?;

        let region_embedding =
            CyclicRegionEmbedding::new(dim, config.region_cycle_len)?;

        let est = &config.estimator.kwargs;
        let estimator = JEBFBackbone::new(
            dim,
            config.estimator_out_dim,
            est.dim,
            est.num_layers,
            est.num_heads,
            est.head_dim,
            est.region_token_num,
            est.c_kernel_size_pool,
            est.m_kernel_size_pool,
            est.c_kernel_size_x,
            est.m_kernel_size_x,
            est.qk_norm,
            true,
            &est.rope_mode,
            false,
            10000.0,
            est.use_ls,
            false,
            false,
            true,
            None,
            &est.attn_type,
            est.use_region_bias,
            0.5,
            true,
        )?;

        Ok(Self {
            embedding_dim: dim,
            mode: config.mode.clone(),
            use_language_embedding: config.use_languages,
            spectrogram_projection,
            encoder,
            noise_embedding,
            time_linear1,
            time_linear2,
            language_embedding,
            segmenter,
            region_embedding,
            estimator,
        })
    }

    pub fn forward_encoder(
        &mut self,
        spectrogram: &Array,
        mask: Option<&Array>,
    ) -> Result<(Array, Array)> {
        let x = self.spectrogram_projection.forward(spectrogram)?;
        let (x, _latent) = self.encoder.forward(&x, mask)?;
        let dim = self.embedding_dim;
        let x_seg = x.index((.., .., ..dim));
        let x_est = x.index((.., .., dim..));
        Ok((x_seg, x_est))
    }

    pub fn forward_segmentation(
        &mut self,
        x: &Array,
        noise: &Array,
        t: Option<&Array>,
        language: Option<&Array>,
        mask: Option<&Array>,
    ) -> Result<(Array, Option<Array>)> {
        let mut h = x + self.noise_embedding.forward(noise)?;

        if self.mode == "d3pm" {
            let t_val = t.unwrap();
            let t_inp = mlx_rs::ops::expand_dims(
                &mlx_rs::ops::expand_dims(t_val, -1)?,
                -1,
            )?;
            let t_emb = self.time_linear1.as_mut().unwrap().forward(&t_inp)?;
            let t_emb = nn::gelu(&t_emb)?;
            let t_emb = self.time_linear2.as_mut().unwrap().forward(&t_emb)?;
            h = &h + &t_emb;
        }

        if self.use_language_embedding {
            let lang = language.unwrap();
            let lang_exp = mlx_rs::ops::expand_dims(lang, -1)?;
            let lang_emb = self.language_embedding.as_mut().unwrap().forward(&lang_exp)?;
            h = &h + &lang_emb;
        }

        let (out, latent) = self.segmenter.forward(&h, mask)?;
        let out = out.squeeze_axes(&[-1])?;
        Ok((out, latent))
    }

    pub fn forward_estimation(
        &mut self,
        x: &Array,
        regions: &Array,
        t_mask: &Array,
        n_mask: &Array,
    ) -> Result<Array> {
        let h = x + self.region_embedding.forward(regions)?;
        let (_out_x, x_down) = self.estimator.forward(&h, regions, t_mask, n_mask)?;
        Ok(x_down)
    }
}
