use anyhow::Result;
use mlx_rs::module::{Module, Param};
use mlx_rs::error::Exception;
use mlx_rs::Array;
use mlx_rs::Dtype;
use mlx_rs::nn;
use mlx_rs::builder::Builder;
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::macros::ModuleParameters;

// ---------------------------------------------------------------------------
// LayScale
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters)]
pub struct LayScale {
    #[param]
    pub scale: Param<Array>,
    pub dim: i32,
}

impl LayScale {
    pub fn new(dim: i32, init_value: f32) -> Self {
        let scale = mlx_rs::ops::ones::<f32>(&[dim]).unwrap() * Array::from_f32(init_value);
        Self {
            scale: Param::new(scale),
            dim,
        }
    }

    pub fn forward(&self, x: &Array) -> Result<Array> {
        let n_dim = x.ndim();
        if n_dim == 1 {
            return Ok(x * &*self.scale);
        }
        let mut shape = vec![1i32; n_dim - 1];
        shape.push(self.dim);
        let s = self.scale.reshape(&shape)?;
        Ok(x * &s)
    }
}

// ---------------------------------------------------------------------------
// RMSNorm (custom init_num, NOT using mlx_rs::fast::rms_norm)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters)]
pub struct RMSNorm {
    #[param]
    pub weight: Param<Array>,
    pub eps: f32,
}

impl RMSNorm {
    pub fn new(dim: i32, init_num: f32, eps: f32) -> Self {
        let weight = mlx_rs::ops::ones::<f32>(&[dim]).unwrap() * Array::from_f32(init_num);
        Self {
            weight: Param::new(weight),
            eps,
        }
    }

    pub fn forward(&self, x: &Array) -> Result<Array> {
        let x_f = x.as_dtype(Dtype::Float32)?;
        let sq = &x_f * &x_f;
        let mean_sq = sq.mean_axis(-1, Some(true))?;
        let norm = &x_f * mlx_rs::ops::rsqrt(&(&mean_sq + Array::from_f32(self.eps)))?;
        let out = &norm * &*self.weight;
        Ok(out.as_dtype(x.dtype())?)
    }
}

// ---------------------------------------------------------------------------
// GLUFFN
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters)]
pub struct GLUFFN {
    #[param]
    pub ln1: nn::Linear,
    #[param]
    pub ln2: nn::Linear,
    pub latent_dim: i32,
}

impl GLUFFN {
    pub fn new(dim: i32, latent_dim: Option<i32>) -> Result<Self> {
        let latent_dim = latent_dim.unwrap_or(dim * 4);
        let ln1 = nn::LinearBuilder::new(dim, latent_dim * 2).build()?;
        let ln2 = nn::LinearBuilder::new(latent_dim, dim).build()?;
        Ok(Self { ln1, ln2, latent_dim })
    }

    pub fn forward(&mut self, x: &Array) -> Result<Array> {
        let h = self.ln1.forward(x)?;
        let x1 = h.index((.., .., ..self.latent_dim));
        let x2 = h.index((.., .., self.latent_dim..));
        let g = nn::gelu(&x1)?;
        let x = &g * &x2;
        let out = self.ln2.forward(&x)?;
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// FFN
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters)]
pub struct FFN {
    #[param]
    pub ln1: nn::Linear,
    #[param]
    pub ln2: nn::Linear,
}

impl FFN {
    pub fn new(dim: i32, latent_dim: Option<i32>) -> Result<Self> {
        let latent_dim = latent_dim.unwrap_or(dim * 4);
        let ln1 = nn::LinearBuilder::new(dim, latent_dim).build()?;
        let ln2 = nn::LinearBuilder::new(latent_dim, dim).build()?;
        Ok(Self { ln1, ln2 })
    }

    pub fn forward(&mut self, x: &Array) -> Result<Array> {
        let h = self.ln1.forward(x)?;
        let h = nn::gelu(&h)?;
        let out = self.ln2.forward(&h)?;
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// CgMLP
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters)]
pub struct CgMLP {
    #[param]
    pub pw1: nn::Conv1d,
    #[param]
    pub norm: RMSNorm,
    #[param]
    pub dw: nn::Conv1d,
    #[param]
    pub pw2: nn::Conv1d,
    pub latent_dim: i32,
    pub use_dw_act: bool,
}

impl CgMLP {
    pub fn new(
        dim: i32,
        kernel_size: i32,
        latent_dim: Option<i32>,
        use_dw_act: bool,
        bias: bool,
    ) -> Result<Self> {
        let latent_dim = latent_dim.unwrap_or(dim);
        let pw1 = nn::Conv1dBuilder::new(dim, latent_dim * 2, 1)
            .bias(bias)
            .build()?;
        let norm = RMSNorm::new(latent_dim, 1.0, 1e-6);
        let padding = (kernel_size - 1) / 2;
        let dw = nn::Conv1dBuilder::new(latent_dim, latent_dim, kernel_size)
            .padding(padding)
            .groups(latent_dim)
            .bias(bias)
            .build()?;
        let pw2 = nn::Conv1dBuilder::new(latent_dim, dim, 1)
            .bias(bias)
            .build()?;
        Ok(Self {
            pw1,
            norm,
            dw,
            pw2,
            latent_dim,
            use_dw_act,
        })
    }

    pub fn forward(&mut self, x: &Array) -> Result<Array> {
        let mut h = self.pw1.forward(x)?;
        h = nn::gelu(&h)?;
        let x1 = h.index((.., .., ..self.latent_dim));
        let x2_raw = h.index((.., .., self.latent_dim..));
        let mut x2 = self.norm.forward(&x2_raw)?;
        x2 = self.dw.forward(&x2)?;
        if self.use_dw_act {
            x2 = nn::gelu(&x2)?;
        }
        let h = &x1 * &x2;
        let out = self.pw2.forward(&h)?;
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// CyclicRegionEmbedding (inference: no random shift)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters)]
pub struct CyclicRegionEmbedding {
    #[param]
    pub embedding: nn::Embedding,
    pub cycle_length: i32,
}

impl CyclicRegionEmbedding {
    pub fn new(embedding_dim: i32, cycle_length: i32) -> Result<Self> {
        let embedding = nn::Embedding::new(cycle_length, embedding_dim)
            .map_err(|e| anyhow::anyhow!("{}", e))?;
        Ok(Self { embedding, cycle_length })
    }

    pub fn forward(&mut self, idx: &Array) -> Result<Array> {
        let modded = mlx_rs::ops::remainder(idx, &Array::from_int(self.cycle_length))?;
        Ok(self.embedding.forward(&modded)?)
    }
}

// ---------------------------------------------------------------------------
// LocalDownsample
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters)]
pub struct LocalDownsample;

impl LocalDownsample {
    pub fn new() -> Self {
        Self
    }

    pub fn forward(&self, x: &Array, regions: &Array, max_n: Option<i32>) -> Result<Array> {
        let n = match max_n {
            Some(n) => n,
            None => regions.max(None)?.item::<i32>(),
        };

        let b_dims: Vec<i32> = std::iter::repeat(1).take(x.ndim() - 2).collect();
        let mut idx_shape = b_dims.clone();
        idx_shape.push(n + 1);
        idx_shape.push(1);
        let idx = Array::arange::<_, i32>(None, n + 1, None)?.reshape(idx_shape.as_slice())?;

        let regions_exp = mlx_rs::ops::expand_dims(regions, -2)?;
        let region_map = idx.eq(&regions_exp)?;
        let region_weight = region_map.as_dtype(Dtype::Float32)?;

        let has_region = region_map.any_axis(-1, Some(true))?;
        let region_size_raw = region_weight.sum_axis(-1, Some(true))?;

        let region_size = mlx_rs::ops::r#where(
            &has_region,
            &region_size_raw,
            &Array::from_f32(1.0),
        )?;
        let weight = &region_weight / &region_size;
        let weight = weight.index((.., 1.., ..));
        let x_down = weight.matmul(x)?;
        Ok(x_down)
    }
}
