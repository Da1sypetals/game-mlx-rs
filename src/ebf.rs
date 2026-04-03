use anyhow::Result;
use mlx_rs::Array;
use mlx_rs::module::{Module, Param};
use mlx_rs::nn;
use mlx_rs::builder::Builder;
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::macros::ModuleParameters;

use crate::common_layers::{LayScale, RMSNorm, GLUFFN, FFN, CgMLP};
use crate::eglu::HalfCacheGLUFFN;
use crate::rope::SingleRoPosEmb;

// ---------------------------------------------------------------------------
// AttnWROPEX
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters)]
pub struct AttnWROPEX {
    #[param]
    pub q_linear: nn::Linear,
    #[param]
    pub kv_linear: nn::Linear,
    #[param]
    pub out_linear: nn::Linear,
    pub rope: Option<SingleRoPosEmb>,
    pub num_heads: i32,
    pub head_dim: i32,
    pub scale: f32,
}

impl AttnWROPEX {
    pub fn new(dim: i32, num_heads: i32, head_dim: i32, use_rope: bool) -> Result<Self> {
        let attn_dim = head_dim * num_heads;
        let q_linear = nn::LinearBuilder::new(dim, attn_dim).build()?;
        let kv_linear = nn::LinearBuilder::new(dim, attn_dim * 2).build()?;
        let out_linear = nn::LinearBuilder::new(attn_dim, dim).build()?;
        let rope = if use_rope {
            Some(SingleRoPosEmb::new(head_dim, 10000.0)?)
        } else {
            None
        };
        Ok(Self {
            q_linear, kv_linear, out_linear, rope,
            num_heads, head_dim,
            scale: (head_dim as f32).powf(-0.5),
        })
    }

    pub fn forward(&mut self, x: &Array) -> Result<Array> {
        let shape = x.shape();
        let (b, t) = (shape[0], shape[1]);
        let h = self.num_heads;
        let d = self.head_dim;

        let q = self.q_linear.forward(x)?;
        let kv = self.kv_linear.forward(x)?;
        let k = kv.index((.., .., ..h * d));
        let v = kv.index((.., .., h * d..));

        let q = q.reshape(&[b, t, h, d])?.transpose_axes(&[0, 2, 1, 3][..])?;
        let k = k.reshape(&[b, t, h, d])?.transpose_axes(&[0, 2, 1, 3][..])?;
        let v = v.reshape(&[b, t, h, d])?.transpose_axes(&[0, 2, 1, 3][..])?;

        let (q, k) = if let Some(ref rope) = self.rope {
            (rope.forward(&q)?, rope.forward(&k)?)
        } else {
            (q, k)
        };

        let out = mlx_rs::fast::scaled_dot_product_attention(&q, &k, &v, self.scale, None, None)?;
        let out = out.transpose_axes(&[0, 2, 1, 3][..])?.reshape(&[b, t, h * d])?;
        let out = self.out_linear.forward(&out)?;
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// PAC
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters)]
pub struct PAC {
    #[param]
    pub attn: AttnWROPEX,
    #[param]
    pub c: CgMLP,
    #[param]
    pub a_norm: RMSNorm,
    #[param]
    pub c_norm: RMSNorm,
    #[param]
    pub merge_linear: nn::Linear,
    #[param]
    pub merge_dw_conv: Option<nn::Conv1d>,
}

impl PAC {
    pub fn new(
        dim: i32, num_heads: i32, head_dim: i32,
        c_kernel_size: i32, m_kernel_size: i32, use_rope: bool,
    ) -> Result<Self> {
        let attn = AttnWROPEX::new(dim, num_heads, head_dim, use_rope)?;
        let c = CgMLP::new(dim, c_kernel_size, None, true, true)?;
        let a_norm = RMSNorm::new(dim, 1.0, 1e-6);
        let c_norm = RMSNorm::new(dim, 1.0, 1e-6);
        let merge_linear = nn::LinearBuilder::new(dim * 2, dim).build()?;
        let merge_dw_conv = if m_kernel_size != 0 {
            Some(nn::Conv1dBuilder::new(dim * 2, dim * 2, m_kernel_size)
                .padding(m_kernel_size / 2)
                .groups(dim * 2)
                .build()?)
        } else {
            None
        };
        Ok(Self { attn, c, a_norm, c_norm, merge_linear, merge_dw_conv })
    }

    pub fn forward(&mut self, x: &Array) -> Result<Array> {
        let a_o = self.attn.forward(&self.a_norm.forward(x)?)?;
        let c_o = self.c.forward(&self.c_norm.forward(x)?)?;
        let mut m_o = mlx_rs::ops::concatenate_axis(&[&a_o, &c_o], -1)?;
        if let Some(ref mut conv) = self.merge_dw_conv {
            let residual = m_o.clone();
            m_o = conv.forward(&m_o)?;
            m_o = &m_o + &residual;
        }
        let out = self.merge_linear.forward(&m_o)?;
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// FFN creation helper
// ---------------------------------------------------------------------------

pub enum FfnType {
    Glu,
    Ffn,
    CgMlp,
    Eglu,
}

// ---------------------------------------------------------------------------
// EBF block
// Uses optional fields for ffn1/ffn2 and lay_scale variants
// All ffn fields stored as GLUFFN since config uses "glu" type
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters)]
pub struct EBF {
    #[param]
    pub ffn1: Option<GLUFFN>,
    #[param]
    pub norm1: Option<RMSNorm>,
    #[param]
    pub lay_scale1: Option<LayScale>,
    #[param]
    pub attn: PAC,
    #[param]
    pub lay_scale2: Option<LayScale>,
    #[param]
    pub ffn2: Option<GLUFFN>,
    #[param]
    pub norm2: Option<RMSNorm>,
    #[param]
    pub lay_scale3: Option<LayScale>,
    pub skip_first_ffn: bool,
    pub skip_out_ffn: bool,
    pub use_ls: bool,
}

impl EBF {
    pub fn new(
        dim: i32, num_heads: i32, head_dim: i32,
        c_kernel_size: i32, m_kernel_size: i32,
        use_rope: bool, use_ls: bool,
        skip_first_ffn: bool, skip_out_ffn: bool,
    ) -> Result<Self> {
        let mk_ls = |d| if use_ls { Some(LayScale::new(d, 1e-6)) } else { None };

        let (ffn1, norm1, lay_scale1) = if !skip_first_ffn {
            (
                Some(GLUFFN::new(dim, Some(dim * 4))?),
                Some(RMSNorm::new(dim, 1.0, 1e-6)),
                mk_ls(dim),
            )
        } else {
            (None, None, None)
        };

        let attn = PAC::new(dim, num_heads, head_dim, c_kernel_size, m_kernel_size, use_rope)?;
        let lay_scale2 = mk_ls(dim);

        let (ffn2, norm2, lay_scale3) = if !skip_out_ffn {
            (
                Some(GLUFFN::new(dim, Some(dim * 4))?),
                Some(RMSNorm::new(dim, 1.0, 1e-6)),
                mk_ls(dim),
            )
        } else {
            (None, None, None)
        };

        Ok(Self {
            ffn1, norm1, lay_scale1,
            attn, lay_scale2,
            ffn2, norm2, lay_scale3,
            skip_first_ffn, skip_out_ffn, use_ls,
        })
    }

    fn mask_fill(x: &Array, mask: Option<&Array>) -> Result<Array> {
        match mask {
            None => Ok(x.clone()),
            Some(m) => {
                let m_exp = mlx_rs::ops::expand_dims(m, -1)?;
                let m_not = m_exp.logical_not()?;
                Ok(mlx_rs::ops::r#where(&m_not, &Array::from_f32(0.0), x)?)
            }
        }
    }

    fn apply_ls(ls: &Option<LayScale>, x: &Array) -> Result<Array> {
        match ls {
            Some(ls) => ls.forward(x),
            None => Ok(x.clone()),
        }
    }

    pub fn forward(&mut self, x: &Array, mask: Option<&Array>) -> Result<Array> {
        let mut x = x.clone();

        if !self.skip_first_ffn {
            x = Self::mask_fill(&x, mask)?;
            let normed = self.norm1.as_ref().unwrap().forward(&x)?;
            let ffn_out = self.ffn1.as_mut().unwrap().forward(&normed)?;
            let scaled = Self::apply_ls(&self.lay_scale1, &ffn_out)?;
            x = &scaled * Array::from_f32(0.5) + &x;
        }

        x = Self::mask_fill(&x, mask)?;
        let attn_out = self.attn.forward(&x)?;
        let scaled = Self::apply_ls(&self.lay_scale2, &attn_out)?;
        x = &scaled + &x;

        x = Self::mask_fill(&x, mask)?;
        if !self.skip_out_ffn {
            let normed = self.norm2.as_ref().unwrap().forward(&x)?;
            let ffn_out = self.ffn2.as_mut().unwrap().forward(&normed)?;
            let scaled = Self::apply_ls(&self.lay_scale3, &ffn_out)?;
            x = &scaled * Array::from_f32(0.5) + &x;
            x = Self::mask_fill(&x, mask)?;
        }

        Ok(x)
    }
}

// ---------------------------------------------------------------------------
// EBFBackbone
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters)]
pub struct EBFBackbone {
    #[param]
    pub input_proj: nn::Linear,
    #[param]
    pub layers: Vec<EBF>,
    #[param]
    pub latent_norm: Option<RMSNorm>,
    #[param]
    pub latent_proj: Option<nn::Linear>,
    #[param]
    pub output_norm: Option<RMSNorm>,
    #[param]
    pub output_proj: nn::Linear,
    pub return_latent: bool,
    pub latent_layer_idx: Option<i32>,
}

impl EBFBackbone {
    pub fn new(
        in_dim: i32, out_dim: i32, return_latent: bool,
        dim: i32, num_layers: i32,
        latent_layer_idx: Option<i32>, latent_out_dim: i32,
        num_heads: i32, head_dim: i32,
        c_kernel_size: i32, m_kernel_size: i32,
        use_ls: bool, skip_first_ffn: bool, skip_out_ffn: bool,
    ) -> Result<Self> {
        let input_proj = nn::LinearBuilder::new(in_dim, dim).build()?;

        let mut layers = Vec::with_capacity(num_layers as usize);
        for _ in 0..num_layers {
            layers.push(EBF::new(
                dim, num_heads, head_dim,
                c_kernel_size, m_kernel_size,
                true, use_ls, skip_first_ffn, skip_out_ffn,
            )?);
        }

        let (latent_norm, latent_proj) = if return_latent {
            (
                Some(RMSNorm::new(dim, 1.0, 1e-6)),
                Some(nn::LinearBuilder::new(dim, latent_out_dim).build()?),
            )
        } else {
            (None, None)
        };

        let output_norm = Some(RMSNorm::new(dim, 1.0, 1e-6));
        let output_proj = nn::LinearBuilder::new(dim, out_dim).build()?;

        Ok(Self {
            input_proj, layers, latent_norm, latent_proj,
            output_norm, output_proj,
            return_latent, latent_layer_idx,
        })
    }

    pub fn forward(&mut self, x: &Array, mask: Option<&Array>) -> Result<(Array, Option<Array>)> {
        let mut x = self.input_proj.forward(x)?;
        let mut latent = None;

        for (i, layer) in self.layers.iter_mut().enumerate() {
            x = layer.forward(&x, mask)?;
            if self.return_latent {
                if let Some(idx) = self.latent_layer_idx {
                    if i as i32 == idx - 1 {
                        let n = self.latent_norm.as_ref().unwrap().forward(&x)?;
                        latent = Some(self.latent_proj.as_mut().unwrap().forward(&n)?);
                    }
                }
            }
        }

        if let Some(ref norm) = self.output_norm {
            x = norm.forward(&x)?;
        }
        let out = self.output_proj.forward(&x)?;

        Ok((out, latent))
    }
}
