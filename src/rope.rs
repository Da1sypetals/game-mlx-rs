use anyhow::Result;
use mlx_rs::Array;
use mlx_rs::Dtype;
use mlx_rs::macros::ModuleParameters;
use mlx_rs::ops::indexing::{Ellipsis, IndexOp, IntoStrideBy};

pub fn compute_inv_freq(dim: i32, theta: f32) -> Result<Array> {
    let half = dim / 2;
    let arange_vec: Vec<f32> = (0..dim)
        .step_by(2)
        .take(half as usize)
        .map(|i| i as f32)
        .collect();
    let arange = Array::from_slice(&arange_vec, &[half]);
    let dim_arr = Array::from_f32(dim as f32);
    let theta_arr = Array::from_f32(theta);
    let exponent = &arange / &dim_arr;
    let base = mlx_rs::ops::power(&theta_arr, &exponent)?;
    Ok(mlx_rs::ops::reciprocal(&base)?)
}

pub fn compute_freqs_cos_sin(seq_len: i32, inv_freq: &Array) -> Result<(Array, Array)> {
    let t_vec: Vec<f32> = (0..seq_len).map(|i| i as f32).collect();
    let t = Array::from_slice(&t_vec, &[seq_len]);
    let freqs = mlx_rs::ops::outer(&t, inv_freq)?;
    let cos_val = mlx_rs::ops::cos(&freqs)?;
    let sin_val = mlx_rs::ops::sin(&freqs)?;
    Ok((cos_val, sin_val))
}

pub fn apply_rotary_emb(x: &Array, freqs_cos: &Array, freqs_sin: &Array) -> Result<Array> {
    let orig_dtype = x.dtype();
    let x_f = x.as_dtype(Dtype::Float32)?;

    let x_r = x_f.index((Ellipsis, (0..).stride_by(2)));
    let x_i = x_f.index((Ellipsis, (1..).stride_by(2)));

    let x_out_r = &(&x_r * freqs_cos) - &(&x_i * freqs_sin);
    let x_out_i = &(&x_r * freqs_sin) + &(&x_i * freqs_cos);

    // Interleave: stack on last dim then reshape
    let stacked = mlx_rs::ops::stack_axis(&[&x_out_r, &x_out_i], -1)?;
    let out = stacked.reshape(x.shape())?;
    Ok(out.as_dtype(orig_dtype)?)
}

pub fn apply_rotary_by_positions(x: &Array, positions: &Array, inv_freq: &Array) -> Result<Array> {
    let pos = mlx_rs::ops::expand_dims(positions, -1)?.as_dtype(Dtype::Float32)?;

    let mut inv_shape = vec![1i32; pos.ndim() - 1];
    inv_shape.push(inv_freq.dim(-1) as i32);
    let inv = inv_freq.reshape(&inv_shape)?;

    let freqs = &pos * &inv;
    let mut freqs_cos = mlx_rs::ops::cos(&freqs)?;
    let mut freqs_sin = mlx_rs::ops::sin(&freqs)?;

    let n_extra = x.ndim() as i32 - positions.ndim() as i32 - 1;
    for _ in 0..n_extra {
        let axis = freqs_cos.ndim() as i32 - 2;
        freqs_cos = mlx_rs::ops::expand_dims(&freqs_cos, axis)?;
        freqs_sin = mlx_rs::ops::expand_dims(&freqs_sin, axis)?;
    }

    apply_rotary_emb(x, &freqs_cos, &freqs_sin)
}

// ---------------------------------------------------------------------------
// SingleRoPosEmb
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters)]
pub struct SingleRoPosEmb {
    pub inv_freq: Array,
    pub dim: i32,
}

impl SingleRoPosEmb {
    pub fn new(dim: i32, theta: f32) -> Result<Self> {
        let inv_freq = compute_inv_freq(dim, theta)?;
        Ok(Self { inv_freq, dim })
    }

    pub fn forward(&self, x: &Array) -> Result<Array> {
        let seq_len = x.dim(-2) as i32;
        let (freqs_cos, freqs_sin) = compute_freqs_cos_sin(seq_len, &self.inv_freq)?;

        let n_lead = x.ndim() - 2;
        let mut shape = vec![1i32; n_lead];
        shape.push(seq_len);
        shape.push(self.dim / 2);
        let freqs_cos = freqs_cos.reshape(&shape)?;
        let freqs_sin = freqs_sin.reshape(&shape)?;

        apply_rotary_emb(x, &freqs_cos, &freqs_sin)
    }
}

// ---------------------------------------------------------------------------
// RegionRoPE
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters)]
pub struct RegionRoPE {
    pub head_dim: i32,
    pub mode: String,
    pub inv_freq: Option<Array>,
    pub inv_freq_global: Option<Array>,
    pub inv_freq_region: Option<Array>,
}

impl RegionRoPE {
    pub fn new(head_dim: i32, mode: &str, theta: f32) -> Result<Self> {
        match mode {
            "local" => {
                let inv_freq = compute_inv_freq(head_dim, theta)?;
                Ok(Self {
                    head_dim,
                    mode: mode.to_string(),
                    inv_freq: Some(inv_freq),
                    inv_freq_global: None,
                    inv_freq_region: None,
                })
            }
            "global" | "mixed" => {
                let half = head_dim / 2;
                let inv_freq_global = compute_inv_freq(half, theta)?;
                let inv_freq_region = compute_inv_freq(half, theta)?;
                Ok(Self {
                    head_dim,
                    mode: mode.to_string(),
                    inv_freq: None,
                    inv_freq_global: Some(inv_freq_global),
                    inv_freq_region: Some(inv_freq_region),
                })
            }
            _ => anyhow::bail!("Unknown RegionRoPE mode: {}", mode),
        }
    }

    pub fn forward(
        &self,
        q: &Array,
        k: &Array,
        q_positions: &Array,
        k_positions: &Array,
        q_region_idx: Option<&Array>,
        k_region_idx: Option<&Array>,
    ) -> Result<(Array, Array)> {
        match self.mode.as_str() {
            "local" => {
                let inv = self.inv_freq.as_ref().unwrap();
                let q_out = apply_rotary_by_positions(q, q_positions, inv)?;
                let k_out = apply_rotary_by_positions(k, k_positions, inv)?;
                Ok((q_out, k_out))
            }
            "global" | "mixed" => {
                let half = self.head_dim / 2;
                let inv_g = self.inv_freq_global.as_ref().unwrap();
                let inv_r = self.inv_freq_region.as_ref().unwrap();

                let q_g = q.index((.., .., .., ..half));
                let q_r = q.index((.., .., .., half..));
                let k_g = k.index((.., .., .., ..half));
                let k_r = k.index((.., .., .., half..));

                let q_g = apply_rotary_by_positions(&q_g, q_positions, inv_g)?;
                let k_g = apply_rotary_by_positions(&k_g, k_positions, inv_g)?;

                let q_ri = q_region_idx.unwrap();
                let k_ri = k_region_idx.unwrap();
                let q_r = apply_rotary_by_positions(&q_r, q_ri, inv_r)?;
                let k_r = apply_rotary_by_positions(&k_r, k_ri, inv_r)?;

                let q_out = mlx_rs::ops::concatenate_axis(&[&q_g, &q_r], -1)?;
                let k_out = mlx_rs::ops::concatenate_axis(&[&k_g, &k_r], -1)?;
                Ok((q_out, k_out))
            }
            _ => unreachable!(),
        }
    }
}
