use anyhow::Result;
use mlx_rs::module::{Module, Param};
use mlx_rs::Array;
use mlx_rs::Dtype;
use mlx_rs::nn;
use mlx_rs::builder::Builder;
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::macros::ModuleParameters;

use crate::common_layers::{LayScale, RMSNorm, GLUFFN, CgMLP};
use crate::rope::RegionRoPE;

// ---------------------------------------------------------------------------
// Region position helpers
// ---------------------------------------------------------------------------

pub fn regions_to_local_positions_v3(regions: &Array) -> Result<Array> {
    let shape = regions.shape();
    let b = shape[0];
    let t = shape[1];

    // shifted = pad(regions[:, :-1], [(0,0), (1,0)])
    let sliced = regions.index((.., ..t - 1));
    let shifted = mlx_rs::ops::pad(&sliced, &[(0, 0), (1, 0)][..], None, None)?;

    let is_start = regions.ne(&shifted)?;

    let ones = mlx_rs::ops::ones::<i32>(&[b, t])?;
    let cumsum = ones.cumsum(Some(-1), None, None)?;

    let zeros_like_cumsum = mlx_rs::ops::zeros::<i32>(&[b, t])?;
    let start_cumsum = mlx_rs::ops::r#where(&is_start, &cumsum, &zeros_like_cumsum)?;
    let segment_id = is_start.as_dtype(Dtype::Int32)?.cumsum(Some(-1), None, None)?;

    let zeros_like_seg = mlx_rs::ops::zeros::<i32>(&[b, t])?;
    let masked_segment_id = mlx_rs::ops::r#where(&is_start, &segment_id, &zeros_like_seg)?;

    let max_seg_arr = segment_id.max(None)?;
    max_seg_arr.eval()?;
    let max_seg = max_seg_arr.item::<i32>() + 1;

    // seg_idx: [1, 1, S]
    let seg_idx_vec: Vec<i32> = (0..max_seg).collect();
    let seg_idx = Array::from_slice(&seg_idx_vec, &[1, 1, max_seg]);

    // masked_seg_exp: [B, T, 1]
    let masked_seg_exp = mlx_rs::ops::expand_dims(&masked_segment_id, -1)?;
    // start_cumsum_exp: [B, T, 1]
    let start_cumsum_exp = mlx_rs::ops::expand_dims(&start_cumsum, -1)?;

    // match_: [B, T, S]
    let match_ = masked_seg_exp.eq(&seg_idx)?;

    // segment_start = (start_cumsum_exp * match_).max(axis=1) -> [B, S]
    let match_f = match_.as_dtype(Dtype::Int32)?;
    let weighted = &start_cumsum_exp * &match_f;
    let segment_start = weighted.max_axis(1, None)?; // [B, S]

    // broadcast_start = segment_start[arange(B)[:, None], segment_id] -> [B, T]
    // Use gather indexing: for each (b_i, t_j), pick segment_start[b_i, segment_id[b_i, t_j]]
    let b_idx_vec: Vec<i32> = (0..b).collect();
    let b_idx = Array::from_slice(&b_idx_vec, &[b, 1]);
    let b_idx_exp = mlx_rs::ops::broadcast_to(&b_idx, &[b, t])?;

    // Flatten for take: segment_start is [B, S], we want segment_start[b, seg_id[b, t]]
    // = flat_segment_start[b * S + seg_id[b, t]]
    let flat_start = segment_start.reshape(&[b * max_seg])?;
    let flat_idx = &(&b_idx_exp * Array::from_int(max_seg)) + &segment_id;
    let broadcast_start = flat_start.index(&flat_idx);

    // local_pos = (cumsum - broadcast_start) * (regions > 0).astype(int32)
    let regions_positive = regions.gt(&Array::from_int(0))?.as_dtype(Dtype::Int32)?;
    let local_pos = &(&cumsum - &broadcast_start) * &regions_positive;
    Ok(local_pos)
}

pub fn compute_positions_local(
    regions: &Array,
    region_token_num: i32,
    n: i32,
    use_pool_offset: bool,
) -> Result<(Array, Array)> {
    let shape = regions.shape();
    let b = shape[0];
    let r = region_token_num;
    let p = n * r;

    let offsets = if use_pool_offset {
        let v: Vec<i32> = (0..r).collect();
        Array::from_slice(&v, &[r])
    } else {
        mlx_rs::ops::zeros::<i32>(&[r])?
    };

    // tile(offsets.reshape(1, R), (n, 1)).reshape(1, P)
    let offsets_row = offsets.reshape(&[1, r])?;
    let tiled = mlx_rs::ops::tile(&offsets_row, &[n, 1])?;
    let tiled_flat = tiled.reshape(&[1, p])?;
    let pool_pos = mlx_rs::ops::broadcast_to(&tiled_flat, &[b, p])?;

    let x_local = regions_to_local_positions_v3(regions)?;
    let regions_positive = regions.gt(&Array::from_int(0))?.as_dtype(Dtype::Int32)?;
    let x_pos = &(&x_local + &Array::from_int(r)) * &regions_positive;

    Ok((pool_pos.as_dtype(Dtype::Float32)?, x_pos.as_dtype(Dtype::Float32)?))
}

// ---------------------------------------------------------------------------
// Attention mask builders
// ---------------------------------------------------------------------------

pub fn build_join_attention_mask(
    regions: &Array,
    region_token_num: i32,
    t_mask: &Array,
    n_mask: &Array,
) -> Result<Array> {
    let shape = regions.shape();
    let b = shape[0];
    let t = shape[1];
    let n = n_mask.shape()[1];
    let r = region_token_num;
    let p = n * r;

    // pool_region: [B, P]
    let arange_n_vec: Vec<i32> = (1..=n).collect();
    let arange_n = Array::from_slice(&arange_n_vec, &[n]);
    let arange_n_col = arange_n.reshape(&[n, 1])?;
    let pool_region_nr = mlx_rs::ops::broadcast_to(&arange_n_col, &[n, r])?;
    let pool_region_row = pool_region_nr.reshape(&[1, p])?;
    let pool_region = mlx_rs::ops::broadcast_to(&pool_region_row, &[b, p])?;

    // full_region: [B, P+T]
    let full_region = mlx_rs::ops::concatenate_axis(&[&pool_region, regions], -1)?;

    // pool_valid: [B, P]
    let n_mask_exp = mlx_rs::ops::expand_dims(n_mask, -1)?;
    let pool_valid_bnr = mlx_rs::ops::broadcast_to(&n_mask_exp, &[b, n, r])?;
    let pool_valid = pool_valid_bnr.reshape(&[b, p])?;

    // full_valid: [B, P+T]
    let full_valid = mlx_rs::ops::concatenate_axis(&[&pool_valid, t_mask], -1)?;

    // is_pool: [B, P+T]
    let ones_bp = mlx_rs::ops::ones::<bool>(&[b, p])?;
    let zeros_bt = mlx_rs::ops::zeros::<bool>(&[b, t])?;
    let is_pool = mlx_rs::ops::concatenate_axis(&[&ones_bp, &zeros_bt], -1)?;

    // same_stream: [B, P+T, P+T]
    let is_pool_q = mlx_rs::ops::expand_dims(&is_pool, -1)?; // [B, P+T, 1]
    let is_pool_k = mlx_rs::ops::expand_dims(&is_pool, -2)?; // [B, 1, P+T]
    let same_stream = is_pool_q.eq(&is_pool_k)?;

    // same_region: [B, P+T, P+T]
    let fr_q = mlx_rs::ops::expand_dims(&full_region, -1)?; // [B, P+T, 1]
    let fr_k = mlx_rs::ops::expand_dims(&full_region, -2)?; // [B, 1, P+T]
    let same_region_raw = fr_q.eq(&fr_k)?;
    let fr_nonzero_q = full_region.ne(&Array::from_int(0))?;
    let fr_nonzero_k = full_region.ne(&Array::from_int(0))?;
    let nz_q = mlx_rs::ops::expand_dims(&fr_nonzero_q, -1)?;
    let nz_k = mlx_rs::ops::expand_dims(&fr_nonzero_k, -2)?;
    let non_pad_region = nz_q.logical_and(&nz_k)?;
    let same_region = same_region_raw.logical_and(&non_pad_region)?;

    // attn_allowed = same_stream | same_region
    let attn_allowed = same_stream.logical_or(&same_region)?;

    // valid_pair: [B, P+T, P+T]
    let fv_q = mlx_rs::ops::expand_dims(&full_valid, -1)?;
    let fv_k = mlx_rs::ops::expand_dims(&full_valid, -2)?;
    let valid_pair = fv_q.logical_and(&fv_k)?;

    // bool mask -> additive mask for sdpa: true -> 0.0, false -> -inf
    let allowed_and_valid = attn_allowed.logical_and(&valid_pair)?;
    let zero = Array::from_f32(0.0);
    let neg_inf = Array::from_f32(-1e9);
    let additive_mask = mlx_rs::ops::r#where(&allowed_and_valid, &zero, &neg_inf)?;

    // [B, 1, P+T, P+T]
    let mask = mlx_rs::ops::expand_dims(&additive_mask, 1)?;
    Ok(mask)
}

// ---------------------------------------------------------------------------
// SplitAttnMasks
// ---------------------------------------------------------------------------

pub struct SplitAttnMasks {
    pub pp: Option<Array>,
    pub xx: Option<Array>,
    pub px: Array,
    pub xp: Array,
}

pub fn build_split_attention_masks(
    regions: &Array,
    region_token_num: i32,
    t_mask: &Array,
    n_mask: &Array,
    region_bias: Option<&RegionBias>,
) -> Result<SplitAttnMasks> {
    let shape = regions.shape();
    let b = shape[0];
    let _t = shape[1];
    let n = n_mask.shape()[1];
    let r = region_token_num;
    let p = n * r;

    // pool_region: [B, P]
    let arange_n_vec: Vec<i32> = (1..=n).collect();
    let arange_n = Array::from_slice(&arange_n_vec, &[n]);
    let arange_n_col = arange_n.reshape(&[n, 1])?;
    let pool_region_nr = mlx_rs::ops::broadcast_to(&arange_n_col, &[n, r])?;
    let pool_region_row = pool_region_nr.reshape(&[1, p])?;
    let pool_region = mlx_rs::ops::broadcast_to(&pool_region_row, &[b, p])?;

    // pool_valid: [B, P]
    let n_mask_exp = mlx_rs::ops::expand_dims(n_mask, -1)?;
    let pool_valid_bnr = mlx_rs::ops::broadcast_to(&n_mask_exp, &[b, n, r])?;
    let pool_valid = pool_valid_bnr.reshape(&[b, p])?;

    let pad_bias = |valid: &Array| -> Result<Array> {
        let v_q = mlx_rs::ops::expand_dims(valid, -1)?; // [B, L, 1]
        let v_k = mlx_rs::ops::expand_dims(valid, -2)?; // [B, 1, L]
        let mask = v_q.logical_and(&v_k)?; // [B, L, L]
        let zero = Array::from_f32(0.0);
        let neg_val = Array::from_f32(-10000.0);
        let bias_2d = mlx_rs::ops::r#where(&mask, &zero, &neg_val)?; // [B, L, L]
        let bias_4d = mlx_rs::ops::expand_dims(&bias_2d, 1)?; // [B, 1, L, L]
        Ok(bias_4d)
    };

    let cross_bias = |q_region: &Array, k_region: &Array, q_valid: &Array, k_valid: &Array| -> Result<Array> {
        let qv_exp = mlx_rs::ops::expand_dims(q_valid, -1)?; // [B, Lq, 1]
        let kv_exp = mlx_rs::ops::expand_dims(k_valid, -2)?; // [B, 1, Lk]
        let pad_mask = qv_exp.logical_and(&kv_exp)?; // [B, Lq, Lk]

        if let Some(rb) = region_bias {
            let pad_mask_4d = mlx_rs::ops::expand_dims(&pad_mask, 1)?; // [B, 1, Lq, Lk]
            let zero = Array::from_f32(0.0);
            let neg_val = Array::from_f32(-10000.0);
            let pb = mlx_rs::ops::r#where(&pad_mask_4d, &zero, &neg_val)?;
            let decay = rb.forward(q_region, k_region)?;
            Ok(&pb + &decay)
        } else {
            let qr_exp = mlx_rs::ops::expand_dims(q_region, -1)?;
            let kr_exp = mlx_rs::ops::expand_dims(k_region, -2)?;
            let same_region = qr_exp.eq(&kr_exp)?;
            let qr_nonzero = q_region.ne(&Array::from_int(0))?;
            let kr_nonzero = k_region.ne(&Array::from_int(0))?;
            let qnz = mlx_rs::ops::expand_dims(&qr_nonzero, -1)?;
            let knz = mlx_rs::ops::expand_dims(&kr_nonzero, -2)?;
            let non_pad = qnz.logical_and(&knz)?;
            let valid_mask = pad_mask.logical_and(&same_region)?.logical_and(&non_pad)?;
            let valid_mask_4d = mlx_rs::ops::expand_dims(&valid_mask, 1)?;
            let zero = Array::from_f32(0.0);
            let neg_val = Array::from_f32(-10000.0);
            Ok(mlx_rs::ops::r#where(&valid_mask_4d, &zero, &neg_val)?)
        }
    };

    // pp_mask: None if all pool_valid are true
    let all_pool = pool_valid.all(None)?;
    all_pool.eval()?;
    let pp_mask = if all_pool.item::<bool>() {
        None
    } else {
        Some(pad_bias(&pool_valid)?)
    };

    let all_t = t_mask.all(None)?;
    all_t.eval()?;
    let xx_mask = if all_t.item::<bool>() {
        None
    } else {
        Some(pad_bias(t_mask)?)
    };

    let px_mask = cross_bias(&pool_region, regions, &pool_valid, t_mask)?;
    let xp_mask = cross_bias(regions, &pool_region, t_mask, &pool_valid)?;

    Ok(SplitAttnMasks {
        pp: pp_mask,
        xx: xx_mask,
        px: px_mask,
        xp: xp_mask,
    })
}

// ---------------------------------------------------------------------------
// RegionBias
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters)]
pub struct RegionBias {
    #[param]
    pub log_alpha: Param<Option<Array>>,
    pub fixed_log_alpha: Option<Array>,
}

impl RegionBias {
    pub fn new(alpha: f32, learnable: bool) -> Self {
        let la = Array::from_f32(alpha.ln());
        if learnable {
            Self {
                log_alpha: Param::new(Some(la)),
                fixed_log_alpha: None,
            }
        } else {
            Self {
                log_alpha: Param::new(None),
                fixed_log_alpha: Some(la),
            }
        }
    }

    fn get_log_alpha(&self) -> &Array {
        if let Some(ref la) = self.log_alpha.value {
            la
        } else {
            self.fixed_log_alpha.as_ref().unwrap()
        }
    }

    pub fn forward(&self, q_region_idx: &Array, k_region_idx: &Array) -> Result<Array> {
        let q_f = mlx_rs::ops::expand_dims(q_region_idx, -1)?.as_dtype(Dtype::Float32)?;
        let k_f = mlx_rs::ops::expand_dims(k_region_idx, -2)?.as_dtype(Dtype::Float32)?;
        let dist = (&q_f - &k_f).abs()?;
        let la = self.get_log_alpha();
        let alpha = mlx_rs::ops::exp(la)?;
        let decay = &(&alpha * &Array::from_f32(-1.0)) * &dist; // [B, Lq, Lk]
        let out = mlx_rs::ops::expand_dims(&decay, 1)?; // [B, 1, Lq, Lk]
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// JointAttention
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters)]
pub struct JointAttention {
    #[param]
    pub pool_qkv: nn::Linear,
    #[param]
    pub x_qkv: nn::Linear,
    #[param]
    pub pool_q_norm: Option<RMSNorm>,
    #[param]
    pub pool_k_norm: Option<RMSNorm>,
    #[param]
    pub x_q_norm: Option<RMSNorm>,
    #[param]
    pub x_k_norm: Option<RMSNorm>,
    #[param]
    pub pool_out: nn::Linear,
    #[param]
    pub x_out: nn::Linear,
    #[param]
    pub pool_norm: RMSNorm,
    #[param]
    pub x_norm: RMSNorm,
    pub rope: Option<RegionRoPE>,
    pub num_heads: i32,
    pub head_dim: i32,
    pub scale: f32,
    pub region_token_num: i32,
    pub rope_mode: String,
    pub use_rope: bool,
    pub use_pool_offset: bool,
    pub do_qk_norm: bool,
}

impl JointAttention {
    pub fn new(
        dim: i32,
        num_heads: i32,
        head_dim: i32,
        region_token_num: i32,
        qk_norm: bool,
        use_rope: bool,
        rope_mode: &str,
        use_pool_offset: bool,
        theta: f32,
    ) -> Result<Self> {
        let attn_dim = num_heads * head_dim;

        let pool_qkv = nn::LinearBuilder::new(dim, attn_dim * 3).build()?;
        let x_qkv = nn::LinearBuilder::new(dim, attn_dim * 3).build()?;

        let (pool_q_norm, pool_k_norm, x_q_norm, x_k_norm) = if qk_norm {
            (
                Some(RMSNorm::new(head_dim, 1.0, 1e-6)),
                Some(RMSNorm::new(head_dim, 1.0, 1e-6)),
                Some(RMSNorm::new(head_dim, 1.0, 1e-6)),
                Some(RMSNorm::new(head_dim, 1.0, 1e-6)),
            )
        } else {
            (None, None, None, None)
        };

        let pool_out = nn::LinearBuilder::new(attn_dim, dim).build()?;
        let x_out = nn::LinearBuilder::new(attn_dim, dim).build()?;
        let pool_norm = RMSNorm::new(dim, 1.0, 1e-6);
        let x_norm = RMSNorm::new(dim, 1.0, 1e-6);

        let rope = if use_rope {
            let rope_m = if rope_mode == "mixed" { "global" } else { "local" };
            Some(RegionRoPE::new(head_dim, rope_m, theta)?)
        } else {
            None
        };

        Ok(Self {
            pool_qkv,
            x_qkv,
            pool_q_norm,
            pool_k_norm,
            x_q_norm,
            x_k_norm,
            pool_out,
            x_out,
            pool_norm,
            x_norm,
            rope,
            num_heads,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
            region_token_num,
            rope_mode: rope_mode.to_string(),
            use_rope,
            use_pool_offset,
            do_qk_norm: qk_norm,
        })
    }

    fn to_heads(&self, x: &Array) -> Result<Array> {
        let s = x.shape();
        let (b, t) = (s[0], s[1]);
        let h = self.num_heads;
        let d = self.head_dim;
        Ok(x.reshape(&[b, t, h, d])?.transpose_axes(&[0, 2, 1, 3][..])?)
    }

    pub fn forward(
        &mut self,
        pool: &Array,
        x: &Array,
        regions: &Array,
        t_mask: &Array,
        n_mask: &Array,
        attn_mask: &Array,
    ) -> Result<(Array, Array)> {
        let n = n_mask.shape()[1];
        let r = self.region_token_num;
        let p = n * r;
        let x_shape = x.shape();
        let b = x_shape[0];
        let t = x_shape[1];
        let h = self.num_heads;
        let d = self.head_dim;

        let pool_normed = self.pool_norm.forward(pool)?;
        let x_normed = self.x_norm.forward(x)?;

        let pqkv = self.pool_qkv.forward(&pool_normed)?;
        let mut pool_q = pqkv.index((.., .., ..h * d));
        let mut pool_k = pqkv.index((.., .., h * d..2 * h * d));
        let pool_v_flat = pqkv.index((.., .., 2 * h * d..));

        let xqkv = self.x_qkv.forward(&x_normed)?;
        let mut x_q = xqkv.index((.., .., ..h * d));
        let mut x_k = xqkv.index((.., .., h * d..2 * h * d));
        let x_v_flat = xqkv.index((.., .., 2 * h * d..));

        pool_q = self.to_heads(&pool_q)?;
        pool_k = self.to_heads(&pool_k)?;
        let pool_v = self.to_heads(&pool_v_flat)?;
        x_q = self.to_heads(&x_q)?;
        x_k = self.to_heads(&x_k)?;
        let x_v = self.to_heads(&x_v_flat)?;

        if self.do_qk_norm {
            pool_q = self.pool_q_norm.as_ref().unwrap().forward(&pool_q)?;
            pool_k = self.pool_k_norm.as_ref().unwrap().forward(&pool_k)?;
            x_q = self.x_q_norm.as_ref().unwrap().forward(&x_q)?;
            x_k = self.x_k_norm.as_ref().unwrap().forward(&x_k)?;
        }

        // Concatenate pool and x along sequence dimension (axis=2)
        let mut q = mlx_rs::ops::concatenate_axis(&[&pool_q, &x_q], 2)?;
        let mut k = mlx_rs::ops::concatenate_axis(&[&pool_k, &x_k], 2)?;
        let v = mlx_rs::ops::concatenate_axis(&[&pool_v, &x_v], 2)?;

        if self.use_rope {
            if let Some(ref rope) = self.rope {
                if self.rope_mode == "local" {
                    let (pool_pos, x_pos) =
                        compute_positions_local(regions, r, n as i32, self.use_pool_offset)?;
                    let full_pos = mlx_rs::ops::concatenate_axis(&[&pool_pos, &x_pos], -1)?;
                    let (q_r, k_r) = rope.forward(&q, &k, &full_pos, &full_pos, None, None)?;
                    q = q_r;
                    k = k_r;
                } else if self.rope_mode == "global" {
                    let pool_pos_vec: Vec<f32> = (0..p).map(|i| i as f32).collect();
                    let pool_pos_1d = Array::from_slice(&pool_pos_vec, &[1, p]);
                    let pool_pos = mlx_rs::ops::broadcast_to(&pool_pos_1d, &[b, p])?;
                    let x_pos_vec: Vec<f32> = (0..t as i32).map(|i| i as f32).collect();
                    let x_pos_1d = Array::from_slice(&x_pos_vec, &[1, t as i32]);
                    let x_pos = mlx_rs::ops::broadcast_to(&x_pos_1d, &[b, t as i32])?;
                    let full_pos = mlx_rs::ops::concatenate_axis(&[&pool_pos, &x_pos], -1)?;
                    let (q_r, k_r) = rope.forward(&q, &k, &full_pos, &full_pos, None, None)?;
                    q = q_r;
                    k = k_r;
                } else if self.rope_mode == "mixed" {
                    let pool_seq_vec: Vec<f32> = (0..p).map(|i| i as f32).collect();
                    let pool_seq_1d = Array::from_slice(&pool_seq_vec, &[1, p]);
                    let pool_seq_pos = mlx_rs::ops::broadcast_to(&pool_seq_1d, &[b, p])?;
                    let x_seq_vec: Vec<f32> = (0..t as i32).map(|i| i as f32).collect();
                    let x_seq_1d = Array::from_slice(&x_seq_vec, &[1, t as i32]);
                    let x_seq_pos = mlx_rs::ops::broadcast_to(&x_seq_1d, &[b, t as i32])?;
                    let q_gpos = mlx_rs::ops::concatenate_axis(&[&pool_seq_pos, &x_seq_pos], -1)?;

                    let (pool_lpos, x_lpos) =
                        compute_positions_local(regions, r, n as i32, self.use_pool_offset)?;
                    let q_ridx = mlx_rs::ops::concatenate_axis(&[&pool_lpos, &x_lpos], -1)?;

                    let (q_r, k_r) = rope.forward(
                        &q, &k, &q_gpos, &q_gpos, Some(&q_ridx), Some(&q_ridx),
                    )?;
                    q = q_r;
                    k = k_r;
                }
            }
        }

        let out = mlx_rs::fast::scaled_dot_product_attention(
            &q, &k, &v, self.scale, attn_mask, None,
        )?;

        // Split pool and x attention outputs
        let pool_attn_raw = out.index((.., .., ..p, ..));
        let x_attn_raw = out.index((.., .., p.., ..));

        let pool_attn_t = pool_attn_raw.transpose_axes(&[0, 2, 1, 3][..])?;
        let pool_attn_flat = pool_attn_t.reshape(&[b, p, h * d])?;
        let x_attn_t = x_attn_raw.transpose_axes(&[0, 2, 1, 3][..])?;
        let x_attn_flat = x_attn_t.reshape(&[b, t as i32, h * d])?;

        let pool_attn = self.pool_out.forward(&pool_attn_flat)?;
        let x_attn = self.x_out.forward(&x_attn_flat)?;

        // Mask outputs
        // pool_mask_exp: [B, P, 1]
        let n_mask_exp = mlx_rs::ops::expand_dims(n_mask, -1)?;
        let pool_mask_bnr = mlx_rs::ops::broadcast_to(&n_mask_exp, &[b, n as i32, r])?;
        let pool_mask_bp = pool_mask_bnr.reshape(&[b, p])?;
        let pool_mask_exp = mlx_rs::ops::expand_dims(&pool_mask_bp, -1)?.as_dtype(Dtype::Float32)?;
        let pool_o = &pool_attn * &pool_mask_exp;

        let t_mask_exp = mlx_rs::ops::expand_dims(t_mask, -1)?.as_dtype(Dtype::Float32)?;
        let x_o = &x_attn * &t_mask_exp;

        Ok((pool_o, x_o))
    }
}

// ---------------------------------------------------------------------------
// SplitJointAttention
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters)]
pub struct SplitJointAttention {
    #[param]
    pub pool_qkv: nn::Linear,
    #[param]
    pub x_qkv: nn::Linear,
    #[param]
    pub pool_q_norm: Option<RMSNorm>,
    #[param]
    pub pool_k_norm: Option<RMSNorm>,
    #[param]
    pub x_q_norm: Option<RMSNorm>,
    #[param]
    pub x_k_norm: Option<RMSNorm>,
    #[param]
    pub pool_norm: RMSNorm,
    #[param]
    pub x_norm: RMSNorm,
    #[param]
    pub pool_merge: nn::Linear,
    #[param]
    pub x_merge: nn::Linear,
    pub rope: Option<RegionRoPE>,
    pub num_heads: i32,
    pub head_dim: i32,
    pub scale: f32,
    pub region_token_num: i32,
    pub rope_mode: String,
    pub use_rope: bool,
    pub use_pool_offset: bool,
    pub do_qk_norm: bool,
}

impl SplitJointAttention {
    pub fn new(
        dim: i32,
        num_heads: i32,
        head_dim: i32,
        region_token_num: i32,
        qk_norm: bool,
        use_rope: bool,
        rope_mode: &str,
        use_pool_offset: bool,
        theta: f32,
    ) -> Result<Self> {
        let attn_dim = num_heads * head_dim;

        let pool_qkv = nn::LinearBuilder::new(dim, attn_dim * 3).build()?;
        let x_qkv = nn::LinearBuilder::new(dim, attn_dim * 3).build()?;

        let (pool_q_norm, pool_k_norm, x_q_norm, x_k_norm) = if qk_norm {
            (
                Some(RMSNorm::new(head_dim, 1.0, 1e-6)),
                Some(RMSNorm::new(head_dim, 1.0, 1e-6)),
                Some(RMSNorm::new(head_dim, 1.0, 1e-6)),
                Some(RMSNorm::new(head_dim, 1.0, 1e-6)),
            )
        } else {
            (None, None, None, None)
        };

        let pool_norm = RMSNorm::new(dim, 1.0, 1e-6);
        let x_norm = RMSNorm::new(dim, 1.0, 1e-6);
        let pool_merge = nn::LinearBuilder::new(attn_dim * 2, dim).build()?;
        let x_merge = nn::LinearBuilder::new(attn_dim * 2, dim).build()?;

        let rope = if use_rope {
            let rope_m = if rope_mode == "mixed" { "global" } else { "local" };
            Some(RegionRoPE::new(head_dim, rope_m, theta)?)
        } else {
            None
        };

        Ok(Self {
            pool_qkv,
            x_qkv,
            pool_q_norm,
            pool_k_norm,
            x_q_norm,
            x_k_norm,
            pool_norm,
            x_norm,
            pool_merge,
            x_merge,
            rope,
            num_heads,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
            region_token_num,
            rope_mode: rope_mode.to_string(),
            use_rope,
            use_pool_offset,
            do_qk_norm: qk_norm,
        })
    }

    fn to_heads(&self, x: &Array) -> Result<Array> {
        let s = x.shape();
        let (b, t) = (s[0], s[1]);
        let h = self.num_heads;
        let d = self.head_dim;
        Ok(x.reshape(&[b, t, h, d])?.transpose_axes(&[0, 2, 1, 3][..])?)
    }

    pub fn forward(
        &mut self,
        pool: &Array,
        x: &Array,
        regions: &Array,
        t_mask: &Array,
        n_mask: &Array,
        attn_masks: &SplitAttnMasks,
    ) -> Result<(Array, Array)> {
        let n = n_mask.shape()[1];
        let r = self.region_token_num;
        let p = n * r;
        let x_shape = x.shape();
        let b = x_shape[0];
        let t = x_shape[1];
        let h = self.num_heads;
        let d = self.head_dim;

        let pool_normed = self.pool_norm.forward(pool)?;
        let x_normed = self.x_norm.forward(x)?;

        let pqkv = self.pool_qkv.forward(&pool_normed)?;
        let mut pool_q = pqkv.index((.., .., ..h * d));
        let mut pool_k = pqkv.index((.., .., h * d..2 * h * d));
        let pool_v_flat = pqkv.index((.., .., 2 * h * d..));

        let xqkv = self.x_qkv.forward(&x_normed)?;
        let mut x_q = xqkv.index((.., .., ..h * d));
        let mut x_k = xqkv.index((.., .., h * d..2 * h * d));
        let x_v_flat = xqkv.index((.., .., 2 * h * d..));

        pool_q = self.to_heads(&pool_q)?;
        pool_k = self.to_heads(&pool_k)?;
        let pool_v = self.to_heads(&pool_v_flat)?;
        x_q = self.to_heads(&x_q)?;
        x_k = self.to_heads(&x_k)?;
        let x_v = self.to_heads(&x_v_flat)?;

        if self.do_qk_norm {
            pool_q = self.pool_q_norm.as_ref().unwrap().forward(&pool_q)?;
            pool_k = self.pool_k_norm.as_ref().unwrap().forward(&pool_k)?;
            x_q = self.x_q_norm.as_ref().unwrap().forward(&x_q)?;
            x_k = self.x_k_norm.as_ref().unwrap().forward(&x_k)?;
        }

        let (pool_q_r, pool_k_r, x_q_r, x_k_r);
        if self.use_rope {
            let rope = self.rope.as_ref().unwrap();
            if self.rope_mode == "local" {
                let (pool_lpos, x_lpos) =
                    compute_positions_local(regions, r, n as i32, self.use_pool_offset)?;
                let (pqr, pkr) = rope.forward(&pool_q, &pool_k, &pool_lpos, &pool_lpos, None, None)?;
                let (xqr, xkr) = rope.forward(&x_q, &x_k, &x_lpos, &x_lpos, None, None)?;
                pool_q_r = pqr;
                pool_k_r = pkr;
                x_q_r = xqr;
                x_k_r = xkr;
            } else if self.rope_mode == "global" {
                let pool_pos_vec: Vec<f32> = (0..p).map(|i| i as f32).collect();
                let pool_pos_1d = Array::from_slice(&pool_pos_vec, &[1, p]);
                let pool_pos = mlx_rs::ops::broadcast_to(&pool_pos_1d, &[b, p])?;
                let x_pos_vec: Vec<f32> = (0..t as i32).map(|i| i as f32).collect();
                let x_pos_1d = Array::from_slice(&x_pos_vec, &[1, t as i32]);
                let x_pos = mlx_rs::ops::broadcast_to(&x_pos_1d, &[b, t as i32])?;
                let (pqr, pkr) = rope.forward(&pool_q, &pool_k, &pool_pos, &pool_pos, None, None)?;
                let (xqr, xkr) = rope.forward(&x_q, &x_k, &x_pos, &x_pos, None, None)?;
                pool_q_r = pqr;
                pool_k_r = pkr;
                x_q_r = xqr;
                x_k_r = xkr;
            } else {
                // mixed
                let pool_gpos_vec: Vec<f32> = (0..p).map(|i| i as f32).collect();
                let pool_gpos_1d = Array::from_slice(&pool_gpos_vec, &[1, p]);
                let pool_gpos = mlx_rs::ops::broadcast_to(&pool_gpos_1d, &[b, p])?;
                let x_gpos_vec: Vec<f32> = (0..t as i32).map(|i| i as f32).collect();
                let x_gpos_1d = Array::from_slice(&x_gpos_vec, &[1, t as i32]);
                let x_gpos = mlx_rs::ops::broadcast_to(&x_gpos_1d, &[b, t as i32])?;
                let (pool_lpos, x_lpos) =
                    compute_positions_local(regions, r, n as i32, self.use_pool_offset)?;
                let (pqr, pkr) = rope.forward(
                    &pool_q, &pool_k, &pool_gpos, &pool_gpos,
                    Some(&pool_lpos), Some(&pool_lpos),
                )?;
                let (xqr, xkr) = rope.forward(
                    &x_q, &x_k, &x_gpos, &x_gpos,
                    Some(&x_lpos), Some(&x_lpos),
                )?;
                pool_q_r = pqr;
                pool_k_r = pkr;
                x_q_r = xqr;
                x_k_r = xkr;
            }
        } else {
            pool_q_r = pool_q;
            pool_k_r = pool_k;
            x_q_r = x_q;
            x_k_r = x_k;
        }

        // Four attention operations
        let pp_mask_sdpa = attn_masks.pp.as_ref().map(mlx_rs::fast::ScaledDotProductAttentionMask::from);
        let pp_out = mlx_rs::fast::scaled_dot_product_attention(
            &pool_q_r, &pool_k_r, &pool_v, self.scale,
            pp_mask_sdpa, None,
        )?;
        let xx_mask_sdpa = attn_masks.xx.as_ref().map(mlx_rs::fast::ScaledDotProductAttentionMask::from);
        let xx_out = mlx_rs::fast::scaled_dot_product_attention(
            &x_q_r, &x_k_r, &x_v, self.scale,
            xx_mask_sdpa, None,
        )?;
        let px_out = mlx_rs::fast::scaled_dot_product_attention(
            &pool_q_r, &x_k_r, &x_v, self.scale,
            &attn_masks.px, None,
        )?;
        let xp_out = mlx_rs::fast::scaled_dot_product_attention(
            &x_q_r, &pool_k_r, &pool_v, self.scale,
            &attn_masks.xp, None,
        )?;

        let pp_flat = pp_out.transpose_axes(&[0, 2, 1, 3][..])?.reshape(&[b, p, h * d])?;
        let px_flat = px_out.transpose_axes(&[0, 2, 1, 3][..])?.reshape(&[b, p, h * d])?;
        let xx_flat = xx_out.transpose_axes(&[0, 2, 1, 3][..])?.reshape(&[b, t as i32, h * d])?;
        let xp_flat = xp_out.transpose_axes(&[0, 2, 1, 3][..])?.reshape(&[b, t as i32, h * d])?;

        let pool_cat = mlx_rs::ops::concatenate_axis(&[&pp_flat, &px_flat], -1)?;
        let x_cat = mlx_rs::ops::concatenate_axis(&[&xx_flat, &xp_flat], -1)?;

        let pool_attn = self.pool_merge.forward(&pool_cat)?;
        let x_attn = self.x_merge.forward(&x_cat)?;

        // Mask outputs
        let n_mask_exp = mlx_rs::ops::expand_dims(n_mask, -1)?;
        let pool_mask_bnr = mlx_rs::ops::broadcast_to(&n_mask_exp, &[b, n as i32, r])?;
        let pool_mask_bp = pool_mask_bnr.reshape(&[b, p])?;
        let pool_mask_exp = mlx_rs::ops::expand_dims(&pool_mask_bp, -1)?.as_dtype(Dtype::Float32)?;
        let pool_o = &pool_attn * &pool_mask_exp;

        let t_mask_exp = mlx_rs::ops::expand_dims(t_mask, -1)?.as_dtype(Dtype::Float32)?;
        let x_o = &x_attn * &t_mask_exp;

        Ok((pool_o, x_o))
    }
}

// ---------------------------------------------------------------------------
// PJAC
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters)]
pub struct PJAC {
    #[param]
    pub jattn: Option<JointAttention>,
    #[param]
    pub jattn_split: Option<SplitJointAttention>,
    #[param]
    pub c_x: CgMLP,
    #[param]
    pub c_pool: CgMLP,
    #[param]
    pub c_norm_x: RMSNorm,
    #[param]
    pub c_norm_pool: RMSNorm,
    #[param]
    pub merge_linear_x: nn::Linear,
    #[param]
    pub merge_linear_pool: nn::Linear,
    #[param]
    pub merge_dw_conv_x: Option<nn::Conv1d>,
    #[param]
    pub merge_dw_conv_pool: Option<nn::Conv1d>,
    pub attn_type: String,
}

impl PJAC {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        dim: i32,
        num_heads: i32,
        head_dim: i32,
        c_kernel_size_pool: i32,
        m_kernel_size_pool: i32,
        c_kernel_size_x: i32,
        m_kernel_size_x: i32,
        region_token_num: i32,
        qk_norm: bool,
        use_rope: bool,
        rope_mode: &str,
        use_pool_offset: bool,
        theta: f32,
        attn_type: &str,
    ) -> Result<Self> {
        let (jattn, jattn_split) = if attn_type == "joint" {
            let ja = JointAttention::new(
                dim, num_heads, head_dim, region_token_num, qk_norm,
                use_rope, rope_mode, use_pool_offset, theta,
            )?;
            (Some(ja), None)
        } else {
            let sa = SplitJointAttention::new(
                dim, num_heads, head_dim, region_token_num, qk_norm,
                use_rope, rope_mode, use_pool_offset, theta,
            )?;
            (None, Some(sa))
        };

        let c_x = CgMLP::new(dim, c_kernel_size_x, None, true, true)?;
        let c_pool = CgMLP::new(dim, c_kernel_size_pool, None, true, true)?;
        let c_norm_x = RMSNorm::new(dim, 1.0, 1e-6);
        let c_norm_pool = RMSNorm::new(dim, 1.0, 1e-6);

        let merge_linear_x = nn::LinearBuilder::new(dim * 2, dim).build()?;
        let merge_linear_pool = nn::LinearBuilder::new(dim * 2, dim).build()?;

        let merge_dw_conv_x = if m_kernel_size_x != 0 {
            Some(
                nn::Conv1dBuilder::new(dim * 2, dim * 2, m_kernel_size_x)
                    .padding(m_kernel_size_x / 2)
                    .groups(dim * 2)
                    .build()?,
            )
        } else {
            None
        };

        let merge_dw_conv_pool = if m_kernel_size_pool != 0 {
            Some(
                nn::Conv1dBuilder::new(dim * 2, dim * 2, m_kernel_size_pool)
                    .padding(m_kernel_size_pool / 2)
                    .groups(dim * 2)
                    .build()?,
            )
        } else {
            None
        };

        Ok(Self {
            jattn,
            jattn_split,
            c_x,
            c_pool,
            c_norm_x,
            c_norm_pool,
            merge_linear_x,
            merge_linear_pool,
            merge_dw_conv_x,
            merge_dw_conv_pool,
            attn_type: attn_type.to_string(),
        })
    }

    pub fn forward_joint(
        &mut self,
        pool: &Array,
        x: &Array,
        regions: &Array,
        t_mask: &Array,
        n_mask: &Array,
        attn_mask: &Array,
    ) -> Result<(Array, Array)> {
        let (a_pool, a_x) = self
            .jattn
            .as_mut()
            .unwrap()
            .forward(pool, x, regions, t_mask, n_mask, attn_mask)?;

        self.merge_outputs(pool, x, &a_pool, &a_x)
    }

    pub fn forward_split(
        &mut self,
        pool: &Array,
        x: &Array,
        regions: &Array,
        t_mask: &Array,
        n_mask: &Array,
        attn_masks: &SplitAttnMasks,
    ) -> Result<(Array, Array)> {
        let (a_pool, a_x) = self
            .jattn_split
            .as_mut()
            .unwrap()
            .forward(pool, x, regions, t_mask, n_mask, attn_masks)?;

        self.merge_outputs(pool, x, &a_pool, &a_x)
    }

    fn merge_outputs(
        &mut self,
        pool: &Array,
        x: &Array,
        a_pool: &Array,
        a_x: &Array,
    ) -> Result<(Array, Array)> {
        let c_pool = self.c_pool.forward(&self.c_norm_pool.forward(pool)?)?;
        let c_x = self.c_x.forward(&self.c_norm_x.forward(x)?)?;

        let mut m_pool = mlx_rs::ops::concatenate_axis(&[a_pool, &c_pool], -1)?;
        let mut m_x = mlx_rs::ops::concatenate_axis(&[a_x, &c_x], -1)?;

        if let Some(ref mut conv) = self.merge_dw_conv_pool {
            let residual = m_pool.clone();
            m_pool = &conv.forward(&m_pool)? + &residual;
        }
        m_pool = self.merge_linear_pool.forward(&m_pool)?;

        if let Some(ref mut conv) = self.merge_dw_conv_x {
            let residual = m_x.clone();
            m_x = &conv.forward(&m_x)? + &residual;
        }
        m_x = self.merge_linear_x.forward(&m_x)?;

        Ok((m_pool, m_x))
    }
}

// ---------------------------------------------------------------------------
// JEBF layer
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters)]
pub struct JEBF {
    #[param]
    pub ffn1_x: Option<GLUFFN>,
    #[param]
    pub ffn1_pool: Option<GLUFFN>,
    #[param]
    pub norm_ffn1_x: Option<RMSNorm>,
    #[param]
    pub norm_ffn1_pool: Option<RMSNorm>,
    #[param]
    pub lay_scale_ffn1_x: Option<LayScale>,
    #[param]
    pub lay_scale_ffn1_pool: Option<LayScale>,
    #[param]
    pub attn: PJAC,
    #[param]
    pub lay_scale_jpac_x: Option<LayScale>,
    #[param]
    pub lay_scale_jpac_pool: Option<LayScale>,
    #[param]
    pub ffn2_x: Option<GLUFFN>,
    #[param]
    pub ffn2_pool: Option<GLUFFN>,
    #[param]
    pub norm_ffn2_x: Option<RMSNorm>,
    #[param]
    pub norm_ffn2_pool: Option<RMSNorm>,
    #[param]
    pub lay_scale_ffn2_x: Option<LayScale>,
    #[param]
    pub lay_scale_ffn2_pool: Option<LayScale>,
    pub skip_first_ffn: bool,
    pub skip_out_ffn: bool,
    pub attn_type: String,
}

impl JEBF {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        dim: i32,
        num_heads: i32,
        head_dim: i32,
        c_kernel_size_pool: i32,
        m_kernel_size_pool: i32,
        c_kernel_size_x: i32,
        m_kernel_size_x: i32,
        region_token_num: i32,
        qk_norm: bool,
        use_rope: bool,
        rope_mode: &str,
        use_pool_offset: bool,
        theta: f32,
        skip_first_ffn: bool,
        skip_out_ffn: bool,
        use_ls: bool,
        attn_type: &str,
    ) -> Result<Self> {
        let mk_ls = |d: i32| -> Option<LayScale> {
            if use_ls {
                Some(LayScale::new(d, 1e-6))
            } else {
                None
            }
        };

        let (ffn1_x, ffn1_pool, norm_ffn1_x, norm_ffn1_pool, lay_scale_ffn1_x, lay_scale_ffn1_pool) =
            if !skip_first_ffn {
                (
                    Some(GLUFFN::new(dim, Some(dim * 4))?),
                    Some(GLUFFN::new(dim, Some(dim * 4))?),
                    Some(RMSNorm::new(dim, 1.0, 1e-6)),
                    Some(RMSNorm::new(dim, 1.0, 1e-6)),
                    mk_ls(dim),
                    mk_ls(dim),
                )
            } else {
                (None, None, None, None, None, None)
            };

        let attn = PJAC::new(
            dim, num_heads, head_dim,
            c_kernel_size_pool, m_kernel_size_pool,
            c_kernel_size_x, m_kernel_size_x,
            region_token_num, qk_norm,
            use_rope, rope_mode, use_pool_offset, theta,
            attn_type,
        )?;

        let lay_scale_jpac_x = mk_ls(dim);
        let lay_scale_jpac_pool = mk_ls(dim);

        let (ffn2_x, ffn2_pool, norm_ffn2_x, norm_ffn2_pool, lay_scale_ffn2_x, lay_scale_ffn2_pool) =
            if !skip_out_ffn {
                (
                    Some(GLUFFN::new(dim, Some(dim * 4))?),
                    Some(GLUFFN::new(dim, Some(dim * 4))?),
                    Some(RMSNorm::new(dim, 1.0, 1e-6)),
                    Some(RMSNorm::new(dim, 1.0, 1e-6)),
                    mk_ls(dim),
                    mk_ls(dim),
                )
            } else {
                (None, None, None, None, None, None)
            };

        Ok(Self {
            ffn1_x,
            ffn1_pool,
            norm_ffn1_x,
            norm_ffn1_pool,
            lay_scale_ffn1_x,
            lay_scale_ffn1_pool,
            attn,
            lay_scale_jpac_x,
            lay_scale_jpac_pool,
            ffn2_x,
            ffn2_pool,
            norm_ffn2_x,
            norm_ffn2_pool,
            lay_scale_ffn2_x,
            lay_scale_ffn2_pool,
            skip_first_ffn,
            skip_out_ffn,
            attn_type: attn_type.to_string(),
        })
    }

    fn apply_ls(ls: &Option<LayScale>, x: &Array) -> Result<Array> {
        match ls {
            Some(ls) => ls.forward(x),
            None => Ok(x.clone()),
        }
    }

    fn mfill_x(x: &Array, t_mask: &Array) -> Result<Array> {
        let m_not = mlx_rs::ops::expand_dims(t_mask, -1)?.logical_not()?;
        let zeros = mlx_rs::ops::zeros::<f32>(x.shape())?;
        Ok(mlx_rs::ops::r#where(&m_not, &zeros, x)?)
    }

    fn mfill_pool(x: &Array, pool_mask: &Array) -> Result<Array> {
        let m_not = mlx_rs::ops::expand_dims(pool_mask, -1)?.logical_not()?;
        let zeros = mlx_rs::ops::zeros::<f32>(x.shape())?;
        Ok(mlx_rs::ops::r#where(&m_not, &zeros, x)?)
    }

    pub fn forward_joint(
        &mut self,
        pool: &Array,
        x: &Array,
        regions: &Array,
        t_mask: &Array,
        n_mask: &Array,
        attn_mask: &Array,
    ) -> Result<(Array, Array)> {
        let n_shape = n_mask.shape();
        let b = n_shape[0];
        let n = n_shape[1];
        let r = pool.shape()[1] as i32 / n as i32;

        let n_mask_exp = mlx_rs::ops::expand_dims(n_mask, -1)?;
        let pool_mask_bnr = mlx_rs::ops::broadcast_to(&n_mask_exp, &[b, n as i32, r])?;
        let pool_mask = pool_mask_bnr.reshape(&[b, n as i32 * r])?;

        let mut x = Self::mfill_x(x, t_mask)?;
        let mut pool = Self::mfill_pool(pool, &pool_mask)?;

        if !self.skip_first_ffn {
            let normed_x = self.norm_ffn1_x.as_ref().unwrap().forward(&x)?;
            let ffn_x = self.ffn1_x.as_mut().unwrap().forward(&normed_x)?;
            let scaled_x = Self::apply_ls(&self.lay_scale_ffn1_x, &ffn_x)?;
            x = &scaled_x + &x;

            let normed_pool = self.norm_ffn1_pool.as_ref().unwrap().forward(&pool)?;
            let ffn_pool = self.ffn1_pool.as_mut().unwrap().forward(&normed_pool)?;
            let scaled_pool = Self::apply_ls(&self.lay_scale_ffn1_pool, &ffn_pool)?;
            pool = &scaled_pool + &pool;
        }

        x = Self::mfill_x(&x, t_mask)?;
        pool = Self::mfill_pool(&pool, &pool_mask)?;

        let (p_o, x_o) = self.attn.forward_joint(&pool, &x, regions, t_mask, n_mask, attn_mask)?;
        x = &Self::apply_ls(&self.lay_scale_jpac_x, &x_o)? + &x;
        pool = &Self::apply_ls(&self.lay_scale_jpac_pool, &p_o)? + &pool;

        x = Self::mfill_x(&x, t_mask)?;
        pool = Self::mfill_pool(&pool, &pool_mask)?;

        if !self.skip_out_ffn {
            let normed_x = self.norm_ffn2_x.as_ref().unwrap().forward(&x)?;
            let ffn_x = self.ffn2_x.as_mut().unwrap().forward(&normed_x)?;
            let scaled_x = Self::apply_ls(&self.lay_scale_ffn2_x, &ffn_x)?;
            x = &scaled_x + &x;

            let normed_pool = self.norm_ffn2_pool.as_ref().unwrap().forward(&pool)?;
            let ffn_pool = self.ffn2_pool.as_mut().unwrap().forward(&normed_pool)?;
            let scaled_pool = Self::apply_ls(&self.lay_scale_ffn2_pool, &ffn_pool)?;
            pool = &scaled_pool + &pool;
        }

        Ok((x, pool))
    }

    pub fn forward_split(
        &mut self,
        pool: &Array,
        x: &Array,
        regions: &Array,
        t_mask: &Array,
        n_mask: &Array,
        attn_masks: &SplitAttnMasks,
    ) -> Result<(Array, Array)> {
        let n_shape = n_mask.shape();
        let b = n_shape[0];
        let n = n_shape[1];
        let r = pool.shape()[1] as i32 / n as i32;

        let n_mask_exp = mlx_rs::ops::expand_dims(n_mask, -1)?;
        let pool_mask_bnr = mlx_rs::ops::broadcast_to(&n_mask_exp, &[b, n as i32, r])?;
        let pool_mask = pool_mask_bnr.reshape(&[b, n as i32 * r])?;

        let mut x = Self::mfill_x(x, t_mask)?;
        let mut pool = Self::mfill_pool(pool, &pool_mask)?;

        if !self.skip_first_ffn {
            let normed_x = self.norm_ffn1_x.as_ref().unwrap().forward(&x)?;
            let ffn_x = self.ffn1_x.as_mut().unwrap().forward(&normed_x)?;
            let scaled_x = Self::apply_ls(&self.lay_scale_ffn1_x, &ffn_x)?;
            x = &scaled_x + &x;

            let normed_pool = self.norm_ffn1_pool.as_ref().unwrap().forward(&pool)?;
            let ffn_pool = self.ffn1_pool.as_mut().unwrap().forward(&normed_pool)?;
            let scaled_pool = Self::apply_ls(&self.lay_scale_ffn1_pool, &ffn_pool)?;
            pool = &scaled_pool + &pool;
        }

        x = Self::mfill_x(&x, t_mask)?;
        pool = Self::mfill_pool(&pool, &pool_mask)?;

        let (p_o, x_o) = self.attn.forward_split(&pool, &x, regions, t_mask, n_mask, attn_masks)?;
        x = &Self::apply_ls(&self.lay_scale_jpac_x, &x_o)? + &x;
        pool = &Self::apply_ls(&self.lay_scale_jpac_pool, &p_o)? + &pool;

        x = Self::mfill_x(&x, t_mask)?;
        pool = Self::mfill_pool(&pool, &pool_mask)?;

        if !self.skip_out_ffn {
            let normed_x = self.norm_ffn2_x.as_ref().unwrap().forward(&x)?;
            let ffn_x = self.ffn2_x.as_mut().unwrap().forward(&normed_x)?;
            let scaled_x = Self::apply_ls(&self.lay_scale_ffn2_x, &ffn_x)?;
            x = &scaled_x + &x;

            let normed_pool = self.norm_ffn2_pool.as_ref().unwrap().forward(&pool)?;
            let ffn_pool = self.ffn2_pool.as_mut().unwrap().forward(&normed_pool)?;
            let scaled_pool = Self::apply_ls(&self.lay_scale_ffn2_pool, &ffn_pool)?;
            pool = &scaled_pool + &pool;
        }

        Ok((x, pool))
    }
}

// ---------------------------------------------------------------------------
// Pool token helpers
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters)]
pub struct LearnablePoolTokens {
    #[param]
    pub emb: Param<Array>,
    pub region_token_num: i32,
}

impl LearnablePoolTokens {
    pub fn new(dim: i32, region_token_num: i32) -> Result<Self> {
        let emb = &mlx_rs::random::normal::<f32>(&[region_token_num, dim], None, None, None)?
            * Array::from_f32(0.02);
        Ok(Self {
            emb: Param::new(emb),
            region_token_num,
        })
    }

    pub fn forward(&self, _x: &Array, _regions: &Array, n_mask: &Array) -> Result<Array> {
        let n_shape = n_mask.shape();
        let b = n_shape[0];
        let n = n_shape[1] as i32;
        let r = self.region_token_num;

        // emb: [R, C] -> [1, 1, R, C] -> tile [B, N, R, C]
        let emb_4d = self.emb.reshape(&[1, 1, r, self.emb.shape()[1] as i32])?;
        let tokens = mlx_rs::ops::tile(&emb_4d, &[b, n, 1, 1])?; // [B, N, R, C]

        // Mask: n_mask [B, N] -> [B, N, 1, 1]
        let mask = mlx_rs::ops::expand_dims(n_mask, -1)?;
        let mask = mlx_rs::ops::expand_dims(&mask, -1)?.as_dtype(Dtype::Float32)?;
        let tokens = &tokens * &mask;

        let c = tokens.shape()[3] as i32;
        Ok(tokens.reshape(&[b, n * r, c])?)
    }
}

// ---------------------------------------------------------------------------
// PoolTokenMerger (mean mode)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters)]
pub struct PoolTokenMerger {
    pub r: i32,
}

impl PoolTokenMerger {
    pub fn new(_dim: i32, region_token_num: i32) -> Self {
        Self {
            r: region_token_num,
        }
    }

    pub fn forward(&self, pool: &Array, n_mask: &Array) -> Result<Array> {
        let shape = pool.shape();
        let b = shape[0];
        let p = shape[1];
        let c = shape[2] as i32;
        let r = self.r;
        let n = p as i32 / r;

        let reshaped = pool.reshape(&[b, n, r, c])?;
        let merged = reshaped.mean_axis(2, None)?; // [B, N, C]

        let mask = mlx_rs::ops::expand_dims(n_mask, -1)?.as_dtype(Dtype::Float32)?;
        Ok(&merged * &mask)
    }
}

// ---------------------------------------------------------------------------
// JEBFBackbone
// ---------------------------------------------------------------------------

pub enum AttnMask {
    Joint(Array),
    Split(SplitAttnMasks),
}

#[derive(Debug, Clone, ModuleParameters)]
pub struct JEBFBackbone {
    #[param]
    pub input_proj: nn::Linear,
    #[param]
    pub pool_token_gen: LearnablePoolTokens,
    #[param]
    pub layers: Vec<JEBF>,
    #[param]
    pub output_norm_x: Option<RMSNorm>,
    #[param]
    pub output_norm_pool: Option<RMSNorm>,
    #[param]
    pub output_proj_x: nn::Linear,
    #[param]
    pub output_proj_pool: nn::Linear,
    #[param]
    pub region_bias: Option<RegionBias>,
    pub pool_merger: Option<PoolTokenMerger>,
    pub region_token_num: i32,
    pub use_out_norm: bool,
    pub pool_out_dim: i32,
    pub attn_type: String,
    pub use_region_bias: bool,
}

impl JEBFBackbone {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        in_dim: i32,
        out_dim: i32,
        dim: i32,
        num_layers: i32,
        num_heads: i32,
        head_dim: i32,
        region_token_num: i32,
        c_kernel_size_pool: i32,
        m_kernel_size_pool: i32,
        c_kernel_size_x: i32,
        m_kernel_size_x: i32,
        qk_norm: bool,
        use_rope: bool,
        rope_mode: &str,
        use_pool_offset: bool,
        theta: f32,
        use_ls: bool,
        skip_first_ffn: bool,
        skip_out_ffn: bool,
        use_out_norm: bool,
        pool_out_dim: Option<i32>,
        attn_type: &str,
        use_region_bias: bool,
        bias_alpha: f32,
        bias_learnable: bool,
    ) -> Result<Self> {
        let pool_out_dim = pool_out_dim.unwrap_or(out_dim);

        let region_bias = if use_region_bias {
            Some(RegionBias::new(bias_alpha, bias_learnable))
        } else {
            None
        };

        let pool_merger = if region_token_num > 1 {
            Some(PoolTokenMerger::new(dim, region_token_num))
        } else {
            None
        };

        let input_proj = nn::LinearBuilder::new(in_dim, dim).build()?;
        let pool_token_gen = LearnablePoolTokens::new(dim, region_token_num)?;

        let mut layers = Vec::with_capacity(num_layers as usize);
        for _ in 0..num_layers {
            layers.push(JEBF::new(
                dim, num_heads, head_dim,
                c_kernel_size_pool, m_kernel_size_pool,
                c_kernel_size_x, m_kernel_size_x,
                region_token_num, qk_norm,
                use_rope, rope_mode, use_pool_offset, theta,
                skip_first_ffn, skip_out_ffn, use_ls,
                attn_type,
            )?);
        }

        let (output_norm_x, output_norm_pool) = if use_out_norm {
            (
                Some(RMSNorm::new(dim, 1.0, 1e-6)),
                Some(RMSNorm::new(dim, 1.0, 1e-6)),
            )
        } else {
            (None, None)
        };

        let output_proj_x = nn::LinearBuilder::new(dim, out_dim).build()?;
        let output_proj_pool = nn::LinearBuilder::new(dim, pool_out_dim).build()?;

        Ok(Self {
            input_proj,
            pool_token_gen,
            layers,
            output_norm_x,
            output_norm_pool,
            output_proj_x,
            output_proj_pool,
            region_bias,
            pool_merger,
            region_token_num,
            use_out_norm,
            pool_out_dim,
            attn_type: attn_type.to_string(),
            use_region_bias,
        })
    }

    fn build_region_bias_mask(
        &self,
        regions: &Array,
        region_token_num: i32,
        t_mask: &Array,
        n_mask: &Array,
    ) -> Result<Array> {
        let shape = regions.shape();
        let b = shape[0];
        let t = shape[1];
        let n = n_mask.shape()[1] as i32;
        let r = region_token_num;
        let p = n * r;

        // pool_region: [B, P]
        let arange_n_vec: Vec<i32> = (1..=n).collect();
        let arange_n = Array::from_slice(&arange_n_vec, &[n]);
        let arange_n_col = arange_n.reshape(&[n, 1])?;
        let pool_region_nr = mlx_rs::ops::broadcast_to(&arange_n_col, &[n, r])?;
        let pool_region_row = pool_region_nr.reshape(&[1, p])?;
        let pool_region = mlx_rs::ops::broadcast_to(&pool_region_row, &[b, p])?;

        let full_region = mlx_rs::ops::concatenate_axis(&[&pool_region, regions], -1)?;

        let n_mask_exp = mlx_rs::ops::expand_dims(n_mask, -1)?;
        let pool_valid_bnr = mlx_rs::ops::broadcast_to(&n_mask_exp, &[b, n, r])?;
        let pool_valid = pool_valid_bnr.reshape(&[b, p])?;
        let full_valid = mlx_rs::ops::concatenate_axis(&[&pool_valid, t_mask], -1)?;

        let ones_bp = mlx_rs::ops::ones::<bool>(&[b, p])?;
        let zeros_bt = mlx_rs::ops::zeros::<bool>(&[b, t as i32])?;
        let is_pool = mlx_rs::ops::concatenate_axis(&[&ones_bp, &zeros_bt], -1)?;

        let is_pool_q = mlx_rs::ops::expand_dims(&is_pool, -1)?;
        let is_pool_k = mlx_rs::ops::expand_dims(&is_pool, -2)?;
        let same_stream = is_pool_q.eq(&is_pool_k)?;

        let fv_q = mlx_rs::ops::expand_dims(&full_valid, -1)?;
        let fv_k = mlx_rs::ops::expand_dims(&full_valid, -2)?;
        let valid_pair = fv_q.logical_and(&fv_k)?;

        let zero = Array::from_f32(0.0);
        let neg_val = Array::from_f32(-10000.0);
        let base_mask = mlx_rs::ops::r#where(&valid_pair, &zero, &neg_val)?;

        let rb = self.region_bias.as_ref().unwrap();
        let region_decay = rb.forward(&full_region, &full_region)?;
        let region_decay_3d = region_decay.squeeze_axes(&[1])?; // [B, L, L]

        let attn_bias = mlx_rs::ops::r#where(
            &same_stream,
            &base_mask,
            &(&base_mask + &region_decay_3d),
        )?;
        let out = mlx_rs::ops::expand_dims(&attn_bias, 1)?; // [B, 1, L, L]
        Ok(out)
    }

    pub fn forward(
        &mut self,
        x: &Array,
        regions: &Array,
        t_mask: &Array,
        n_mask: &Array,
    ) -> Result<(Array, Array)> {
        let r = self.region_token_num;

        let mut x = self.input_proj.forward(x)?;
        let mut pool = self.pool_token_gen.forward(&x, regions, n_mask)?;

        if self.attn_type == "joint" {
            let attn_mask = if self.use_region_bias {
                self.build_region_bias_mask(regions, r, t_mask, n_mask)?
            } else {
                build_join_attention_mask(regions, r, t_mask, n_mask)?
            };

            for layer in self.layers.iter_mut() {
                let (x_new, pool_new) =
                    layer.forward_joint(&pool, &x, regions, t_mask, n_mask, &attn_mask)?;
                x = x_new;
                pool = pool_new;
            }
        } else {
            let region_bias_ref = if self.use_region_bias {
                self.region_bias.as_ref()
            } else {
                None
            };
            let attn_masks =
                build_split_attention_masks(regions, r, t_mask, n_mask, region_bias_ref)?;

            for layer in self.layers.iter_mut() {
                let (x_new, pool_new) =
                    layer.forward_split(&pool, &x, regions, t_mask, n_mask, &attn_masks)?;
                x = x_new;
                pool = pool_new;
            }
        }

        if self.use_out_norm {
            x = self.output_norm_x.as_ref().unwrap().forward(&x)?;
            pool = self.output_norm_pool.as_ref().unwrap().forward(&pool)?;
        }

        if let Some(ref merger) = self.pool_merger {
            pool = merger.forward(&pool, n_mask)?;
        }

        let out_x = self.output_proj_x.forward(&x)?;
        let out_pool = self.output_proj_pool.forward(&pool)?;
        Ok((out_x, out_pool))
    }
}
