use anyhow::Result;
use mlx_rs::Array;
use mlx_rs::Dtype;
use mlx_rs::ops::indexing::{IndexOp, argmax_axis, argmin_axis};

/// Sliding window along last dim via as_strided.
pub fn unfold_last(x: &Array, size: i32) -> Result<Array> {
    let shape = x.shape();
    let ndim = shape.len();
    let t = shape[ndim - 1];

    let mut orig_strides: Vec<i64> = vec![0; ndim];
    orig_strides[ndim - 1] = 1;
    for i in (0..ndim - 1).rev() {
        orig_strides[i] = orig_strides[i + 1] * shape[i + 1] as i64;
    }

    let new_t = t - size + 1;
    let mut new_shape: Vec<i32> = shape[..ndim - 1].to_vec();
    new_shape.push(new_t);
    new_shape.push(size);

    let mut new_strides = orig_strides[..ndim - 1].to_vec();
    new_strides.push(1);
    new_strides.push(1);

    Ok(mlx_rs::ops::as_strided(x, &new_shape, &new_strides, 0)?)
}

/// Find local extrema in last dimension.
pub fn find_local_extremum(
    x: &Array,
    threshold: Option<f32>,
    radius: i32,
    maxima: bool,
) -> Result<Array> {
    let inf_val = if maxima {
        f32::INFINITY
    } else {
        f32::NEG_INFINITY
    };

    let ndim = x.ndim();
    let mut pad_widths = vec![(0i32, 0i32); ndim];
    pad_widths[ndim - 1] = (radius, radius);
    let inf_arr = Array::from_f32(inf_val);
    let x_pad = mlx_rs::ops::pad(x, &pad_widths[..], Some(inf_arr), None)?;

    let windows = unfold_last(&x_pad, 2 * radius + 1)?;
    let radius_arr = Array::from_int(radius);

    if maxima {
        let am = argmax_axis(&windows, -1, None)?;
        let result = am.eq(&radius_arr)?;
        if let Some(thr) = threshold {
            let above = x.ge(&Array::from_f32(thr))?;
            Ok(result.logical_and(&above)?)
        } else {
            Ok(result)
        }
    } else {
        let am = argmin_axis(&windows, -1, None)?;
        let result = am.eq(&radius_arr)?;
        if let Some(thr) = threshold {
            let below = x.le(&Array::from_f32(thr))?;
            Ok(result.logical_and(&below)?)
        } else {
            Ok(result)
        }
    }
}

pub fn decode_soft_boundaries(
    boundaries: &Array,
    barriers: Option<&Array>,
    mask: Option<&Array>,
    threshold: f32,
    radius: i32,
) -> Result<Array> {
    let pos_inf = Array::from_f32(f32::INFINITY);
    let mut b = boundaries.clone();

    if let Some(m) = mask {
        b = mlx_rs::ops::r#where(m, &b, &pos_inf)?;
    }
    if let Some(bar) = barriers {
        let bar_fill = Array::full::<f32>(b.shape(), &pos_inf)?;
        b = mlx_rs::ops::r#where(bar, &bar_fill, &b)?;
    }

    let maxima = find_local_extremum(&b, Some(threshold), radius, true)?;

    if let Some(m) = mask {
        Ok(maxima.logical_and(m)?)
    } else {
        Ok(maxima)
    }
}

pub fn decode_gaussian_blurred_probs(
    probs: &Array,
    min_val: f32,
    max_val: f32,
    deviation: f32,
    threshold: f32,
) -> Result<(Array, Array)> {
    let ndim = probs.ndim();
    let n = probs.dim(-1) as i32;
    let width = ((deviation / (max_val - min_val) * (n - 1) as f32).ceil()) as i32;

    let idx_vec: Vec<i32> = (0..n).collect();
    let mut idx_shape = vec![1i32; ndim];
    idx_shape[ndim - 1] = n;
    let idx = Array::from_slice(&idx_vec, &idx_shape);

    let center_values =
        Array::linspace::<f32, f32>(min_val, max_val, Some(n))?.reshape(&idx_shape)?;

    let centers = argmax_axis(probs, -1, Some(true))?;

    let zero = Array::from_int(0);
    let n_arr = Array::from_int(n);
    let width_arr = Array::from_int(width);

    let start = mlx_rs::ops::maximum(&(&centers - &width_arr), &zero)?;
    let end = mlx_rs::ops::minimum(&(&centers + &width_arr + Array::from_int(1)), &n_arr)?;

    let ge_start = idx.ge(&start)?;
    let lt_end = idx.lt(&end)?;
    let idx_masks = ge_start.logical_and(&lt_end)?;

    let idx_masks_f = idx_masks.as_dtype(Dtype::Float32)?;
    let weights = probs * &idx_masks_f;

    let product_sum = &weights * &center_values;
    let product_sum = product_sum.sum_axis(-1, None)?;
    let weight_sum = weights.sum_axis(-1, None)?;

    let eps = Array::from_f32(1e-8);
    let values = &product_sum / &(&weight_sum + &eps);

    let max_probs = probs.max_axis(-1, None)?;
    let presence = max_probs.ge(&Array::from_f32(threshold))?;

    Ok((values, presence))
}
