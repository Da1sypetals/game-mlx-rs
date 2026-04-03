use anyhow::Result;
use mlx_rs::Array;
use mlx_rs::Dtype;

/// Cosine noise schedule: p = (1 + cos(t * pi)) * 0.5
pub fn d3pm_time_schedule(t: &Array) -> Result<Array> {
    let pi = Array::from_f32(std::f32::consts::PI);
    let t_pi = &(t * &pi);
    let cos_val = mlx_rs::ops::cos(t_pi)?;
    Ok(&(&cos_val + Array::from_f32(1.0)) * Array::from_f32(0.5))
}

/// Remove random boundaries with uniform probability p.
pub fn remove_boundaries(boundaries: &Array, p: &Array) -> Result<Array> {
    let p_exp = mlx_rs::ops::expand_dims(p, -1)?;
    let q = Array::from_f32(1.0) - &p_exp;
    let rnd = mlx_rs::random::uniform::<_, f32>(
        Array::from_f32(0.0),
        Array::from_f32(1.0),
        Some(boundaries.shape()),
        None,
    )?;
    let keep = rnd.le(&q)?;
    Ok(keep.logical_and(boundaries)?)
}

/// Remove mutable boundaries with adjusted probability.
pub fn remove_mutable_boundaries(
    boundaries: &Array,
    immutable: &Array,
    p: &Array,
) -> Result<Array> {
    let not_immutable = immutable.logical_not()?;
    let boundaries_mutable = boundaries.logical_and(&not_immutable)?;

    let n = boundaries.as_dtype(Dtype::Float32)?.sum_axis(-1, None)?;
    let m = boundaries_mutable.as_dtype(Dtype::Float32)?.sum_axis(-1, None)?;

    let eps = Array::from_f32(1e-8);
    let one = Array::from_f32(1.0);
    let p_adjusted = &(&n * p) / &(&m + &eps);
    let p_clamped = mlx_rs::ops::minimum(&p_adjusted, &one)?;

    let boundaries_mutable_remain = remove_boundaries(&boundaries_mutable, &p_clamped)?;
    Ok(boundaries_mutable_remain.logical_or(immutable)?)
}
