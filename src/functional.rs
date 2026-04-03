use anyhow::Result;
use mlx_rs::Array;
use mlx_rs::Dtype;
use mlx_rs::ops::indexing::IndexOp;

/// [B, T] bool boundaries -> [B, T] int region indices starting at 1
pub fn boundaries_to_regions(boundaries: &Array, mask: Option<&Array>) -> Result<Array> {
    let regions = boundaries.as_dtype(Dtype::Int32)?.cumsum(Some(-1), None, None)? + Array::from_int(1);
    if let Some(m) = mask {
        Ok(&regions * m.as_dtype(Dtype::Int32)?)
    } else {
        Ok(regions)
    }
}

/// [B, T] region ids -> [B, T] bool boundary flags
pub fn regions_to_boundaries(regions: &Array) -> Result<Array> {
    // regions[..., 1:] > regions[..., :-1]
    let shifted = regions.index((.., ..(-1 as i32)));
    let current = regions.index((.., 1..));
    let diff = current.gt(&shifted)?;
    // Pad with False at the beginning of last dim
    let ndim = regions.ndim();
    let mut pad_widths = vec![(0i32, 0i32); ndim];
    pad_widths[ndim - 1] = (1, 0);
    Ok(mlx_rs::ops::pad(&diff, &pad_widths[..], None::<Array>, None)?)
}

/// Count frames per region. regions: [B, T], returns [B, N] frame counts.
pub fn regions_to_durations(regions: &Array, max_n: Option<i32>) -> Result<Array> {
    let max_n = match max_n {
        Some(n) => n,
        None => {
            let max_val = regions.max(None)?;
            max_val.eval()?;
            max_val.item::<i32>()
        }
    };

    let idx_vec: Vec<i32> = (1..=max_n).collect();
    let idx = Array::from_slice(&idx_vec, &[1, 1, max_n]);

    let regions_exp = mlx_rs::ops::expand_dims(regions, -1)?; // [B, T, 1]
    let eq = regions_exp.eq(&idx)?; // [B, T, N]
    let counts = eq.as_dtype(Dtype::Int32)?.sum_axis(-2, None)?; // [B, N]

    Ok(counts)
}

/// Convert note durations [B, N] to binary boundary map [B, T]
pub fn format_boundaries(durations: &Array, length: i32, timestep: f32) -> Result<Array> {
    let timestep_arr = Array::from_f32(timestep);
    let cum_dur = durations.cumsum(Some(1), None, None)?;
    let boundary_frames = mlx_rs::ops::round(&(&cum_dur / &timestep_arr), None)?.as_dtype(Dtype::Int32)?;

    // Take all but last column: [B, N-1]
    let n = boundary_frames.dim(-1) as i32;
    let bf = boundary_frames.index((.., ..(n - 1)));
    let bf_exp = mlx_rs::ops::expand_dims(&bf, 1)?; // [B, 1, N-1]

    let idx_vec: Vec<i32> = (0..length).collect();
    let idx = Array::from_slice(&idx_vec, &[1, length, 1]); // [1, T, 1]

    let eq = idx.eq(&bf_exp)?; // [B, T, N-1]
    let boundaries = eq.any_axes(&[-1], None)?; // [B, T]

    Ok(boundaries)
}

/// Pad + take_along_axis
pub fn flatten_sequences(x: &Array, idx: &Array) -> Result<Array> {
    let ndim = x.ndim();
    let mut pad_widths = vec![(0i32, 0i32); ndim];
    pad_widths[ndim - 1] = (1, 0);
    let x_pad = mlx_rs::ops::pad(x, &pad_widths[..], None::<Array>, None)?;
    Ok(x_pad.take_along_axis(idx, -1)?)
}
