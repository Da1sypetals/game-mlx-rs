use anyhow::Result;
use mlx_rs::Array;
use mlx_rs::module::Param;
use mlx_rs::nn;
use mlx_rs::macros::ModuleParameters;

// ---------------------------------------------------------------------------
// HalfCacheGLUFFN (eval-only path: standard forward)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, ModuleParameters)]
pub struct HalfCacheGLUFFN {
    #[param]
    pub w_gate: Param<Array>,
    #[param]
    pub w_up: Param<Array>,
    #[param]
    pub w_down: Param<Array>,
    #[param]
    pub bias_gate: Param<Option<Array>>,
    #[param]
    pub bias_up: Param<Option<Array>>,
    #[param]
    pub bias_down: Param<Option<Array>>,
    pub gate_type: String,
}

impl HalfCacheGLUFFN {
    pub fn new(d_model: i32, d_ff: i32, gate_type: &str, bias: bool) -> Result<Self> {
        let scale = (1.0f32 / d_model as f32).sqrt();
        let w_gate = mlx_rs::random::uniform::<_, f32>(-scale, scale, &[d_ff, d_model], None)?;
        let w_up = mlx_rs::random::uniform::<_, f32>(-scale, scale, &[d_ff, d_model], None)?;
        let w_down = mlx_rs::random::uniform::<_, f32>(-scale, scale, &[d_model, d_ff], None)?;

        let (bias_gate, bias_up, bias_down) = if bias {
            (
                Some(mlx_rs::ops::zeros::<f32>(&[d_ff])?),
                Some(mlx_rs::ops::zeros::<f32>(&[d_ff])?),
                Some(mlx_rs::ops::zeros::<f32>(&[d_model])?),
            )
        } else {
            (None, None, None)
        };

        Ok(Self {
            w_gate: Param::new(w_gate),
            w_up: Param::new(w_up),
            w_down: Param::new(w_down),
            bias_gate: Param::new(bias_gate),
            bias_up: Param::new(bias_up),
            bias_down: Param::new(bias_down),
            gate_type: gate_type.to_string(),
        })
    }

    fn gate_fn(x: &Array, gate_type: &str) -> Result<Array> {
        match gate_type {
            "silu" => Ok(nn::silu(x)?),
            "sigmoid" => Ok(nn::sigmoid(x)?),
            "gelu" => Ok(nn::gelu(x)?),
            _ => anyhow::bail!("Unknown gate type: {}", gate_type),
        }
    }

    pub fn forward(&self, x: &Array) -> Result<Array> {
        let mut gate = x.matmul(self.w_gate.t())?;
        if let Some(bg) = &self.bias_gate.value {
            gate = &gate + bg;
        }
        let mut up = x.matmul(self.w_up.t())?;
        if let Some(bu) = &self.bias_up.value {
            up = &up + bu;
        }
        let gate_act = Self::gate_fn(&gate, &self.gate_type)?;
        let hidden = &gate_act * &up;
        let mut out = hidden.matmul(self.w_down.t())?;
        if let Some(bd) = &self.bias_down.value {
            out = &out + bd;
        }
        Ok(out)
    }
}
