//! The U-Net, exported to ONNX by the Python side, run on the CPU with tract.

use crate::store::{Composite, CHANNELS, CORE, MARGIN, NODATA};
use anyhow::{Context, Result};
use std::path::Path;
use tract_onnx::prelude::*;

pub const INPUT: usize = CORE + 2 * MARGIN; // 256

pub struct Model {
    plan: std::sync::Arc<TypedRunnableModel>,
    mean: Vec<f32>,
    std: Vec<f32>,
    pub classes: Vec<String>,
}

impl Model {
    pub fn load(dir: &Path) -> Result<Model> {
        let norm: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("norm.json"))?)?;
        let floats = |k: &str| -> Result<Vec<f32>> {
            norm[k].as_array().with_context(|| format!("norm.json: {k}"))?.iter().map(|v| v.as_f64().map(|x| x as f32).context("number")).collect()
        };
        let classes = norm["classes"].as_array().context("classes")?.iter().map(|v| v.as_str().unwrap_or("").to_string()).collect();
        let plan = tract_onnx::onnx()
            .model_for_path(dir.join("unet.onnx"))?
            .with_input_fact(0, f32::fact([1, CHANNELS, INPUT, INPUT]).into())?
            .into_optimized()?
            .into_runnable()?;
        Ok(Model { plan, mean: floats("mean")?, std: floats("std")?, classes })
    }

    /// Normalised input for the block whose core starts at (row, col): the same transformation
    /// as the Python side (standardised log reflectance; 0 where any channel has no data, and
    /// outside the region).
    pub fn input(&self, comp: &Composite, row: usize, col: usize) -> (Tensor, Vec<bool>) {
        let n = comp.size as isize;
        let mut x = vec![0f32; CHANNELS * INPUT * INPUT];
        let mut valid = vec![false; INPUT * INPUT];
        for r in 0..INPUT {
            let rr = row as isize + r as isize - MARGIN as isize;
            if rr < 0 || rr >= n {
                continue;
            }
            for c in 0..INPUT {
                let cc = col as isize + c as isize - MARGIN as isize;
                if cc < 0 || cc >= n {
                    continue;
                }
                let (rr, cc) = (rr as usize, cc as usize);
                if (0..CHANNELS).any(|ch| comp.at(ch, rr, cc) == 0) {
                    continue;
                }
                valid[r * INPUT + c] = true;
                for ch in 0..CHANNELS {
                    let v = ((comp.at(ch, rr, cc) as f32).ln_1p() - self.mean[ch]) / self.std[ch];
                    x[(ch * INPUT + r) * INPUT + c] = v;
                }
            }
        }
        (tract_ndarray::Array4::from_shape_vec((1, CHANNELS, INPUT, INPUT), x).unwrap().into_tensor(), valid)
    }

    /// Classes of the CORE x CORE block at (row, col); NODATA where there is no imagery.
    pub fn classify_block(&self, comp: &Composite, row: usize, col: usize) -> Result<Vec<u8>> {
        let (x, valid) = self.input(comp, row, col);
        let mut out = self.plan.run(tvec!(x.into()))?;
        let out = out.remove(0).into_tensor();
        let logits = out.to_plain_array_view::<f32>()?;
        let k = logits.shape()[1];
        let mut cls = vec![NODATA; CORE * CORE];
        for r in 0..CORE {
            for c in 0..CORE {
                let (ir, ic) = (r + MARGIN, c + MARGIN);
                if !valid[ir * INPUT + ic] {
                    continue;
                }
                let mut best = 0;
                for j in 1..k {
                    if logits[[0, j, ir, ic]] > logits[[0, best, ir, ic]] {
                        best = j;
                    }
                }
                cls[r * CORE + c] = best as u8;
            }
        }
        Ok(cls)
    }
}
