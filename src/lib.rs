use candle_core::{Tensor, Var};
use candle_nn::{VarBuilder, VarMap};
use log::info;
use optimisers::{
    lbfgs::{GradConv, StepConv},
    Model,
};
use rand::{Rng, SeedableRng};
use std::{fs, path::Path};

use crate::training::{l2_norm, run_lbfgs_training};
pub mod training;

pub trait SimpleModel: Sized + Model + Clone {
    type SetupVars: Sized;

    /// Create a new model from a variable builder and data
    fn new(vs: VarBuilder, setup_vars: Self::SetupVars) -> candle_core::Result<Self>;
    /// Test evaluation
    fn test_eval(&self) -> candle_core::Result<f32>;
}

/// Run basin hopping global minimisation
pub fn basin_hopping<M: SimpleModel>(
    model: &M,
    mut varmap: VarMap,
    l2_reg: Option<f64>,
    path: &Path,
    temperature: f32,
    pert_range: f64,
    lbfgs_steps: usize,
    step_conv: StepConv,
    grad_conv: GradConv,
    history_size: usize,
) -> anyhow::Result<Vec<String>> {
    if path.exists() {
        if !path.is_dir() {
            anyhow::bail!(
                "Path {} exists and is not a directory",
                path.to_string_lossy()
            );
        }
    } else {
        fs::create_dir_all(path)?;
    }

    let mut min_loss = f32::INFINITY;
    let mut min_name = " ".to_string();
    let mut names: Vec<String> = Vec::new();

    let mut current_loss = f32::INFINITY;
    let mut current_name = " ".to_string();
    let mut rng = rand_chacha::ChaCha8Rng::from_seed([46; 32]);
    // panic!("n: {}", n);

    // let (model, mut varmap) = setup_lbfgs_training::<M>(&m)?;

    // let temperature = 1.;
    for i in 0..100 {
        info!("Epoch {}", i);
        let name = format!("model_{:03}.st", i);
        let save_path = path.join(&name);
        let f = run_lbfgs_training(
            model,
            &varmap,
            l2_reg,
            lbfgs_steps,
            step_conv,
            grad_conv,
            history_size,
        )?;

        #[allow(clippy::cast_possible_truncation)]
        let l2_fac = if let Some(reg) = l2_reg {
            (l2_norm(&varmap.all_vars())? * reg) as f32
        } else {
            0.
        };
        info!("loss inc L2: {}", f + l2_fac);
        info!("L2 reg: {}", l2_fac);
        varmap.save(&save_path)?;

        // Metropolis Hastings
        if f + l2_fac < min_loss {
            // new minimum
            info!("new global min from {} to {}", min_loss, f + l2_fac);
            info!(
                "STEP: decrease in loss from {} to {}",
                current_loss,
                f + l2_fac
            );
            min_loss = f + l2_fac;
            min_name = name.clone();
            // by definition lower than previous value
            current_loss = f + l2_fac;
            current_name = name.clone();
        } else if f + l2_fac < current_loss {
            info!(
                "STEP: decrease in loss from {} to {}",
                current_loss,
                f + l2_fac
            );
            current_loss = f + l2_fac;
            current_name = name.clone();
        } else {
            let delta = f + l2_fac - current_loss;
            let p = (-delta / temperature).exp(); // T = temp in units of Kb so P = exp(-delta/T)
            let n = rng.gen_range(0_f32..1.);
            if n < p {
                info!("STEP: accepted MH, from {} to {}", current_loss, f + l2_fac);
                current_loss = f + l2_fac;
                current_name = name.clone();
            } else {
                // reject
                info!(
                    "NOSTEP: rejected MH, loss {}, proposed {}",
                    current_loss,
                    f + l2_fac
                );
                let current_path = path.join(&current_name);
                varmap.load(&current_path)?;
            }
        }
        names.push(name);
        perturb(&mut varmap.all_vars(), pert_range)?;
    }
    info!("final min loss: {}", min_loss);
    info!("final min name: {}\n", min_name);
    Ok(names)
}

fn perturb(vs: &mut Vec<Var>, range: f64) -> candle_core::Result<()> {
    for v in vs {
        let pert = Tensor::rand_like(v.as_tensor(), -range, range)?;
        v.set(&v.add(&pert)?)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    // use super::*;

    #[test]
    fn it_works() {
        // let result = add(2, 2);
        assert_eq!(4, 4);
    }
}
