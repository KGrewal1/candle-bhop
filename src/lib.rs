/*!
 Basin Hopping optimisation for use with the candle machine learning framework
*/

use candle_core::{Tensor, Var};
use candle_nn::{VarBuilder, VarMap};
use log::info;
use optimisers::{
    lbfgs::{GradConv, LineSearch, ParamsLBFGS, StepConv},
    Model,
};
use rand::{Rng, SeedableRng};
use std::{fs, path::Path};

use crate::training::{l2_norm, run_lbfgs_training};
pub mod training;

/// Trait needed for the model to be used in the basin hopping optimisation
/// Requires a new function to create a new model from a variable builder and data
/// and a test evaluation function
pub trait SimpleModel: Sized + Model + Clone {
    type SetupVars: Sized;

    /// Create a new model from a variable builder and data
    fn new(vs: VarBuilder, setup_vars: Self::SetupVars) -> candle_core::Result<Self>;
    /// Test evaluation
    fn test_eval(&self) -> candle_core::Result<f32>;
}

/// test
pub struct BhopConfig {
    /// the number of basin hopping steps
    pub steps: usize,
    /// the temperature for the MC criterion
    pub temperature: f64,
    /// The size of each basin hopping step
    pub step_size: f64,
    /// The number of lbfgs steps
    pub lbfgs_steps: usize,
    /// The step convergence criterion
    pub step_conv: StepConv,
    /// The gradient convergence criterion
    pub grad_conv: GradConv,
    /// The history size for the lbfgs optimiser
    pub history_size: usize,
    /// The L2 regularisation factor
    pub l2_reg: Option<f64>,
    /// the line search method
    pub linesearch: Option<LineSearch>,
    /// The random seed to be used for the Monte Carlo eval
    pub seed: u64,
}

/// Run basin hopping global minimisation
pub fn basin_hopping<M: SimpleModel, P: AsRef<Path>>(
    model: &M,
    mut varmap: VarMap,
    path: P,
    config: BhopConfig,
) -> anyhow::Result<Vec<String>> {
    let path: &Path = path.as_ref();
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

    let mut min_loss = f64::INFINITY;
    let mut min_name = " ".to_string();
    let mut names: Vec<String> = Vec::new();

    let mut current_loss = f64::INFINITY;
    let mut current_name = " ".to_string();
    let mut rng = rand_xoshiro::Xoshiro256StarStar::seed_from_u64(config.seed);

    for i in 0..config.steps {
        info!("Epoch {}", i);
        let name = format!("model_{:03}.st", i);
        let save_path = path.join(&name);
        let lbfgs_params = ParamsLBFGS {
            lr: 1.,
            history_size: config.history_size,
            line_search: config.linesearch,
            step_conv: config.step_conv,
            grad_conv: config.grad_conv,
            weight_decay: config.l2_reg.map(|x| 2. * x),
        };
        let f = run_lbfgs_training(model, &varmap, lbfgs_params, config.lbfgs_steps)?;

        #[allow(clippy::cast_possible_truncation)]
        let l2_fac = if let Some(reg) = config.l2_reg {
            (l2_norm(&varmap.all_vars())? * reg) as f64
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
            let p = (-delta / config.temperature).exp(); // T = temp in units of Kb so P = exp(-delta/T)
            let n = rng.gen_range(0_f64..1.);
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
        perturb(&mut varmap.all_vars(), config.step_size)?;
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
