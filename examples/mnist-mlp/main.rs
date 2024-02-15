use anyhow::{Context, Result};
use bhop::BhopConfig;
use candle_core::Result as CResult;
use candle_datasets::vision::Dataset;
use env_logger::Builder;
use log::LevelFilter;
use std::path::Path;
use training::setup_lbfgs_training;

use crate::misclassification::perturbed_misclassification;

mod misclassification;
mod models;
mod training;

fn load_data() -> CResult<Dataset> {
    candle_datasets::vision::mnist::load()
}

fn main() -> Result<()> {
    let mut builder = Builder::new();
    // builder.format_timestamp(None);
    builder.format_target(false);
    builder.filter(None, LevelFilter::Info);
    builder.init();
    let m = load_data().context("Failed to load data")?;

    let (model, varmap) = setup_lbfgs_training(&m)?;
    let l2_reg = Some(5e-9);
    let temperature = 1.;
    let pert_range = 1.;
    let lbfgs_steps = 200_000;

    let config = BhopConfig {
        steps: 100,
        temperature,
        step_size: pert_range,
        lbfgs_steps,
        step_conv: optimisers::lbfgs::StepConv::MinStep(0.),
        grad_conv: optimisers::lbfgs::GradConv::MinForce(1e-4),
        history_size: 10,
        l2_reg,
        seed: 42,
    };

    let names = bhop::basin_hopping(&model, varmap, "mlp_weights", config)?;

    for name in &names {
        for name2 in &names {
            perturbed_misclassification(&m, name, name2, Path::new("new_weights"))?;
        }
    }

    Ok(())
}
