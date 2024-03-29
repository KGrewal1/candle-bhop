use anyhow::Result;
use bhop::BhopConfig;
use candle_core::DType;
use env_logger::Builder;
use log::LevelFilter;
use optimisers::lbfgs::LineSearch;
use training::setup_lbfgs_training;

use crate::misclassification::perturbed_misclassification;

mod load_mnist;
mod misclassification;
mod models;
mod training;

const DATATYPE: DType = DType::F32;
fn main() -> Result<()> {
    let mut builder = Builder::new();
    // builder.format_timestamp(None);
    builder.format_target(false);
    builder.filter(None, LevelFilter::Info);
    builder.init();
    // let m = load_data().context("Failed to load data")?;

    let (model, varmap) = setup_lbfgs_training()?;
    let l2_reg = Some(1e-5);
    let temperature = 0.05;
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
        linesearch: Some(LineSearch::StrongWolfe(1e-4, 0.9, 1e-9)),
        seed: 42,
    };

    let names = bhop::basin_hopping(&model, varmap, "mlp_weights", config)?;

    for name in &names {
        for name2 in &names {
            perturbed_misclassification(name, name2, "mlp_weights")?;
        }
    }

    Ok(())
}
