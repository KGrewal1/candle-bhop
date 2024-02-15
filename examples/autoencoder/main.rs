use anyhow::Result;
use bhop::BhopConfig;
use candle_core::DType;
use env_logger::Builder;
use log::LevelFilter;
use training::setup_training;

mod load_mnist;
mod model;
mod training;

const DATATYPE: DType = DType::F16;
fn main() -> Result<()> {
    let mut builder = Builder::new();
    // builder.format_timestamp(None);
    builder.format_target(false);
    builder.filter(None, LevelFilter::Info);
    builder.init();

    let (model, varmap) = setup_training()?;

    let l2_reg = Some(5e-9);
    let temperature = 1.;
    let pert_range = 0.1;
    let lbfgs_steps = 20_000;

    let config = BhopConfig {
        steps: 100,
        temperature,
        step_size: pert_range,
        lbfgs_steps,
        step_conv: optimisers::lbfgs::StepConv::MinStep(0.),
        grad_conv: optimisers::lbfgs::GradConv::MinForce(1e-3),
        history_size: 10,
        l2_reg,
        seed: 42,
    };

    let _names = bhop::basin_hopping(&model, varmap, "autoencoder_weights", config)?;

    Ok(())
}
