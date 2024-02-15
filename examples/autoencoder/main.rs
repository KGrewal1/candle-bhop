use anyhow::Result;
use candle_core::DType;
use env_logger::Builder;
use log::LevelFilter;
use std::path::Path;
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

    let _names = bhop::basin_hopping(
        &model,
        varmap,
        l2_reg,
        Path::new("autoencoder_weights"),
        temperature,
        pert_range,
        lbfgs_steps,
        optimisers::lbfgs::StepConv::MinStep(0.),
        optimisers::lbfgs::GradConv::MinForce(1e-3),
        10,
    )?;

    Ok(())
}
