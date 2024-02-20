use anyhow::Result;
use bhop::{BhopConfig, SimpleModel};
use candle_core::DType;
use candle_nn::{init, Optimizer};
use env_logger::Builder;
use log::{info, warn, LevelFilter};
use optimisers::{
    adam::{Adam, ParamsAdam},
    lbfgs::LineSearch,
    Model,
};
use training::setup_training;

mod load_dataset;
mod models;
mod training;

const DATATYPE: DType = DType::BF16;
fn main() -> Result<()> {
    let mut builder = Builder::new();
    // builder.format_timestamp(None);
    builder.format_target(false);
    builder.filter(None, LevelFilter::Info);
    builder.init();
    // let m = load_data().context("Failed to load data")?;

    let (model, varmap) = setup_training()?;

    let l2_reg = Some(5e-9);
    let temperature = 1.;
    let pert_range = 1.;
    let lbfgs_steps = 200_000;

    let weight_decay = l2_reg.map(|l| optimisers::Decay::WeightDecay(l * 2.));
    let adam_params = ParamsAdam {
        lr: 5e-4,
        beta_1: 0.9,
        beta_2: 0.999,
        eps: 1e-8,
        weight_decay,
        amsgrad: true,
    };
    let mut optimiser = Adam::new(varmap.all_vars(), adam_params)?;
    let mut initial_loss = model.loss()?.to_dtype(DType::F32)?.to_scalar::<f32>()?;
    for i in 1..=100_000 {
        let loss = model.loss()?;
        optimiser.backward_step(&loss)?;
        info!(
            "{:4} train loss: {:8.5}, test loss {:8.5}",
            i,
            loss.to_dtype(candle_core::DType::F32)?.to_scalar::<f32>()?,
            model.test_eval()?
        );
        if i % 100 == 0 {
            let loss = model
                .loss()?
                .to_dtype(candle_core::DType::F32)?
                .to_scalar::<f32>()?;
            if loss < initial_loss {
                varmap.save("alz_weights.st")?;
                warn!("New saved loss: {}", loss);
                initial_loss = loss;
            }
        }
        if i % 1000 == 0 {
            let lr = optimiser.learning_rate();
            optimiser.set_learning_rate(0.75 * lr);
        }
    }
    let loss = model
        .loss()?
        .to_dtype(candle_core::DType::F32)?
        .to_scalar::<f32>()?;
    println!("Final loss {loss}");

    if loss < initial_loss {
        varmap.save("alz_weights.st")?;
    }

    // let config = BhopConfig {
    // steps: 1,
    // temperature,
    // step_size: pert_range,
    // lbfgs_steps,
    // step_conv: optimisers::lbfgs::StepConv::MinStep(0.),
    // grad_conv: optimisers::lbfgs::GradConv::MinForce(1e-4),
    // history_size: 10,
    // l2_reg,
    // linesearch: Some(LineSearch::StrongWolfe(1e-4, 0.9, 1e-9)),
    // seed: 42,
    // };
    //
    // let _names = bhop::basin_hopping(&model, varmap, "alzheimer_weights", config)?;

    Ok(())
}
