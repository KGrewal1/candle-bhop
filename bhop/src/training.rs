use candle_core::Var;
use candle_nn::VarMap;
use log::{debug, info, warn};
use optimisers::lbfgs::{GradConv, Lbfgs, LineSearch, ParamsLBFGS, StepConv};
use optimisers::LossOptimizer;

use crate::SimpleModel;

pub(super) fn run_lbfgs_training<M: SimpleModel>(
    model: &M,
    varmap: &VarMap,
    l2_norm: Option<f64>,
    lbfgs_steps: usize,
    grad_conv: GradConv,
    history_size: usize,
) -> anyhow::Result<f32> {
    let params = ParamsLBFGS {
        lr: 1.,
        history_size,
        line_search: Some(LineSearch::StrongWolfe(1e-4, 0.9, 1e-9)),
        step_conv: StepConv::MinStep(0.),
        grad_conv: grad_conv,
        weight_decay: l2_norm.map(|x| 2. * x), //
                                               // ..Default::default()
    };

    let mut loss = model.loss()?;
    info!(
        "iniital loss: {}",
        loss.to_dtype(candle_core::DType::F32)?.to_scalar::<f32>()?
    );

    // create an optimiser
    let mut optimiser = Lbfgs::new(varmap.all_vars(), params, model.clone())?;
    let mut fn_evals = 1;
    let mut converged = false;

    for step in 0..lbfgs_steps {
        // if step % 1000 == 0 {
        //     info!("step: {}", step);
        //     info!("loss: {}", loss.to_scalar::<f32>()?);
        //     info!("test acc: {:5.2}%", model.test_eval()? * 100.);
        // }
        // get the loss

        // step the tensors by backpropagating the loss
        let res = optimiser.backward_step(&loss)?;
        match res {
            optimisers::ModelOutcome::Converged(new_loss, evals) => {
                info!("step: {}", step);
                info!(
                    "loss: {}",
                    new_loss
                        .to_dtype(candle_core::DType::F32)?
                        .to_scalar::<f32>()?
                );
                info!("test acc: {:5.2}%", model.test_eval()? * 100.);
                fn_evals += evals;
                loss = new_loss;
                converged = true;
                info!("converged after {} fn evals", fn_evals);
                break;
            }
            optimisers::ModelOutcome::Stepped(new_loss, evals) => {
                debug!("step: {}", step);
                debug!(
                    "loss: {}",
                    new_loss
                        .to_dtype(candle_core::DType::F32)?
                        .to_scalar::<f32>()?
                );
                debug!("test acc: {:5.2}%", model.test_eval()? * 100.);
                fn_evals += evals;
                loss = new_loss;
            }
        }
    }
    if !converged {
        info!("test acc: {:5.2}%", model.test_eval()? * 100.);
        warn!("did not converge after {} fn evals", fn_evals)
    }
    info!(
        "loss: {}",
        loss.to_dtype(candle_core::DType::F32)?.to_scalar::<f32>()?
    );
    info!("{} fn evals", fn_evals);
    Ok(loss.to_dtype(candle_core::DType::F32)?.to_scalar::<f32>()?)
}

pub(super) fn l2_norm(vs: &[Var]) -> candle_core::Result<f64> {
    let mut norm = 0.;
    for v in vs {
        norm += v
            .as_tensor()
            .powf(2.)?
            .sum_all()?
            .to_dtype(candle_core::DType::F64)?
            .to_scalar::<f64>()?;
    }
    Ok(norm)
}
