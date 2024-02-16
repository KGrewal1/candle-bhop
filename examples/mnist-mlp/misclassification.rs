use std::path::Path;

use bhop::SimpleModel;
use candle_core::{DType, Tensor, D};
use candle_nn::{VarBuilder, VarMap};
use log::info;

use crate::{
    load_mnist,
    models::{Forward, Mlp},
};

pub fn perturbed_misclassification<P: AsRef<Path>>(
    // m: &candle_datasets::vision::Dataset,
    weight_path_a: &str,
    weight_path_b: &str,
    path: P,
) -> anyhow::Result<()> {
    let path: &Path = path.as_ref();
    // check to see if cuda device availabke
    let dev = candle_core::Device::cuda_if_available(0)?;
    // println!("Training on device {dev:?}");
    let (train_images, train_labels, _test_images, _test_labels) = load_mnist::load_mnist()?;
    // load the test images
    let train_images = train_images.to_device(&dev)?.to_device(&dev)?;
    let train_labels = train_labels.to_dtype(DType::U32)?.to_device(&dev)?;
    let random_data = Tensor::randn_like(&train_images, 0., 1.)?;
    // let random_data = (random_data + &train_images)?.clamp(0., 255.)?;

    // create a new variable store
    let mut varmap_a = VarMap::new();
    // create a new variable builder
    let vs_a = VarBuilder::from_varmap(&varmap_a, DType::F32, &dev);

    // create model from variables
    let setup = crate::models::MySetupVars {
        train_data: random_data.clone(),
        train_labels: random_data.clone(),
        test_data: train_images.clone(),
        test_labels: train_labels.clone(),
    };

    let model_a = Mlp::new(vs_a.clone(), setup)?;

    // info!("loading weights from {weight_path_a}");
    varmap_a.load(path.join(weight_path_a))?;

    // get the log probabilities of the test images
    let logits_a = model_a.forward()?;
    // get the sum of the correct predictions
    let preds_a = logits_a.argmax(D::Minus1)?;

    // create a new variable store for model b
    let mut varmap_b = VarMap::new();
    // create a new variable builder
    let vs_b = VarBuilder::from_varmap(&varmap_b, DType::F32, &dev);
    // create model from variables
    let setup = crate::models::MySetupVars {
        train_data: random_data.clone(),
        train_labels: random_data.clone(),
        test_data: train_images.clone(),
        test_labels: train_labels.clone(),
    };
    let model_b = Mlp::new(vs_b.clone(), setup)?;

    // info!("loading weights from {weight_path_b}");
    varmap_b.load(path.join(weight_path_b))?;

    // get the log probabilities of the test images
    let logits_b = model_b.forward()?;
    // get the sum of the correct predictions
    let preds_b = logits_b.argmax(D::Minus1)?;

    // get the pairwise misclassification
    let sum_concordant = preds_a
        .eq(&preds_b)?
        .to_dtype(DType::F32)?
        .sum_all()?
        .to_scalar::<f32>()?;

    #[allow(clippy::cast_precision_loss)]
    let concordance = sum_concordant / random_data.dims()[0] as f32;

    info!(
        " pairwise misclassification {} {}: {:5.2}%",
        weight_path_a,
        weight_path_b,
        100. * concordance
    );

    Ok(())
}
