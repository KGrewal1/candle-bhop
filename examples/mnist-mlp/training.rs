use bhop::SimpleModel;
use candle_core::DType;
use candle_nn::{VarBuilder, VarMap};
use log::info;

use crate::models::Mlp;

pub fn setup_lbfgs_training(m: &candle_datasets::vision::Dataset) -> anyhow::Result<(Mlp, VarMap)> {
    // check to see if cuda device availabke
    let dev = candle_core::Device::cuda_if_available(0)?;
    info!("Training on device {dev:?}");
    // dev.set_seed(0)?;

    // get the labels from the dataset
    let train_labels = m.train_labels.to_dtype(DType::U32)?.to_device(&dev)?;
    // train_labels
    //     .to_dtype(DType::F32)?
    //     .save_safetensors("mnist_train_label", "mnist_train_label.st")?;
    // get the input from the dataset and put on device
    let train_images = m.train_images.to_device(&dev)?;
    // train_images.save_safetensors("mnist_train_images", "mnist_train_images.st")?;
    // get the labels from the dataset
    let test_labels = m.test_labels.to_dtype(DType::U32)?.to_device(&dev)?;
    // get the input from the dataset and put on device
    let test_images = m.test_images.to_device(&dev)?;

    // create a new variable store
    let varmap = VarMap::new();
    // create a new variable builder
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &dev);

    let setup = crate::models::MySetupVars {
        train_data: train_images,
        train_labels,
        test_data: test_images,
        test_labels,
    };
    // create model from variables
    let model = Mlp::new(vs.clone(), setup)?;
    Ok((model, varmap))
}
