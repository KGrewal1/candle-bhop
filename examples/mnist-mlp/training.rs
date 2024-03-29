use bhop::SimpleModel;
use candle_core::DType;
use candle_nn::{VarBuilder, VarMap};
use log::info;

use crate::{load_mnist, models::Mlp};

pub fn setup_lbfgs_training() -> anyhow::Result<(Mlp, VarMap)> {
    // check to see if cuda device availabke
    let dev = candle_core::Device::cuda_if_available(0)?;
    info!("Training on device {dev:?}");
    // dev.set_seed(0)?;

    let (train_images, train_labels, test_images, test_labels) = load_mnist::load_mnist()?;

    // get the labels from the dataset
    let train_labels = train_labels.to_dtype(DType::U32)?.to_device(&dev)?;
    // train_labels
    //     .to_dtype(DType::F32)?
    //     .save_safetensors("mnist_train_label", "mnist_train_label.st")?;
    // get the input from the dataset and put on device
    let train_images = train_images.to_device(&dev)?;
    // train_images.save_safetensors("mnist_train_images", "mnist_train_images.st")?;
    // get the labels from the dataset
    let test_labels = test_labels.to_dtype(DType::U32)?.to_device(&dev)?;
    // get the input from the dataset and put on device
    let test_images = test_images.to_device(&dev)?;

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
