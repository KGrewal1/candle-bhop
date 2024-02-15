use bhop::SimpleModel;
use candle_nn::{VarBuilder, VarMap};
use log::info;

use crate::{
    load_mnist,
    model::{ConvNet, MySetupVars},
    DATATYPE,
};

pub fn setup_training(// m: &candle_datasets::vision::Dataset,
) -> anyhow::Result<(ConvNet, VarMap)> {
    // check to see if cuda device availabke
    let dev = candle_core::Device::cuda_if_available(0)?;
    info!("Training on device {dev:?}");
    // dev.set_seed(0)?;

    let (train_images, _train_labels, test_images, _test_labels) = load_mnist::load_mnist()?;

    let train_images = train_images.narrow(0, 40000, 4096)?.to_device(&dev)?;

    let test_images = test_images.narrow(0, 0, 4096)?.to_device(&dev)?;
    // get the labels from the dataset
    // training loops
    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DATATYPE, &dev);
    let setup = MySetupVars {
        train_data: train_images,
        test_data: test_images,
    };
    let convnet = ConvNet::new(vs.clone(), setup)?;
    Ok((convnet, varmap))
}
