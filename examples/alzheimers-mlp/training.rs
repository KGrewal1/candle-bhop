use bhop::SimpleModel;
use candle_core::DType;
use candle_nn::{VarBuilder, VarMap};
use log::info;
const NIMAGES: usize = 2048;
use crate::{load_dataset, models::Mlp, DATATYPE};

pub fn setup_training(load_path: Option<String>) -> anyhow::Result<(Mlp, VarMap)> {
    // check to see if cuda device availabke
    let dev = candle_core::Device::cuda_if_available(0)?;
    info!("Training on device {dev:?}");
    // dev.set_seed(0)?;

    let (train_images, train_labels, test_images, test_labels) = load_dataset::load_mnist()?;

    // let iter = DatasetRandomIter::new(ds, valid, seq_len, device)
    // let batches = Batcher::new2((train_images, train_labels));

    // get the labels from the dataset
    let train_labels = train_labels
        .narrow(0, 0, NIMAGES)?
        .to_dtype(DType::U32)?
        .to_device(&dev)?;

    let train_images = train_images.narrow(0, 0, NIMAGES)?.to_device(&dev)?;

    let test_labels = test_labels.to_dtype(DType::U32)?.to_device(&dev)?;
    let test_images = test_images.to_device(&dev)?;

    // create a new variable store
    let mut varmap = VarMap::new();
    // create a new variable builder
    let vs = VarBuilder::from_varmap(&varmap, DATATYPE, &dev);

    let setup = crate::models::MySetupVars {
        train_data: train_images,
        train_labels,
        test_data: test_images,
        test_labels,
    };
    // create model from variables
    let model = Mlp::new(vs.clone(), setup)?;

    if let Some(path) = load_path {
        varmap.load(path)?;
    }
    Ok((model, varmap))
}
