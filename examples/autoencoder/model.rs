use bhop::SimpleModel;
// use anyhow::Result;
use candle_core::{Result, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, Linear, VarBuilder};
use optimisers::Model;

// pub trait SimpleModel: Sized {
//     fn new(vs: VarBuilder, train_input: Tensor, train_output: Tensor) -> Result<Self>;
//     fn forward(&self) -> Result<Tensor>;
// }

#[derive(Clone, Debug)]
pub struct ConvNet {
    conv1: Conv2d,
    conv2: Conv2d,
    conv3: Conv2d,
    dense1: Linear,
    dense2: Linear,
    dense3: Linear,
    dense4: Linear,
    conv4: Conv2d,
    conv5: Conv2d,
    conv6: Conv2d,
    // dropout: candle_nn::Dropout,
    train_input: Tensor,
    train_output: Tensor,
    test_input: Tensor,
    test_output: Tensor,
}

// autoencoder only needs one dataset as maps data to itself
pub struct MySetupVars {
    pub train_data: Tensor,
    pub test_data: Tensor,
}

impl ConvNet {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // , train: bool
        //xs: &Tensor,
        let (b_sz, _i, _img_x, _img_y) = input.dims4()?;
        // let xs = self.dropout.forward_t(&xs, train)?;
        input
            .reshape((b_sz, 1, 28, 28))?
            .apply(&self.conv1)?
            .apply(&self.conv2)?
            .max_pool2d(2)?
            .apply(&self.conv3)?
            .max_pool2d(2)?
            .reshape((b_sz, 800))?
            .apply(&self.dense1)?
            .apply(&self.dense2)?
            .apply(&self.dense3)?
            .apply(&self.dense4)?
            .reshape((b_sz, 32, 5, 5))?
            .interpolate2d(10, 10)?
            .apply(&self.conv4)?
            .interpolate2d(24, 24)?
            .apply(&self.conv5)?
            .apply(&self.conv6)
    }
}

impl SimpleModel for ConvNet {
    type SetupVars = MySetupVars;
    fn new(vs: VarBuilder, setup: MySetupVars) -> Result<Self> {
        let conv1 = candle_nn::conv2d(1, 16, 3, Default::default(), vs.pp("conv1"))?;
        let conv2 = candle_nn::conv2d(16, 16, 3, Default::default(), vs.pp("conv2"))?;
        let conv3 = candle_nn::conv2d(16, 32, 3, Default::default(), vs.pp("conv3"))?;
        let dense1 = candle_nn::linear(800, 128, vs.pp("dense1"))?;
        let dense2 = candle_nn::linear(128, 16, vs.pp("dense2"))?;
        let dense3 = candle_nn::linear(16, 128, vs.pp("dense3"))?;
        let dense4 = candle_nn::linear(128, 800, vs.pp("dense4"))?;
        let conv4 = candle_nn::conv2d(
            32,
            16,
            3,
            Conv2dConfig {
                padding: 2,
                ..Default::default()
            },
            vs.pp("conv4"),
        )?;
        let conv5 = candle_nn::conv2d(
            16,
            16,
            3,
            Conv2dConfig {
                padding: 2,
                ..Default::default()
            },
            vs.pp("conv5"),
        )?;
        let conv6 = candle_nn::conv2d(
            16,
            1,
            3,
            Conv2dConfig {
                padding: 2,
                ..Default::default()
            },
            vs.pp("conv6"),
        )?;
        let _dropout = candle_nn::Dropout::new(0.99);
        Ok(Self {
            conv1,
            conv2,
            conv3,
            dense1,
            dense2,
            dense3,
            dense4,
            conv4,
            conv5,
            conv6,
            // dropout,
            train_input: setup.train_data.clone(),
            train_output: setup.train_data,
            test_input: setup.test_data.clone(),
            test_output: setup.test_data,
        })
    }

    fn test_eval(&self) -> candle_core::Result<f32> {
        // , train: bool
        //xs: &Tensor,
        let pixels = self.forward(&self.test_input)?;
        let n_elem = pixels.elem_count() as f64;
        ((&pixels - &self.test_output)?.sqr()? / n_elem)?
            .sum_all()?
            .to_dtype(candle_core::DType::F32)?
            .to_scalar::<f32>()
    }
}

impl Model for ConvNet {
    fn loss(&self) -> Result<Tensor> {
        let pixels = self.forward(&self.train_input)?;
        let n_elem = pixels.elem_count() as f64;
        ((&pixels - &self.train_output)?.sqr()? / n_elem)?.sum_all()
    }
}
