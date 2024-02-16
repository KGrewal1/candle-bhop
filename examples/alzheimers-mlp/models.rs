use bhop::SimpleModel;
use candle_core::{DType, Result, Tensor, D};
use candle_nn::{loss, Conv2d, Linear, VarBuilder};
use optimisers::Model;

const IMAGE_DIM: usize = 128;
const LABELS: usize = 4;

#[derive(Clone)]
pub struct Mlp {
    conv1: Conv2d,
    conv2: Conv2d,
    conv3: Conv2d,
    ln1: Linear,
    ln2: Linear,
    ln3: Linear,
    ln4: Linear,
    train_data: Tensor,
    train_labels: Tensor,
    test_data: Tensor,
    test_labels: Tensor,
}

pub struct MySetupVars {
    pub train_data: Tensor,
    pub train_labels: Tensor,
    pub test_data: Tensor,
    pub test_labels: Tensor,
}

impl SimpleModel for Mlp {
    type SetupVars = MySetupVars;
    fn new(vs: VarBuilder, setup: MySetupVars) -> Result<Self> {
        let conv1 = candle_nn::conv2d(1, 4, 3, Default::default(), vs.pp("conv1"))?;
        let conv2 = candle_nn::conv2d(4, 4, 3, Default::default(), vs.pp("conv2"))?;
        let conv3 = candle_nn::conv2d(4, 8, 3, Default::default(), vs.pp("conv3"))?;
        let ln1 = candle_nn::linear(1568, 512, vs.pp("ln1"))?;
        let ln2 = candle_nn::linear(512, 128, vs.pp("ln2"))?;
        let ln3 = candle_nn::linear(128, 16, vs.pp("ln3"))?;
        let ln4 = candle_nn::linear(16, LABELS, vs.pp("ln4"))?;
        Ok(Self {
            conv1,
            conv2,
            conv3,
            ln1,
            ln2,
            ln3,
            ln4,
            train_data: setup.train_data,
            train_labels: setup.train_labels,
            test_data: setup.test_data,
            test_labels: setup.test_labels,
        })
    }

    fn test_eval(&self) -> Result<f32> {
        let test_logits = self.forward(&self.test_data)?;
        // get the sum of the correct predictions
        let sum_ok = test_logits
            .argmax(D::Minus1)?
            .eq(&self.test_labels)?
            .to_dtype(DType::F32)?
            .sum_all()?
            .to_scalar::<f32>()?;
        #[allow(clippy::cast_precision_loss)]
        Ok(sum_ok / self.test_labels.dims1()? as f32)
    }
}

impl Model for Mlp {
    fn loss(&self) -> Result<Tensor> {
        let logits = self.forward(&self.train_data)?;
        // softmax the log probabilities
        // let log_sm = ops::log_softmax(&logits, D::Minus1)?;
        // get the loss
        loss::cross_entropy(&logits, &self.train_labels)
    }
}

impl Mlp {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let (b_sz, _i, _img_x, _img_y) = input.dims4()?;
        input
            .reshape((b_sz, 1, IMAGE_DIM, IMAGE_DIM))?
            .apply(&self.conv1)?
            .max_pool2d(2)?
            .apply(&self.conv2)?
            .max_pool2d(2)?
            .apply(&self.conv3)?
            .max_pool2d(2)?
            .reshape((b_sz, 1568))?
            .apply(&self.ln1)?
            .apply(&self.ln2)?
            .apply(&self.ln3)?
            .apply(&self.ln4)
    }
}
