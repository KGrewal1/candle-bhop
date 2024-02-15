use bhop::SimpleModel;
use candle_core::{DType, Result, Tensor, D};
use candle_nn::{loss, Linear, Module, VarBuilder};
use optimisers::Model;

const IMAGE_DIM: usize = 784;
const LABELS: usize = 10;

#[derive(Clone)]
pub struct Mlp {
    ln1: Linear,
    ln2: Linear,
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
        let ln1 = candle_nn::linear(IMAGE_DIM, 10, vs.pp("ln1"))?;
        let ln2 = candle_nn::linear(10, LABELS, vs.pp("ln2"))?;
        Ok(Self {
            ln1,
            ln2,
            train_data: setup.train_data,
            train_labels: setup.train_labels,
            test_data: setup.test_data,
            test_labels: setup.test_labels,
        })
    }

    // fn forward(&self) -> Result<Tensor> {
    //     let xs = self.ln1.forward(&self.train_data)?;
    //     let xs = xs.tanh()?;
    //     self.ln2.forward(&xs)
    // }
    fn test_eval(&self) -> Result<f32> {
        let xs = self.ln1.forward(&self.test_data)?;
        let xs = xs.tanh()?;
        let test_logits = self.ln2.forward(&xs)?;
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
        let logits = self.forward()?;
        // softmax the log probabilities
        // let log_sm = ops::log_softmax(&logits, D::Minus1)?;
        // get the loss
        loss::cross_entropy(&logits, &self.train_labels)
    }
}

pub trait Forward {
    fn forward(&self) -> Result<Tensor>;
}
impl Forward for Mlp {
    fn forward(&self) -> Result<Tensor> {
        let xs = self.ln1.forward(&self.train_data)?;
        let xs = xs.tanh()?;
        self.ln2.forward(&xs)
    }
}
