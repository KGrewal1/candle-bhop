use clap::Parser;

#[derive(Parser)]
pub struct Args {
    /// The learning rate
    #[arg(short = 'l', long, default_value_t = 5e-4)]
    pub lr: f64,

    /// The number of epochs
    #[arg(short = 'e', long, default_value_t = 15_000)]
    pub epochs: usize,

    /// basin hopping or adam
    #[arg(short = 'b', long, default_value_t = false)]
    pub bhop: bool,

    /// basin hopping or adam
    #[arg(short = 'd', long)]
    pub load: Option<String>,

    /// basin hopping or adam
    #[arg(short = 'r', long, default_value_t = 1e-5)]
    pub l2reg: f64,
}
