mod dataset;
mod serialization;

use dataset::load_digits;
use image::{ImageBuffer, Luma};
use micrograd_rs::{
    engine::{Expr, Gradients, NodeId, Operations, Values},
    iter_ext::IteratorExt as _,
    nn::{self, FullyConnectedLayer},
    view::{IndexTuple, View},
};
use std::fs;

#[derive(Clone, Debug)]
struct ModelParams {
    batch_size: usize,
    l0_size: usize,
    l1_size: usize,
    l2_size: usize,
    l3_size: usize,
}

struct Network {
    l0: View<Vec<NodeId>, (nn::B, nn::O)>,
    l1: FullyConnectedLayer,
    l2: FullyConnectedLayer,
    l3: FullyConnectedLayer,
}

struct TrainingModel {
    network: Network,
    y_true: View<Vec<NodeId>, (nn::B, nn::O)>,
    loss: NodeId,
}

struct InferenceModel {
    network: Network,
}

impl Network {
    fn new(params: ModelParams, ops: &mut Operations) -> Self {
        let l0 = nn::input_layer_vec((nn::B(params.batch_size), nn::O(params.l0_size)), ops);
        let l1 = FullyConnectedLayer::new(
            l0.as_deref().reindex(nn::batched_output_to_input),
            nn::O(params.l1_size),
            ops,
            Expr::relu,
        );
        let l2 = FullyConnectedLayer::new(
            l1.outputs().as_deref().reindex(nn::batched_output_to_input),
            nn::O(params.l2_size),
            ops,
            Expr::relu,
        );
        let l3 = FullyConnectedLayer::new(
            l2.outputs().as_deref().reindex(nn::batched_output_to_input),
            nn::O(params.l3_size),
            ops,
            Expr::relu,
        );

        Self { l0, l1, l2, l3 }
    }

    fn init_parameters(&self, values: &mut Values) {
        init_layer_parameters(&self.l1, values);
        init_layer_parameters(&self.l2, values);
        init_layer_parameters(&self.l3, values);
    }

    fn update_weights(&self, values: &mut Values, gradients: &Gradients) {
        update_weights(&self.l1, values, gradients);
        update_weights(&self.l2, values, gradients);
        update_weights(&self.l3, values, gradients);
    }

    fn set_input(&self, batch_index: nn::B, pixels: &[u8], values: &mut Values) {
        for (output_index, &pixel) in pixels.iter().enumerate_with(nn::O) {
            values[self.l0[(batch_index, output_index)]] =
                pixel as f64 / (dataset::PIXEL_MAX as f64);
        }
    }

    fn predictions(&self) -> View<&[NodeId], (nn::B, nn::O)> {
        self.l3.outputs()
    }

    fn parameters(&self) -> impl Iterator<Item = NodeId> {
        self.l1
            .parameters()
            .chain(self.l2.parameters())
            .chain(self.l3.parameters())
    }
}

impl TrainingModel {
    fn new(params: ModelParams, ops: &mut Operations) -> Self {
        let network = Network::new(params.clone(), ops);
        let y_true = nn::input_layer_vec((nn::B(params.batch_size), nn::O(params.l3_size)), ops);
        let loss = loss_mse(
            network
                .l3
                .outputs()
                .as_deref()
                .reindex(nn::batched_output_to_input),
            y_true.as_deref().reindex(nn::batched_output_to_input),
            ops,
        )
        .unwrap();

        Self {
            network,
            y_true,
            loss,
        }
    }

    fn parameters(&self) -> impl Iterator<Item = NodeId> {
        self.network
            .l1
            .parameters()
            .chain(self.network.l2.parameters())
            .chain(self.network.l3.parameters())
    }

    fn init_parameters(&self, values: &mut Values) {
        self.network.init_parameters(values);
    }

    fn update_weights(&self, values: &mut Values, gradients: &Gradients) {
        self.network.update_weights(values, gradients);
    }

    fn set_input(&self, batch_index: nn::B, pixels: &[u8], values: &mut Values) {
        self.network.set_input(batch_index, pixels, values);
    }

    fn set_target(&self, batch_index: nn::B, label: u8, values: &mut Values) {
        for (output_index, label_index) in (0..=dataset::LABEL_MAX).enumerate_with(nn::O) {
            values[self.y_true[(batch_index, output_index)]] =
                if label_index == label { 1.0 } else { 0.0 };
        }
    }

    fn loss(&self) -> NodeId {
        self.loss
    }
}

impl InferenceModel {
    fn new(params: ModelParams, ops: &mut Operations) -> Self {
        let network = Network::new(params.clone(), ops);
        Self { network }
    }

    fn set_input(&self, batch_index: nn::B, pixels: &[u8], values: &mut Values) {
        self.network.set_input(batch_index, pixels, values);
    }

    fn predict_single(&self, pixels: &[u8], values: &mut Values, ops: &Operations) -> u8 {
        // Set input for single sample (batch index 0)
        self.set_input(nn::B(0), pixels, values);

        // Forward pass
        ops.forward(values);

        // Find the class with highest output
        self.network
            .predictions()
            .iter()
            .enumerate()
            .max_by(|&(_, &a), &(_, &b)| {
                values[a]
                    .partial_cmp(&values[b])
                    .expect("NaN in prediction")
            })
            .map(|(i, _)| i as u8)
            .unwrap()
    }
}

fn init_layer_parameters(layer: &FullyConnectedLayer, values: &mut Values) {
    for &weight in layer.weights().iter() {
        values[weight] = (rand::random::<f64>() - 0.5) * 0.1;
    }
    for &bias in layer.biases().iter() {
        values[bias] = 0.0;
    }
}

fn update_weights(layer: &FullyConnectedLayer, values: &mut Values, gradients: &Gradients) {
    for &node in layer.weights().iter().chain(layer.biases().iter()) {
        values[node] -= gradients[node];
    }
}

fn loss_mse(
    y_pred: View<&[NodeId], (nn::B, nn::I)>,
    y: View<&[NodeId], (nn::B, nn::I)>,
    ops: &mut Operations,
) -> Option<NodeId> {
    assert_eq!(y_pred.len(), y.len());
    // We can sum over the batch and over the output dimensions.
    let mut terms = y_pred.len().indices().map(|i| (y_pred[i] - y[i]).pow_2());
    let first = terms.next().map(|term| ops.insert(term))?;
    Some(terms.fold(first, |sum, term| ops.insert(sum + term)))
}

fn main() -> anyhow::Result<()> {
    println!("Loading digits dataset...");
    let dataset = load_digits()?;
    println!("Loaded {} samples", dataset.len());

    // Split dataset into train and test sets (80/20 split)
    let split_idx = (dataset.len() * 4) / 5;
    let (train_data, test_data) = dataset.split_at(split_idx);
    println!(
        "Train samples: {}, Test samples: {}",
        train_data.len(),
        test_data.len()
    );

    let model_params = ModelParams {
        batch_size: 16,
        l0_size: 64, // pixels in each image
        l1_size: 64,
        l2_size: 64,
        l3_size: dataset::LABEL_MAX as usize + 1,
    };

    // Construct computation graph.
    let mut ops = Operations::default();
    let model = TrainingModel::new(model_params.clone(), &mut ops);
    let ops = ops;

    // Create buffers.
    let mut values = Values::new(ops.len());
    let mut gradients = Gradients::new(ops.len());

    // Initialize parameters
    model.init_parameters(&mut values);

    const LR: f64 = 0.005;

    let mut indices: Vec<usize> = (0..train_data.len()).collect();
    for epoch in 0..10 {
        // Create and shuffle index vector for this epoch
        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();
        indices.shuffle(&mut rng);

        for (step, sample_indices) in indices.chunks_exact(model_params.batch_size).enumerate() {
            for (batch_index, sample_index) in sample_indices.iter().copied().enumerate_with(nn::B)
            {
                // Set input pixels and target
                let &(ref pixels, label) = &train_data[sample_index];
                model.set_input(batch_index, pixels, &mut values);
                model.set_target(batch_index, label, &mut values);
            }

            ops.forward(&mut values);
            let batch_loss = values[model.loss()];
            ops.backward(&values, &mut gradients, model.loss(), LR);

            println!("epoch {epoch}, step {step}, loss = {batch_loss}");

            // Update parameters
            model.update_weights(&mut values, &gradients);
        }
    }

    // Compute predictions on test set
    println!("\nComputing predictions on test set...");
    let mut inference_ops = Operations::default();
    let inference_model = InferenceModel::new(model_params, &mut inference_ops);
    let inference_ops = inference_ops;

    let mut inference_values = Values::new(inference_ops.len());

    // Copy trained weights directly
    for (training_param, inference_param) in
        model.parameters().zip(inference_model.network.parameters())
    {
        inference_values[inference_param] = values[training_param];
    }

    let predictions = test_data
        .iter()
        .map(|&(ref pixels, _)| {
            inference_model.predict_single(pixels, &mut inference_values, &inference_ops)
        })
        .collect::<Vec<_>>();

    // Calculate accuracy
    let correct = predictions
        .iter()
        .copied()
        .zip(test_data.iter())
        .filter(|&(pred, &(_, actual))| pred == actual)
        .count();
    let accuracy = correct as f64 / test_data.len() as f64;
    println!(
        "Test accuracy: {:.2}% ({}/{} correct)",
        accuracy * 100.0,
        correct,
        test_data.len()
    );

    // Save test images with predictions
    save_digit_images(
        test_data,
        Some(&predictions),
        "test_predictions",
        test_data.len(),
    )?;

    Ok(())
}

#[allow(unused)]
fn save_digit_images(
    dataset: &[([u8; 64], u8)],
    predictions: Option<&[u8]>,
    output_dir: &str,
    max_images: usize,
) -> anyhow::Result<()> {
    fs::create_dir_all(output_dir)?;

    for (i, (pixels, actual_label)) in dataset.iter().enumerate().take(max_images) {
        // Create 8x8 grayscale image from 64 pixels
        let img = ImageBuffer::from_fn(8, 8, |x, y| {
            let pixel_index = (y * 8 + x) as usize;
            let pixel_value = (pixels[pixel_index] as f32 / PIXEL_MAX as f32 * 255.0) as u8;
            Luma([pixel_value])
        });

        let filename = if let Some(preds) = predictions {
            let predicted_label = preds[i];
            let status = if predicted_label == *actual_label {
                "correct"
            } else {
                "wrong"
            };
            format!(
                "{}/digit_{:04}_{}_pred{}_actual{}.png",
                output_dir, i, status, predicted_label, actual_label
            )
        } else {
            format!("{}/digit_{:04}_actual{}.png", output_dir, i, actual_label)
        };

        img.save(&filename)?;
    }

    println!(
        "Saved {} digit images to {}/",
        dataset.len().min(max_images),
        output_dir
    );
    Ok(())
}
