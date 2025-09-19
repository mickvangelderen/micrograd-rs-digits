mod dataset;

use dataset::load_digits;
use image::{ImageBuffer, Luma};
use micrograd_rs::{
    engine::{Expr, Gradients, NodeId, Operations, Values},
    iter_ext::IteratorExt as _,
    nn::{self, FullyConnectedLayer},
    view::{IndexTuple, View},
};
use std::fs;

const PIXEL_MAX: u8 = 16;
const LABEL_MAX: u8 = 9;

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

    for &(ref pixels, label) in &dataset {
        assert!(pixels.iter().copied().all(|v| v <= PIXEL_MAX));
        assert!(label <= LABEL_MAX);
    }

    // Split dataset into train and test sets (80/20 split)
    let split_idx = (dataset.len() * 4) / 5;
    let (train_data, test_data) = dataset.split_at(split_idx);
    println!(
        "Train samples: {}, Test samples: {}",
        train_data.len(),
        test_data.len()
    );

    let batch_size = 16;
    let l0_size = 64; // pixels in each image
    let l1_size = 64;
    let l2_size = 64;
    let l3_size = LABEL_MAX as usize + 1;

    // Construct computation graph.
    let mut ops = Operations::default();

    let l0 = nn::input_layer_vec((nn::B(batch_size), nn::O(l0_size)), &mut ops);
    let l1 = FullyConnectedLayer::new(
        l0.as_deref().reindex(nn::batched_output_to_input),
        nn::O(l1_size),
        &mut ops,
        Expr::relu,
    );
    let l2 = FullyConnectedLayer::new(
        l1.outputs().as_deref().reindex(nn::batched_output_to_input),
        nn::O(l2_size),
        &mut ops,
        Expr::relu,
    );
    let l3 = FullyConnectedLayer::new(
        l2.outputs().as_deref().reindex(nn::batched_output_to_input),
        nn::O(l3_size),
        &mut ops,
        Expr::relu,
    );
    let y_true = nn::input_layer_vec((nn::B(batch_size), nn::O(l3_size)), &mut ops); // One-hot encoded target
    let loss = loss_mse(
        l3.outputs().as_deref().reindex(nn::batched_output_to_input),
        y_true.as_deref().reindex(nn::batched_output_to_input),
        &mut ops,
    )
    .unwrap();
    let ops = ops;

    // Create buffers.
    let mut values = Values::new(ops.len());
    let mut gradients = Gradients::new(ops.len());

    // Initialize parameters
    init_layer_parameters(&l1, &mut values);
    init_layer_parameters(&l2, &mut values);
    init_layer_parameters(&l3, &mut values);

    const LR: f64 = 0.005;

    let mut indices: Vec<usize> = (0..train_data.len()).collect();
    for epoch in 0..10 {
        // Create and shuffle index vector for this epoch
        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();
        indices.shuffle(&mut rng);

        for (step, sample_indices) in indices.chunks_exact(batch_size).enumerate() {
            for (batch_index, sample_index) in sample_indices.iter().copied().enumerate_with(nn::B)
            {
                // Set input pixels
                let &(ref pixels, label) = &train_data[sample_index];
                for (output_index, &pixel) in pixels.iter().enumerate_with(nn::O) {
                    values[l0[(batch_index, output_index)]] = pixel as f64 / (PIXEL_MAX as f64);
                }

                // Set one-hot encoded target
                for (output_index, label_index) in (0..=LABEL_MAX).enumerate_with(nn::O) {
                    values[y_true[(batch_index, output_index)]] =
                        if label_index == label { 1.0 } else { 0.0 };
                }
            }

            ops.forward(&mut values);
            let batch_loss = values[loss];
            ops.backward(&values, &mut gradients, loss, LR);

            println!("epoch {epoch}, step {step}, loss = {batch_loss}");

            // Update parameters
            update_weights(&l1, &mut values, &gradients);
            update_weights(&l2, &mut values, &gradients);
            update_weights(&l3, &mut values, &gradients);
        }
    }

    // // Compute predictions on test set
    // println!("\nComputing predictions on test set...");
    // let mut predictions = Vec::new();

    // for (pixels, _) in test_data {
    //     // Set input pixels
    //     for (i, &pixel) in pixels.iter().enumerate() {
    //         values[l0[i]] = pixel as f64 / (PIXEL_MAX as f64);
    //     }

    //     ops.forward(&mut values);

    //     // Find the class with highest output
    //     let predicted_label = l3
    //         .outputs()
    //         .iter()
    //         .enumerate()
    //         .max_by_key(|&(_, &node_id)| (values[node_id] * 1000000.0) as i64)
    //         .map(|(i, _)| i as u8)
    //         .unwrap();

    //     predictions.push(predicted_label);
    // }

    // // Calculate accuracy
    // let correct = predictions
    //     .iter()
    //     .copied()
    //     .zip(test_data.iter())
    //     .filter(|&(pred, &(_, actual))| pred == actual)
    //     .count();
    // let accuracy = correct as f64 / test_data.len() as f64;
    // println!(
    //     "Test accuracy: {:.2}% ({}/{} correct)",
    //     accuracy * 100.0,
    //     correct,
    //     test_data.len()
    // );

    // // Save test images with predictions
    // save_digit_images(
    //     test_data,
    //     Some(&predictions),
    //     "test_predictions",
    //     test_data.len(),
    // )?;

    Ok(())
}

#[allow(unused)]
fn compute_and_print_histograms(dataset: &Vec<([u8; 64], u8)>) {
    // Compute pixel histogram
    let mut pixel_histogram = [0u32; 256];
    for (pixels, _) in dataset {
        for &pixel in pixels {
            pixel_histogram[pixel as usize] += 1;
        }
    }
    print_histogram(&pixel_histogram, "Pixel value");

    // Compute label histogram
    let mut label_histogram = [0u32; 10];
    for (_, label) in dataset {
        label_histogram[*label as usize] += 1;
    }
    print_histogram(&label_histogram, "Label");
}

fn print_histogram<const N: usize>(histogram: &[u32; N], name: &str) {
    println!("\n{} histogram:", name);
    for (value, &count) in histogram.iter().enumerate() {
        if count > 0 {
            println!("  {}: {}", value, count);
        }
    }
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
