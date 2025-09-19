mod dataset;
mod models;
mod serialization;

use clap::{Parser, Subcommand};
use dataset::load_digits;
use image::{ImageBuffer, Luma};
use micrograd_rs::{
    engine::{Gradients, Operations, Values},
    iter_ext::IteratorExt as _,
    nn,
};
use models::mlp::{InferenceModel, ModelParams, TrainingModel};

#[derive(Debug, Copy, Clone)]
struct TrainingParams {
    epochs: usize,
    learning_rate: f64,
}
use serialization::{Load, Save};
use std::fs;

#[derive(Parser)]
#[command(name = "micrograd-rs-digits")]
#[command(about = "Train and test neural networks on digit recognition")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Train {
        #[arg(default_value = "out/weights.bin")]
        weights_path: String,
        #[arg(long, default_value = "10")]
        epochs: usize,
        #[arg(long, default_value = "0.01")]
        learning_rate: f64,
    },
    Test {
        #[arg(default_value = "out/weights.bin")]
        weights_path: String,
    },
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    println!("Loading digits dataset...");
    let dataset = load_digits()?;
    println!("Loaded {} samples", dataset.len());

    let split_idx = (dataset.len() * 4) / 5;
    let (train_data, test_data) = dataset.split_at(split_idx);

    match cli.command {
        Commands::Train {
            weights_path,
            epochs,
            learning_rate,
        } => {
            let training_params = TrainingParams {
                epochs,
                learning_rate,
            };
            train(weights_path, train_data, training_params)
        }
        Commands::Test { weights_path } => test(weights_path, test_data),
    }
}

fn train(
    weights_path: String,
    train_data: &[([u8; 64], u8)],
    training_params: TrainingParams,
) -> anyhow::Result<()> {
    println!("Train samples: {}", train_data.len());
    println!(
        "Training for {} epochs with learning rate {}",
        training_params.epochs, training_params.learning_rate
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
    let model = TrainingModel::new(model_params, &mut ops);
    let ops = ops;

    // Create buffers.
    let mut values = Values::new(ops.len());
    let mut gradients = Gradients::new(ops.len());

    // Initialize parameters
    model.init_parameters(&mut values);

    let mut indices: Vec<usize> = (0..train_data.len()).collect();
    for epoch in 0..training_params.epochs {
        // Create and shuffle index vector for this epoch
        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();
        indices.shuffle(&mut rng);

        let total_steps = indices.chunks_exact(model_params.batch_size).len();
        for (step, sample_indices) in indices.chunks_exact(model_params.batch_size).enumerate() {
            let inputs = model.inputs();
            let targets = model.targets();

            for (batch_index, sample_index) in sample_indices.iter().copied().enumerate_with(nn::B)
            {
                // Set input pixels and target
                let &(ref pixels, label) = &train_data[sample_index];

                // Set normalized pixels directly using view indexing
                for (input_index, &pixel) in pixels.iter().enumerate_with(nn::I) {
                    values[inputs[(batch_index, input_index)]] =
                        pixel as f64 / dataset::PIXEL_MAX as f64;
                }

                // Set one-hot encoded target directly using view indexing
                for (output_index, target_index) in (0..=dataset::LABEL_MAX).enumerate_with(nn::O) {
                    values[targets[(batch_index, output_index)]] =
                        if target_index == label { 1.0 } else { 0.0 };
                }
            }

            ops.forward(&mut values);
            let batch_loss = values[model.loss()];
            ops.backward(
                &values,
                &mut gradients,
                model.loss(),
                training_params.learning_rate,
            );

            println!(
                "epoch {:3}/{:3}, step {:4}/{:4}, loss = {batch_loss}",
                epoch + 1,
                training_params.epochs,
                step + 1,
                total_steps
            );

            // Update parameters
            model.update_weights(&mut values, &gradients);
        }
    }

    // Save trained weights
    println!("Saving weights to {}", weights_path);
    {
        if let Some(parent) = std::path::Path::new(&weights_path).parent() {
            fs::create_dir_all(parent)?;
        }
        let file = std::fs::File::create(&weights_path)?;
        let mut writer = std::io::BufWriter::new(file);
        model.save(&values, &mut writer)?;
    }
    println!("Training completed!");

    Ok(())
}

fn test(weights_path: String, test_data: &[([u8; 64], u8)]) -> anyhow::Result<()> {
    println!("Test samples: {}", test_data.len());

    // May only differ from training in fields that do not affect model parameters, such as the batch size.
    let model_params = ModelParams {
        batch_size: 1,
        l0_size: 64, // pixels in each image
        l1_size: 64,
        l2_size: 64,
        l3_size: dataset::LABEL_MAX as usize + 1,
    };

    // Construct computation graph for inference
    let mut inference_ops = Operations::default();
    let inference_model = InferenceModel::new(model_params, &mut inference_ops);
    let inference_ops = inference_ops;

    let mut inference_values = Values::new(inference_ops.len());

    // Load trained weights
    println!("Loading weights from {}", weights_path);
    {
        let mut reader = std::io::BufReader::new(std::fs::File::open(&weights_path)?);
        inference_model
            .network
            .load(&mut inference_values, &mut reader)?;
    }

    let predictions = test_data
        .iter()
        .map(|(pixels, _)| {
            // Normalize pixels to [0, 1] range
            let normalized_pixels: Vec<f64> = pixels
                .iter()
                .map(|&pixel| pixel as f64 / dataset::PIXEL_MAX as f64)
                .collect();
            inference_model.predict_single(
                &normalized_pixels,
                &mut inference_values,
                &inference_ops,
            )
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

    save_digit_images(
        test_data,
        Some(&predictions),
        "out/predictions",
        test_data.len(),
    )?;

    Ok(())
}

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
            let pixel_value =
                (pixels[pixel_index] as f32 / dataset::PIXEL_MAX as f32 * 255.0) as u8;
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
