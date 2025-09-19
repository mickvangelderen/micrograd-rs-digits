#![allow(unused)] // to be published as lib in micrograd-rs

use std::io::{Read, Write};

use anyhow::Result;
use byteorder::{LE, ReadBytesExt, WriteBytesExt};
use micrograd_rs::{
    engine::{Expr, Gradients, NodeId, Operations, Values},
    iter_ext::IteratorExt as _,
    nn::{self, FullyConnectedLayer},
    view::{IndexTuple, View},
};

use crate::serialization::{Load, Save};

#[derive(Debug, Clone)]
pub struct FullyConnectedLayerParams {
    pub size: usize,
}

#[derive(Debug, Clone)]
pub struct ModelParams {
    pub batch_size: usize,
    pub input_size: usize,
    pub layers: Vec<FullyConnectedLayerParams>,
}

pub struct MultiLayerPerceptron {
    pub layers: Vec<FullyConnectedLayer>,
}

impl MultiLayerPerceptron {
    pub fn new(
        input_layer: View<&[NodeId], (nn::B, nn::O)>,
        layer_params: &[FullyConnectedLayerParams],
        ops: &mut Operations,
    ) -> Self {
        let layers = layer_params
            .iter()
            .fold(Vec::with_capacity(layer_params.len()), |mut layers, params| {
                let prev_output = layers
                    .last()
                    .map_or(input_layer, |layer: &FullyConnectedLayer| layer.outputs());
                let layer = FullyConnectedLayer::new(
                    prev_output.as_deref().reindex(nn::batched_output_to_input),
                    nn::O(params.size),
                    ops,
                    Expr::relu,
                );
                layers.push(layer);
                layers
            });

        Self { layers }
    }

    pub fn init_parameters(&self, values: &mut Values, rng: &mut impl rand::Rng) {
        for layer in &self.layers {
            init_layer_parameters(layer, values, rng);
        }
    }

    pub fn update_weights(&self, values: &mut Values, gradients: &Gradients) {
        for layer in &self.layers {
            update_weights(layer, values, gradients);
        }
    }

    pub fn outputs(&self) -> View<&[NodeId], (nn::B, nn::O)> {
        self.layers
            .last()
            .expect("Network must have at least one layer")
            .outputs()
    }

    pub fn parameters(&self) -> impl Iterator<Item = NodeId> {
        self.layers.iter().flat_map(|layer| {
            layer
                .weights()
                .into_iter()
                .copied()
                .chain(layer.biases().into_iter().copied())
        })
    }
}

impl Save for MultiLayerPerceptron {
    fn save(&self, values: &Values, writer: &mut impl Write) -> Result<()> {
        writer.write_u64::<LE>(self.layers.len() as u64)?;

        for layer in &self.layers {
            layer.save(values, writer)?;
        }
        Ok(())
    }
}

impl Load for MultiLayerPerceptron {
    fn load(&self, values: &mut Values, reader: &mut impl Read) -> Result<()> {
        let layer_count = reader.read_u64::<LE>()? as usize;

        if layer_count != self.layers.len() {
            anyhow::bail!(
                "Layer count mismatch: expected {} but got {}",
                self.layers.len(),
                layer_count
            );
        }

        for layer in &self.layers {
            layer.load(values, reader)?;
        }
        Ok(())
    }
}

pub struct TrainingModel {
    pub input_layer: View<Vec<NodeId>, (nn::B, nn::O)>,
    pub network: MultiLayerPerceptron,
    pub y_true: View<Vec<NodeId>, (nn::B, nn::O)>,
    pub loss: NodeId,
}

impl TrainingModel {
    pub fn new(params: &ModelParams, ops: &mut Operations) -> Self {
        let input_layer = nn::input_layer_vec((nn::B(params.batch_size), nn::O(params.input_size)), ops);
        let network = MultiLayerPerceptron::new(input_layer.as_deref(), &params.layers, ops);

        let output_size = params.layers.last().expect("Model must have at least one layer").size;
        let y_true = nn::input_layer_vec((nn::B(params.batch_size), nn::O(output_size)), ops);

        let loss = loss_mse(
            network.outputs().as_deref().reindex(nn::batched_output_to_input),
            y_true.as_deref().reindex(nn::batched_output_to_input),
            ops,
        )
        .unwrap();

        Self {
            input_layer,
            network,
            y_true,
            loss,
        }
    }

    pub fn init_parameters(&self, values: &mut Values, rng: &mut impl rand::Rng) {
        self.network.init_parameters(values, rng);
    }

    pub fn update_weights(&self, values: &mut Values, gradients: &Gradients) {
        self.network.update_weights(values, gradients);
    }

    pub fn inputs(&self) -> View<&[NodeId], (nn::B, nn::I)> {
        self.input_layer.as_deref().reindex(nn::batched_output_to_input)
    }

    pub fn targets(&self) -> View<&[NodeId], (nn::B, nn::O)> {
        self.y_true.as_deref()
    }

    pub fn loss(&self) -> NodeId {
        self.loss
    }

    pub fn parameters(&self) -> impl Iterator<Item = NodeId> {
        self.network.parameters()
    }
}

impl Save for TrainingModel {
    fn save(&self, values: &Values, writer: &mut impl Write) -> Result<()> {
        self.network.save(values, writer)
    }
}

impl Load for TrainingModel {
    fn load(&self, values: &mut Values, reader: &mut impl Read) -> Result<()> {
        self.network.load(values, reader)
    }
}

pub struct InferenceModel {
    pub input_layer: View<Vec<NodeId>, (nn::B, nn::O)>,
    pub network: MultiLayerPerceptron,
}

impl InferenceModel {
    pub fn new(params: &ModelParams, ops: &mut Operations) -> Self {
        assert_eq!(params.batch_size, 1);
        let input_layer = nn::input_layer_vec((nn::B(params.batch_size), nn::O(params.input_size)), ops);
        let network = MultiLayerPerceptron::new(input_layer.as_deref(), &params.layers, ops);
        Self { input_layer, network }
    }

    pub fn inputs(&self) -> View<&[NodeId], (nn::B, nn::I)> {
        self.input_layer.as_deref().reindex(nn::batched_output_to_input)
    }

    pub fn predict_single(&self, input_values: &[f64], values: &mut Values, ops: &Operations) -> u8 {
        let inputs = self.inputs();
        for (input_index, &input_val) in input_values.iter().enumerate_with(nn::I) {
            values[inputs[(nn::B(0), input_index)]] = input_val;
        }

        ops.forward(values);

        // Find the class with highest output
        self.network
            .outputs()
            .iter()
            .enumerate()
            .max_by(|&(_, &a), &(_, &b)| values[a].partial_cmp(&values[b]).expect("NaN in prediction"))
            .map(|(i, _)| i as u8)
            .unwrap()
    }

    pub fn parameters(&self) -> impl Iterator<Item = NodeId> {
        self.network.parameters()
    }
}

impl Save for InferenceModel {
    fn save(&self, values: &Values, writer: &mut impl Write) -> Result<()> {
        self.network.save(values, writer)
    }
}

impl Load for InferenceModel {
    fn load(&self, values: &mut Values, reader: &mut impl Read) -> Result<()> {
        self.network.load(values, reader)
    }
}

fn init_layer_parameters(layer: &FullyConnectedLayer, values: &mut Values, rng: &mut impl rand::Rng) {
    use rand::distr::Distribution;

    let dist = rand::distr::Uniform::new(-0.05, 0.05).unwrap();

    for (&weight, value) in layer.weights().iter().zip(dist.sample_iter(rng)) {
        values[weight] = value;
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

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use micrograd_rs::engine::Operations;

    use super::*;

    #[test]
    fn save_and_load_training_model() {
        let params = ModelParams {
            batch_size: 2,
            input_size: 3,
            layers: vec![
                FullyConnectedLayerParams { size: 4 },
                FullyConnectedLayerParams { size: 2 },
                FullyConnectedLayerParams { size: 1 },
            ],
        };

        // Write serialized training model weights into in-memory buffer.
        let mut reader = {
            let mut ops = Operations::default();
            let model = TrainingModel::new(&params, &mut ops);
            let ops = ops;
            let mut values = Values::new(ops.len());
            for (index, node) in model.parameters().enumerate() {
                values[node] = index as f64;
            }
            let mut serialized = Vec::new();
            Save::save(&model, &values, &mut serialized).unwrap();
            Cursor::new(serialized)
        };

        // Load serialized model weights into inference model.
        let mut ops = Operations::default();
        let model = InferenceModel::new(
            &ModelParams {
                batch_size: 1,
                input_size: params.input_size,
                layers: params.layers.clone(),
            },
            &mut ops,
        );
        let ops = ops;
        let mut values = Values::new(ops.len());
        Load::load(&model, &mut values, &mut reader).unwrap();

        for (index, node) in model.parameters().enumerate() {
            assert_eq!(values[node], index as f64);
        }
    }
}
