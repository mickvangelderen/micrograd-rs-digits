use crate::serialization::{Load, Save};
use anyhow::Result;
use micrograd_rs::{
    engine::{Expr, Gradients, NodeId, Operations, Values},
    iter_ext::IteratorExt as _,
    nn::{self, FullyConnectedLayer},
    view::{IndexTuple, View},
};
use std::io::{Read, Write};

#[derive(Debug, Copy, Clone)]
pub struct ModelParams {
    pub batch_size: usize,
    pub l0_size: usize,
    pub l1_size: usize,
    pub l2_size: usize,
    pub l3_size: usize,
}

pub struct Network {
    pub l0: View<Vec<NodeId>, (nn::B, nn::O)>,
    pub l1: FullyConnectedLayer,
    pub l2: FullyConnectedLayer,
    pub l3: FullyConnectedLayer,
}

pub struct TrainingModel {
    pub network: Network,
    pub y_true: View<Vec<NodeId>, (nn::B, nn::O)>,
    pub loss: NodeId,
}

pub struct InferenceModel {
    pub network: Network,
}

impl Network {
    pub fn new(params: ModelParams, ops: &mut Operations) -> Self {
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

    pub fn init_parameters(&self, values: &mut Values, rng: &mut impl rand::Rng) {
        init_layer_parameters(&self.l1, values, rng);
        init_layer_parameters(&self.l2, values, rng);
        init_layer_parameters(&self.l3, values, rng);
    }

    pub fn update_weights(&self, values: &mut Values, gradients: &Gradients) {
        update_weights(&self.l1, values, gradients);
        update_weights(&self.l2, values, gradients);
        update_weights(&self.l3, values, gradients);
    }

    pub fn inputs(&self) -> View<&[NodeId], (nn::B, nn::I)> {
        self.l0.as_deref().reindex(nn::batched_output_to_input)
    }

    pub fn outputs(&self) -> View<&[NodeId], (nn::B, nn::O)> {
        self.l3.outputs()
    }

    pub fn parameters(&self) -> impl Iterator<Item = NodeId> {
        self.l1
            .parameters()
            .chain(self.l2.parameters())
            .chain(self.l3.parameters())
    }
}

impl TrainingModel {
    pub fn new(params: ModelParams, ops: &mut Operations) -> Self {
        let network = Network::new(params, ops);
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

    pub fn init_parameters(&self, values: &mut Values, rng: &mut impl rand::Rng) {
        self.network.init_parameters(values, rng);
    }

    pub fn update_weights(&self, values: &mut Values, gradients: &Gradients) {
        self.network.update_weights(values, gradients);
    }

    pub fn inputs(&self) -> View<&[NodeId], (nn::B, nn::I)> {
        self.network.inputs()
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

impl InferenceModel {
    pub fn new(params: ModelParams, ops: &mut Operations) -> Self {
        assert_eq!(params.batch_size, 1);
        let network = Network::new(params, ops);
        Self { network }
    }

    pub fn inputs(&self) -> View<&[NodeId], (nn::B, nn::I)> {
        self.network.inputs()
    }

    pub fn predict_single(
        &self,
        input_values: &[f64],
        values: &mut Values,
        ops: &Operations,
    ) -> u8 {
        // Set input values directly using view indexing
        let inputs = self.inputs();
        for (input_index, &input_val) in input_values.iter().enumerate_with(nn::I) {
            values[inputs[(nn::B(0), input_index)]] = input_val;
        }

        // Forward pass
        ops.forward(values);

        // Find the class with highest output
        self.network
            .outputs()
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

    pub fn parameters(&self) -> impl Iterator<Item = NodeId> {
        self.network.parameters()
    }
}

fn init_layer_parameters(
    layer: &FullyConnectedLayer,
    values: &mut Values,
    rng: &mut impl rand::Rng,
) {
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

impl Save for Network {
    fn save(&self, values: &Values, writer: &mut impl Write) -> Result<()> {
        self.l1.save(values, writer)?;
        self.l2.save(values, writer)?;
        self.l3.save(values, writer)?;
        Ok(())
    }
}

impl Load for Network {
    fn load(&self, values: &mut Values, reader: &mut impl Read) -> Result<()> {
        self.l1.load(values, reader)?;
        self.l2.load(values, reader)?;
        self.l3.load(values, reader)?;
        Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;
    use micrograd_rs::engine::Operations;
    use std::io::Cursor;

    #[test]
    fn save_and_load_training_model() {
        let params = ModelParams {
            batch_size: 2,
            l0_size: 3,
            l1_size: 4,
            l2_size: 2,
            l3_size: 1,
        };

        // Write serialized training model weights into in-memory buffer.
        let mut reader = {
            let mut ops = Operations::default();
            let model = TrainingModel::new(params, &mut ops);
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
            ModelParams {
                batch_size: 1,
                ..params
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
