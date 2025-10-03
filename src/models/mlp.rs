#![allow(unused)] // to be published as lib in micrograd-rs

use std::io::{Read, Write};

use byteorder::{LE, ReadBytesExt, WriteBytesExt};
use micrograd_rs::{
    engine::{Expr, Gradients, NodeId, Operations, Values},
    iter_ext::IteratorExt as _,
    nn::{
        self, Deserialize, FullyConnectedLayer, FullyConnectedLayerParams, MultiLayerPerceptron,
        MultiLayerPerceptronParams, Result, Serialize,
    },
    view::{Index, IndexTuple, View},
};

#[derive(Debug, Clone)]
pub struct ModelParams {
    pub batch_size: nn::B,
    pub input_size: nn::O,
    pub mlp: Vec<FullyConnectedLayerParams>,
}

pub struct TrainingModel {
    pub input_layer: View<Vec<NodeId>, (nn::B, nn::O)>,
    pub network: MultiLayerPerceptron,
    pub y_true: View<Vec<NodeId>, (nn::B, nn::O)>,
    pub loss: NodeId,
}

impl TrainingModel {
    pub fn new(params: &ModelParams, ops: &mut Operations) -> Self {
        let input_layer = nn::input_layer_vec((params.batch_size, params.input_size), ops);

        let network = MultiLayerPerceptron::new(input_layer.as_deref(), params.mlp.as_slice(), ops);

        let y_true = nn::input_layer_vec(network.outputs().shape(), ops);

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

impl Serialize for TrainingModel {
    fn serialize(&self, values: &Values, writer: &mut impl Write) -> Result<()> {
        self.network.serialize(values, writer)
    }
}

impl Deserialize for TrainingModel {
    fn deserialize(&self, values: &mut Values, reader: &mut impl Read) -> Result<()> {
        self.network.deserialize(values, reader)
    }
}

pub struct InferenceModel {
    pub input_layer: View<Vec<NodeId>, (nn::B, nn::O)>,
    pub network: MultiLayerPerceptron,
}

impl InferenceModel {
    pub fn new(params: &ModelParams, ops: &mut Operations) -> Self {
        let input_layer = nn::input_layer_vec((params.batch_size, params.input_size), ops);
        let network = MultiLayerPerceptron::new(input_layer.as_deref(), params.mlp.as_slice(), ops);
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

impl Serialize for InferenceModel {
    fn serialize(&self, values: &Values, writer: &mut impl Write) -> Result<()> {
        self.network.serialize(values, writer)
    }
}

impl Deserialize for InferenceModel {
    fn deserialize(&self, values: &mut Values, reader: &mut impl Read) -> Result<()> {
        self.network.deserialize(values, reader)
    }
}

fn loss_mse(
    y_pred: View<&[NodeId], (nn::B, nn::I)>,
    y: View<&[NodeId], (nn::B, nn::I)>,
    ops: &mut Operations,
) -> Option<NodeId> {
    assert_eq!(y_pred.shape(), y.shape());
    // We can sum over the batch and over the output dimensions.
    let mut terms = y_pred.shape().indices().map(|i| (y_pred[i] - y[i]).pow_2());
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
            batch_size: nn::B(2),
            input_size: nn::O(3),
            mlp: vec![
                FullyConnectedLayerParams { output_size: nn::O(4) },
                FullyConnectedLayerParams { output_size: nn::O(2) },
                FullyConnectedLayerParams { output_size: nn::O(1) },
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
            Serialize::serialize(&model, &values, &mut serialized).unwrap();
            Cursor::new(serialized)
        };

        // Load serialized model weights into inference model.
        let mut ops = Operations::default();
        let model = InferenceModel::new(
            &ModelParams {
                batch_size: nn::B(1),
                input_size: params.input_size,
                mlp: params.mlp.clone(),
            },
            &mut ops,
        );
        let ops = ops;
        let mut values = Values::new(ops.len());
        Deserialize::deserialize(&model, &mut values, &mut reader).unwrap();

        for (index, node) in model.parameters().enumerate() {
            assert_eq!(values[node], index as f64);
        }
    }
}
