use crate::{InferenceModel, Network, TrainingModel};
use anyhow::Result;
use byteorder::{LE, ReadBytesExt, WriteBytesExt};
use micrograd_rs::{
    engine::Values,
    nn::{self, FullyConnectedLayer},
};
use std::io::{Read, Write};

pub trait Save {
    fn save(&self, values: &Values, writer: &mut impl Write) -> Result<()>;
}

pub trait Load {
    fn load(&self, values: &mut Values, reader: &mut impl Read) -> Result<()>;
}

impl Save for FullyConnectedLayer {
    fn save(&self, values: &Values, writer: &mut impl Write) -> Result<()> {
        writer.write_u64::<LE>(usize::from(self.batch_count) as u64)?;
        writer.write_u64::<LE>(usize::from(self.input_count) as u64)?;
        writer.write_u64::<LE>(usize::from(self.output_count) as u64)?;

        for &node in self.weights().iter() {
            writer.write_f64::<LE>(values[node])?;
        }
        for &node in self.biases().iter() {
            writer.write_f64::<LE>(values[node])?;
        }
        Ok(())
    }
}

impl Load for FullyConnectedLayer {
    fn load(&self, values: &mut Values, reader: &mut impl Read) -> Result<()> {
        let batch_count = nn::B(reader.read_u64::<LE>()? as usize);
        let input_count = nn::I(reader.read_u64::<LE>()? as usize);
        let output_count = nn::O(reader.read_u64::<LE>()? as usize);

        let expected = (self.batch_count, self.input_count, self.output_count);
        let actual = (batch_count, input_count, output_count);
        if expected != actual {
            anyhow::bail!("Layer shape mismatch: expected {expected:?} but got {actual:?}",);
        }

        for &node in self.weights().iter() {
            values[node] = reader.read_f64::<LE>()?;
        }
        for &node in self.biases().iter() {
            values[node] = reader.read_f64::<LE>()?;
        }
        Ok(())
    }
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
    use crate::{ModelParams, TrainingModel};
    use micrograd_rs::engine::Operations;
    use std::io::Cursor;

    #[test]
    fn save_and_load() {
        // Create a test model using the actual model structs
        let params = ModelParams {
            batch_size: 2,
            l0_size: 3,
            l1_size: 4,
            l2_size: 2,
            l3_size: 1,
        };

        let mut ops = Operations::default();
        let model = TrainingModel::new(params, &mut ops);
        let ops = ops;

        let mut values = Values::new(ops.len());
        for (index, node) in model.parameters().enumerate() {
            values[node] = index as f64;
        }

        let mut serialized = Vec::new();

        Save::save(&model, &values, &mut serialized).unwrap();

        // Reset values to NaN and load
        values.fill(f64::NAN);

        let mut cursor = Cursor::new(serialized);
        Load::load(&model, &mut values, &mut cursor).unwrap();

        // Verify all weights and biases match the original predictable values
        for (index, node) in model.parameters().enumerate() {
            assert_eq!(values[node], index as f64);
        }
    }
}
