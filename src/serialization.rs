use std::io::{Read, Write};

use anyhow::Result;
use byteorder::{LE, ReadBytesExt, WriteBytesExt};
use micrograd_rs::{
    engine::Values,
    nn::{self, FullyConnectedLayer},
};

pub trait Save {
    fn save(&self, values: &Values, writer: &mut impl Write) -> Result<()>;
}

pub trait Load {
    fn load(&self, values: &mut Values, reader: &mut impl Read) -> Result<()>;
}

impl Save for FullyConnectedLayer {
    fn save(&self, values: &Values, writer: &mut impl Write) -> Result<()> {
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
        let input_count = nn::I(reader.read_u64::<LE>()? as usize);
        let output_count = nn::O(reader.read_u64::<LE>()? as usize);

        let expected = (self.input_count, self.output_count);
        let actual = (input_count, output_count);
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

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use micrograd_rs::{engine::Operations, nn};

    use super::*;

    #[test]
    fn save_and_load_fully_connected_layer() {
        let mut ops = Operations::default();
        let input = nn::input_layer_vec((nn::B(2), nn::O(3)), &mut ops);
        let layer = FullyConnectedLayer::new(
            input.as_deref().reindex(nn::batched_output_to_input),
            nn::O(4),
            &mut ops,
            micrograd_rs::engine::Expr::relu,
        );

        let mut values = micrograd_rs::engine::Values::new(ops.len());

        // Set predictable values for weights and biases
        for (index, &node) in layer.weights().iter().enumerate() {
            values[node] = (index as f64) * 0.1;
        }
        for (index, &node) in layer.biases().iter().enumerate() {
            values[node] = (index as f64) * 0.01;
        }

        let mut serialized = Vec::new();
        Save::save(&layer, &values, &mut serialized).unwrap();

        // Reset values to NaN and load
        values.fill(f64::NAN);

        let mut cursor = Cursor::new(serialized);
        Load::load(&layer, &mut values, &mut cursor).unwrap();

        // Verify weights and biases match original values
        for (index, &node) in layer.weights().iter().enumerate() {
            assert_eq!(values[node], (index as f64) * 0.1);
        }
        for (index, &node) in layer.biases().iter().enumerate() {
            assert_eq!(values[node], (index as f64) * 0.01);
        }
    }
}
