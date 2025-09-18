use std::{fs, io};

use anyhow::Result;

const DIGITS_URL: &str = "https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/sklearn/datasets/data/digits.csv.gz";

pub type DigitsDataset = Vec<([u8; 64], u8)>;

fn load_digits_file() -> Result<fs::File> {
    let cache_path = {
        let temp_dir = std::env::temp_dir();
        [
            temp_dir.as_os_str(),
            "micrograd-rs-digits".as_ref(),
            "digits.csv.gz".as_ref(),
        ]
        .into_iter()
        .collect::<std::path::PathBuf>()
    };

    if let Some(parent) = cache_path.parent() {
        fs::create_dir_all(parent)?;
    }

    match fs::OpenOptions::new().read(true).open(&cache_path) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            // Need to create.
        }
        other => {
            return other.map_err(Into::into);
        }
    }

    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&cache_path)?;

    // FIXME: Should use tempfile and atomic rename and file lock instead, now readers can open a partially written file.
    let response = ureq::get(DIGITS_URL).call()?;
    io::copy(&mut response.into_reader(), &mut file)?;

    Ok(fs::OpenOptions::new().read(true).open(&cache_path)?)
}

pub fn load_digits() -> Result<DigitsDataset> {
    let file = load_digits_file()?;
    let mut csv = csv::Reader::from_reader(flate2::read::GzDecoder::new(file));

    csv.records()
        .map(|result| -> Result<([u8; 64], u8), anyhow::Error> {
            let record = result?;

            let pixels: [u8; 64] = (0..64)
                .map(|i| -> Result<u8, anyhow::Error> {
                    let val: f64 = record
                        .get(i)
                        .ok_or_else(|| anyhow::anyhow!("Missing pixel at index {}", i))?
                        .parse()?;
                    Ok(val as u8)
                })
                .collect::<Result<Vec<_>, _>>()?
                .try_into()
                .map_err(|_| anyhow::anyhow!("Expected 64 pixels"))?;

            let label: u8 = record
                .get(64)
                .ok_or_else(|| anyhow::anyhow!("Missing label column"))?
                .parse()?;

            Ok((pixels, label))
        })
        .collect()
}
