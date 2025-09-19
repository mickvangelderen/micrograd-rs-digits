use std::{fs, io};

use anyhow::Result;

const DIGITS_URL: &str = "https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/sklearn/datasets/data/digits.csv.gz";
pub const PIXEL_MAX: u8 = 16;
pub const LABEL_MAX: u8 = 9;

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

    let dataset = csv
        .records()
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
        .collect::<Result<Vec<_>>>()?;

    for &(ref pixels, label) in &dataset {
        assert!(pixels.iter().copied().all(|v| v <= PIXEL_MAX));
        assert!(label <= LABEL_MAX);
    }

    Ok(dataset)
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
