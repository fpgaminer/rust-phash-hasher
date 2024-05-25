use anyhow::Context;
use clap::Parser;
use image::{self, imageops, io::Reader as ImageReader};
use indicatif::ParallelProgressIterator;
use nalgebra::SMatrix;
use rayon::prelude::*;
use std::{
	collections::{HashMap, HashSet},
	fs::File,
	io::{BufRead, BufReader, Cursor, Read, Seek, SeekFrom, Write},
	path::{Path, PathBuf},
	thread,
};

type Matrix32x32 = SMatrix<f32, 32, 32>;


#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
	/// Input file. Each line is expected to be a path to an image. If "-", read from stdin.
	#[arg(short, long, default_value = "-")]
	input: String,

	/// Output file.  Will be re-read on subsequent runs to avoid recomputing phashes.
	#[arg(short, long)]
	output: PathBuf,
}


fn main() {
	let args = Args::parse();

	let dct_matrix = get_dct_matrix(32);
	let dct_matrix_t = dct_matrix.transpose();

	// Read output
	let mut output_file = File::options().read(true).write(true).create(true).open(&args.output).unwrap();

	// Read output file to get the list of images that have already been processed
	let cache = read_result(&mut output_file);

	// Read the list of images from the input file
	// Skip images that are already in the cache
	let images = read_input_list(&args.input).filter(|path| !cache.contains_key(path)).collect::<HashSet<_>>();

	// Compute phashes for remaining hashes
	let (tx, rx): (std::sync::mpsc::SyncSender<(PathBuf, u64)>, _) = std::sync::mpsc::sync_channel(256);

	// This thread writes the phashes to the file
	let collector_thread = thread::spawn(move || {
		// Write phashes to the output file
		for (path, phash) in rx.iter() {
			let path = path.to_str().unwrap();

			if path.contains('\t') || path.contains('\n') {
				eprintln!("Warning: path contains tab or newline, it will be skipped: {}", path);
				continue;
			}

			writeln!(output_file, "{}\t{}", path, phash).unwrap();
			output_file.flush().unwrap();
		}
	});

	eprintln!("Computing phashes...");

	images.par_iter().progress_count(images.len() as u64).for_each_with(tx, |tx, path| {
		let phash = match image_path_to_phash(path, &dct_matrix, &dct_matrix_t) {
			Ok(phash) => phash,
			Err(err) => {
				eprintln!("Error computing phash for {}: {}", path.display(), err);
				return;
			},
		};

		tx.send((path.clone(), phash)).unwrap();
	});

	collector_thread.join().unwrap();
}


fn read_input_list(path_or_stdin: &str) -> impl Iterator<Item = PathBuf> {
	if path_or_stdin == "-" {
		let stdin = std::io::stdin();
		let reader = stdin.lock();
		Box::new(reader.lines().map_while(Result::ok).map(|line| PathBuf::from(line.trim()))) as Box<dyn Iterator<Item = PathBuf>>
	} else {
		let file = File::open(path_or_stdin).unwrap();
		let reader = BufReader::new(file);
		Box::new(reader.lines().map_while(Result::ok).map(|line| PathBuf::from(line.trim()))) as Box<dyn Iterator<Item = PathBuf>>
	}
}


/// Read the output file and return a map of paths to phashes.
/// Handles the case where the file is truncated.
/// Leaves the file pointer at the end of the file, ready for appending.
fn read_result<R: Read + Seek>(reader: &mut R) -> HashMap<PathBuf, u64> {
	let mut cache = HashMap::new();

	let mut valid_len = 0;

	reader.seek(SeekFrom::Start(0)).unwrap();

	let mut reader = BufReader::new(reader);

	loop {
		let mut line = String::new();
		match reader.read_line(&mut line) {
			Ok(0) | Err(_) => break,
			Ok(_) => (),
		}

		// If it doesn't end in a newline, it's truncated
		if !line.ends_with('\n') {
			break;
		}

		// Split by tab
		let mut parts = line.split('\t').map(|part| part.trim()).collect::<Vec<_>>();

		if parts.len() != 2 {
			break;
		}

		let phash = match parts.pop().unwrap().parse::<u64>() {
			Ok(phash) => phash,
			Err(_) => break,
		};

		let path = PathBuf::from(parts.pop().unwrap());

		cache.insert(path, phash);
		valid_len = reader.stream_position().unwrap();
	}

	let reader = reader.into_inner();

	reader.seek(SeekFrom::Start(valid_len)).unwrap();

	cache
}


/// Compute the phash for an image.
fn image_path_to_phash(path: &Path, dct_matrix: &Matrix32x32, dct_matrix_t: &Matrix32x32) -> anyhow::Result<u64> {
	let data = std::fs::read(path).context("Error reading image")?;

	let img = ImageReader::new(Cursor::new(data))
		.with_guessed_format()
		.context("Error guessing image format")?
		.decode()
		.context("Error decoding image")?;

	// Convert to a 32x32 grayscale image
	let img = imageops::grayscale(&img);
	let img = imageops::resize(&img, 32, 32, imageops::FilterType::Lanczos3);

	// Convert to a 32x32 matrix
	let img = img.into_vec();
	let img = Matrix32x32::from_row_iterator(img.iter().map(|v| *v as f32));

	// Compute DCT
	let dct_vals = dct_matrix * img * dct_matrix_t;

	// We only want the upper-left 8x8 block, ignoring the first row and column
	let dct_vals = dct_vals.fixed_view::<8, 8>(1, 1);

	// Convert to a 1D array
	let dct_vals = dct_vals.iter().collect::<Vec<_>>();

	// Calculate median
	let mut sorted = dct_vals.clone();
	sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
	let median = (sorted[31] + sorted[32]) / 2.0;

	// Convert to a bit array
	let dct_vals = dct_vals.into_iter().map(|v| if *v >= median { 1 } else { 0 }).collect::<Vec<_>>();

	// Convert to a u64
	let mut hash = 0;
	for (i, v) in dct_vals.iter().enumerate() {
		if *v == 1 {
			hash |= 1 << i;
		}
	}

	Ok(hash)
}


// Based on pHash
fn get_dct_matrix(size: usize) -> Matrix32x32 {
	let c1 = (2.0 / (size as f32)).sqrt();

	Matrix32x32::from_fn(|y, x| {
		if y == 0 {
			return 1.0 / (size as f32).sqrt();
		}
		c1 * ((std::f32::consts::PI / 2.0 / (size as f32)) * (y as f32) * ((2 * x + 1) as f32)).cos()
	})
}
