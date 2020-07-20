use num::complex::Complex;
use num::Zero;

const PI: f64 = std::f64::consts::PI;

struct SignalDescription {
    // Frequency in Hz of the signal
    frequency: usize,
    amplitude: f64,

    // Offset between in interval [0; 2pi)
    offset: f64
}

/**
 * Sampel datapoints in `interval` seconds doing `samples_rate` Hz sampling.
 * Generate the signal based on `signal_description`
 */
fn sample(interval: f64, sample_rate: usize, signal_description: &[SignalDescription]) -> Vec<f64> {
    let measurement_count = (interval * (sample_rate as f64)) as usize;
    assert!(is_power_of_two(measurement_count), "Measurement count should be a power of two!");

    let mut measurements = Vec::with_capacity(measurement_count);
    for i in 0 .. measurement_count {
        let t = (i as f64) / (sample_rate as f64); 

        let mut measurement = 0f64;
        for signal in signal_description {
            let freq = signal.frequency as f64;
            measurement += signal.amplitude * f64::sin(2f64 * PI * freq * t + signal.offset);
        }

        measurements.push(measurement);
    }

    measurements
}

pub fn naive_dft(samples: &[f64]) -> Vec<Complex<f64>> {
    // TODO(knielsen): Add assertion about spacing between measurements

    let mut coefficients = Vec::with_capacity(samples.len());

    let N = samples.len();
    for j in 0..N {
        let mut c = Complex::<f64>::zero();
        for k in 0..N {
            let sample = Complex::new(samples[k], 0.0);
            let exponent = Complex::<f64>::new(0.0, -2.0 * PI * ((k * j) as f64) / (N as f64));

            c += sample * exponent.exp();
        }

        coefficients.push(c / Complex::<f64>::new(N as f64, 0.0));
    }

    coefficients
}

fn is_power_of_two(n: usize) -> bool {
    return (n != 0) && ((n & (n - 1)) == 0);
}

pub fn recursive_fft(samples: &[f64]) -> Vec<Complex<f64>> {
    let N = samples.len();
    assert!(is_power_of_two(N), "Samples must be a power of two!");
    if N <= 4 {
        return naive_dft(samples);
    }

    let even_samples: Vec<f64> = samples.iter().enumerate().filter(|(i, _)| i % 2 == 0).map(|(_, s)| s.clone()).collect();
    let odd_samples: Vec<f64> = samples.iter().enumerate().filter(|(i, _)| i % 2 != 0).map(|(_, s)| s.clone()).collect();

    let even_coef = recursive_fft(&even_samples);
    let odd_coef = recursive_fft(&odd_samples);

    let mut coefficients = Vec::with_capacity(N);
    for j in 0..N/2 {
        let exponent = Complex::<f64>::new(0.0, -2.0 * PI * (j as f64) / (N as f64));
        let c = Complex::<f64>::new(0.5, 0.0) * (even_coef[j] + exponent.exp() * odd_coef[j]);
        coefficients.push(c);
    }
    for j in 0..N/2 {
        let offset = N as f64 / 2.0;
        let exponent = Complex::<f64>::new(0.0, -2.0 * PI * (offset + j as f64) / (N as f64));
        let c = Complex::<f64>::new(0.5, 0.0) * (even_coef[j] + exponent.exp() * odd_coef[j]);
        coefficients.push(c);
    }

    coefficients
}

const fn num_bits<T>() -> usize { std::mem::size_of::<T>() * 8 }

fn log_2(x: usize) -> u32 {
    num_bits::<u64>() as u32 - x.leading_zeros() - 1
}

pub fn generate_permutation(N: usize) -> Vec<usize> {
    assert!(is_power_of_two(N), "Samples must be a power of two!");

    fn recurse(input: &[usize]) -> Vec<usize> {
        if input.len() == 0 {
            return vec![];
        }
        if input.len() == 1 {
            return vec![input[0]];
        }

        let even: Vec<usize> = input.iter().enumerate().filter(|(i, _)| i % 2 == 0).map(|(_, s)| s.clone()).collect();
        let odd: Vec<usize> = input.iter().enumerate().filter(|(i, _)| i % 2 != 0).map(|(_, s)| s.clone()).collect();

        let mut res = Vec::with_capacity(input.len());
        for e in recurse(&even) { res.push(e); }
        for o in recurse(&odd) { res.push(o); }

        res
    };

    let mut identity: Vec<usize> = Vec::with_capacity(N);
    for i in 0..N { identity.push(i); }
    recurse(&identity)
}

pub fn iterative_fft_prepare(samples: &[f64], permutation: &[usize]) -> Vec<f64> {
    assert!(is_power_of_two(samples.len()), "Samples must be a power of two!");

    let mut permuted = Vec::with_capacity(samples.len());
    for i in 0..samples.len() {
        permuted.push(samples[permutation[i]]);
    }
    permuted
}

/**
 * Assumes `samples` has been permutated to the correct form
 */
pub fn iterative_fft(samples: &[f64]) -> Vec<Complex<f64>> {
    let N = samples.len();
    assert!(is_power_of_two(N), "Samples must be a power of two!");

    let mut cur = vec![Complex::zero(); N];

    // Base case
    for i in 0..N {
        cur[i] = Complex::<f64>::new(samples[i], 0.0);
    }

    // Inductive case
    let levels = log_2(N);
    for level in (0 .. levels).rev() {
        let num_sublists = 2_u64.pow(level);

        let mut i: u64 = 0;
        while i < (N as u64) {
            let sublist_length = (N as u64) / num_sublists;
            let sublist = i / sublist_length;
            let sublist_start = sublist * sublist_length;
            let j = i % sublist_length;
            assert!(j < sublist_length / 2, "j should be only in the first half of the sublist!");

            let even_index = (sublist_start + j) as usize;
            let odd_index = (even_index + (sublist_length as usize) / 2) as usize;

            let prev_even = cur[even_index];
            let prev_odd = cur[odd_index];

            let exponent = Complex::<f64>::new(0.0, -2.0 * PI * (j as f64) / (sublist_length as f64)).exp();
            let pertubation = exponent * prev_odd;

            cur[even_index] = Complex::<f64>::new(0.5, 0.0) * (prev_even + pertubation);
            cur[odd_index] = Complex::<f64>::new(0.5, 0.0) * (prev_even - pertubation);

            if j + 1 == sublist_length / 2 {
                i += sublist_length / 2 + 1;
            } else {
                i += 1;
            }
        }
    }

    cur
}

fn validate_coefficients(actual: &[Complex<f64>], expected: &[Complex<f64>]) {
    const EPSILON: f64 = 0.00001;

    if actual.len() != expected.len() {
        panic!("Expected same length for actual ({}) and expected ({})", actual.len(), expected.len());
    }

    for i in 0 .. actual.len() {
        let diff = (actual[i] - expected[i]).norm();
        if diff > EPSILON {
            println!("Actual: {:?}", &actual);
            println!("");
            println!("Expected: {:?}", &expected);
            println!("");
            panic!("Difference: {}. Expected: {}, Actual: {}", diff, expected[i], actual[i]);
        }
    }
}

#[test]
fn test_recursive_fft() {
    let description = vec![
        SignalDescription { frequency: 6, amplitude: 1f64, offset: 0f64 },
        SignalDescription { frequency: 500, amplitude: 5f64, offset: 0.1337f64 },
        SignalDescription { frequency: 248, amplitude: 10f64, offset: 1.437f64 }
    ];

    let interval = 2.0; // seconds
    let samples = sample(interval, 1024, &description);

    let dft_coefficients = naive_dft(&samples);
    let fft_coefficients = recursive_fft(&samples);

    validate_coefficients(&fft_coefficients, &dft_coefficients);
}

#[test]
fn test_iterative_fft() {
    let description = vec![
        SignalDescription { frequency: 6, amplitude: 1f64, offset: 0f64 },
        SignalDescription { frequency: 500, amplitude: 5f64, offset: 0.1337f64 },
        SignalDescription { frequency: 248, amplitude: 10f64, offset: 1.437f64 }
    ];

    let interval = 2.0; // seconds
    let samples = sample(interval, 1024, &description);
    let permutation = generate_permutation(samples.len());

    let dft_coefficients = naive_dft(&samples);
    let fft_coefficients = iterative_fft(&iterative_fft_prepare(&samples, &permutation));

    validate_coefficients(&fft_coefficients, &dft_coefficients);
}
