use ndarray::Array2;
use rand::distributions::{Distribution, Uniform};
use std::fs::File;
use std::io::{BufRead, BufReader};

// Set-up
fn one_at_index(ind: i32) -> Array2<f64> {
    let mut vec = vec![0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0];
    vec[ind as usize] = 1.0;
    let a = Array2::from_shape_vec((1,10), vec).unwrap();
    return a;
}

fn div(ls: &mut Vec<i32>, val: &i32) -> Array2<f64> {
    let vec: Vec<f64> = ls.iter().map(|&x| (x/val) as f64).collect();
    let a = Array2::from_shape_vec((1, vec.len()), vec).unwrap();
    return a;
}

fn process_data(in_out_ls: &mut Vec<(Array2<f64>, Array2<f64>)>, filename: &str) {
    let file = File::open(filename).expect("File failed to open");
    let reader = BufReader::new(file);

    for line in reader.lines() {
        let line = line.expect("Failed to parse line");

        let data: Vec<i32> = line.split(',').map(|s| s.parse::<i32>().expect("Failed to parse int")).collect();
        let output_data = one_at_index(data[0]);
        let input_data = div(&mut data[1..].to_vec(), &255);

        in_out_ls.push((input_data, output_data));      
    }
}

// Nerual Network Setup
fn print_network(network: &[(Array2<f64>, Array2<f64>)]) {
    for (l, layer) in network.iter().enumerate() {
        println!("layer: {}\nweight_matrix:\n{:?}\nbias_matrix:\n{:?}\n", l, layer.0, layer.1);
    }
}

fn create_random_wm(layer1_size: usize, layer2_size: usize) -> Array2<f64> {
    let mut ran_gen= rand::thread_rng();
    Array2::from_shape_fn((layer1_size, layer2_size), |_| Uniform::new(-1.0, 1.0).sample(&mut ran_gen))
}

fn create_random_bs(layer_size: usize) -> Array2<f64> {
    let mut ran_gen= rand::thread_rng();
    Array2::from_shape_fn((1, layer_size), |_| Uniform::new(-1.0, 1.0).sample(&mut ran_gen))
}

fn create_network(layer_sizes: &[usize]) -> Vec<(Array2<f64>, Array2<f64>)> {
    let mut network = Vec::with_capacity(layer_sizes.len() - 1);

    for i in 1..layer_sizes.len() {
        let size1 = layer_sizes[i - 1];
        let size2 = layer_sizes[i]; // current layer size
        let wm = create_random_wm(size1, size2);
        let bs = create_random_bs(size2);
        network.push((wm, bs));
    }

    network
}

// BACKWARD PROPAGATION
fn sigmoid(dot: &Array2<f64>) -> Array2<f64> {
    let result = dot.map(|&elem| (1.0/(1.0 + (-1.0*elem).exp())));
    result
}

fn sigmoid_prime(dot: &Array2<f64>) -> Array2<f64> {
    let sig = sigmoid(dot);
    &sig * (1.0 - &sig)
}

fn find_dot(w: &Array2<f64>, b: &Array2<f64>, x: &Array2<f64>) -> Array2<f64> {
    x.dot(w) + b
}

fn perceptron<F>(activation_function: F, dot: &Array2<f64>) -> Array2<f64>
where F: Fn(&Array2<f64>) -> Array2<f64>,
{
    activation_function(dot)
}

fn forward_propagate(
    x: &Array2<f64>,
    network: &[(Array2<f64>, Array2<f64>)],
) -> (Array2<f64>, Array2<f64>, Vec<Array2<f64>>, Vec<Array2<f64>>) {
    let mut a_vec = x.to_owned();
    let tmp = Array2::from_shape_vec((1,2), vec![0.0,0.0]).unwrap();
    let mut dot_vecs_ls = vec![tmp]; // don't use dot_vec[0]; just a placeholder
    let mut a_vecs_ls = vec![a_vec.to_owned()];

    for layer in network {
        let w_matrix = &layer.0;
        let b_scalar = &layer.1;
        let dot_vec = find_dot(w_matrix, b_scalar, &a_vec);
        a_vec = perceptron(sigmoid, &dot_vec);
        dot_vecs_ls.push(dot_vec.to_owned());
        a_vecs_ls.push(a_vec.to_owned());
    }

    (a_vec, dot_vecs_ls.last().unwrap().to_owned(), dot_vecs_ls, a_vecs_ls)
}

fn back_propagation(actual_table: &[(Array2<f64>, Array2<f64>)], network: &mut Vec<(Array2<f64>, Array2<f64>)>, k_num_epochs: i32) {
    // Training
    let mut lamb = 0.1;
    for i in 0..k_num_epochs {
        println!("Epoch: {}      Accuracy: {}", i, test_network_accuracy_mnist(actual_table, network));
        for tup in actual_table {
            let (x, expected_output) = tup;

            // Forward Propagate
            let (a_vec, dot_vec, dot_vecs_ls, a_vecs_ls) = forward_propagate(x, network); // a_vec = final layer's output perceptron, dot_vec = final layer's dot

            // Backward Propagate - Calculate Gradient Descent Values
            let mut delL_ls = vec![sigmoid_prime(&dot_vec) * (expected_output - &a_vec)]; // del_N           #del_Ls[-1] = delN (gradient function for LAST FUNCTION)
            for l in (0..network.len() - 1).rev() {
                let delL_vector = sigmoid_prime(&dot_vecs_ls[l + 1]) * (&delL_ls[0]).dot(&network[l + 1].0.t());
                delL_ls.insert(0, delL_vector);
            }

            // Backward Propagate - Update Values
            for (l, layer) in network.iter_mut().enumerate() {
                layer.1 = &layer.1 + (lamb * &delL_ls[l]); // update bias
                layer.0 = &layer.0 + (lamb * ((&a_vecs_ls[l]).t().dot(&delL_ls[l]))); // update weight
            }
        }
        lamb *= 0.99;
        //let mut file = File::create("network.pkl").unwrap();
        //bincode::serialize_into(&mut file, network).unwrap();
    }
}

fn arrays_equal(arr1: &Array2<f64>, arr2: &Array2<f64>) -> bool {
    if arr1.shape() != arr2.shape() {
        return false;
    }

    for (val1, val2) in arr1.iter().zip(arr2.iter()) {
        if val1 != val2 {
            return false;
        }
    }

    true
}

fn make_highest_one(arr: &Array2<f64>) -> Array2<f64> {
    let max_val = arr.fold(f64::NEG_INFINITY, |m, &x| m.max(x));
    let mut result = Array2::from_elem((arr.nrows(), arr.ncols()), 0.0);

    for ((i, j), &val) in arr.indexed_iter() {
        if val == max_val {
           result[[i, j]] = 1.0; 
        }
    }

    result
}

fn test_network_accuracy_mnist(actual_table: &[(Array2<f64>, Array2<f64>)], network: &mut Vec<(Array2<f64>, Array2<f64>)>) -> f64 {
    let mut num_correct: f64 = 0.0;
    let num_instances: f64 = actual_table.len() as f64;
    
    for tup in actual_table {
        let (x, expected_output) = tup;
        let (network_output, _, _, _) =  forward_propagate(x, network);
        if arrays_equal(&make_highest_one(&network_output), expected_output) {
            num_correct += 1.0;
        }
    }
    
    num_correct / num_instances
}

fn main() {
    // Global Variables
    let mut in_out_ls: Vec<(Array2<f64>, Array2<f64>)> = Vec::new(); // [ ( input_array <1x784 np.array of ints>, output_array <1x10 np.array of ints> ), ... ]
    const K_NUM_EPOCHS: i32 = 65;

    process_data(&mut in_out_ls, "mnist_train.csv");

    back_propagation(&in_out_ls, &mut create_network(&[784, 128, 64, 10]), K_NUM_EPOCHS);

    //let mut savefile = File::open("network.pkl").unwrap();
    //let mut network: Vec<(Array2<f64>, Array1<f64>)> = bincode::deserialize_from(&mut savefile).unwrap();
    //back_propagation(&in_out_ls, &mut network);
}