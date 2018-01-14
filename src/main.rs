mod gradient_descent;
mod alchemy;

use alchemy::{EntropyConstraint, VarAndValue, EntropyOptimizer};

extern crate nalgebra;

fn main() {
    let mut contras: Vec<EntropyConstraint> = Vec::new();
    contras.push(EntropyConstraint::SingleNeq(VarAndValue{var: 0, value: 4}));
    contras.push(EntropyConstraint::SingleNeq(VarAndValue{var: 1, value: 1}));
    contras.push(EntropyConstraint::DoubleNeq(0, 1));
    contras.push(EntropyConstraint::DoubleNeq(1, 2));
    let optimizer = EntropyOptimizer{ varc:3, k: 25, contras};
    let best = optimizer.optimize();
    println!("{}", best);
    println!("{}", best.entropy());
}
