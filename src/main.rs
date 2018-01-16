mod gradient_descent;
mod alchemy;

use std::collections::HashSet;

use alchemy::{EntropyConstraint, VarAndValue, EntropyOptimizer};

extern crate nalgebra;

fn main() {
    let mut contras: HashSet<EntropyConstraint> = HashSet::new();
    contras.insert(EntropyConstraint::SingleNeq(VarAndValue{var: 0, value: 4}));
    contras.insert(EntropyConstraint::SingleNeq(VarAndValue{var: 1, value: 1}));
    contras.insert(EntropyConstraint::DoubleNeq(0, 1));
    contras.insert(EntropyConstraint::DoubleNeq(1, 2));
    let optimizer = EntropyOptimizer{ varc:3, k: 25, contras};
    let best = optimizer.optimize();
    println!("{}", best);
    println!("{}", best.entropy());
}
