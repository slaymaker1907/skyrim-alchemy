use std::f64;
use nalgebra::{MatrixN, Dynamic, VectorN};

pub trait Gradient {
    fn gradient(&self, x: &VectorN<f64, Dynamic>) -> VectorN<f64, Dynamic>;
    fn hessian(&self, x: &VectorN<f64, Dynamic>) -> MatrixN<f64, Dynamic>;
}

type DynVec = VectorN<f64, Dynamic>;
type DynMat = MatrixN<f64, Dynamic>;

pub fn solve_iter(mut a: MatrixN<f64, Dynamic>, b: &mut DynVec) {
    for col in 0..a.nrows() {
        for row in 0..a.ncols() {
            a[(row, col)] += 1e-3;
        }
    }

    assert!(a.lu().solve_mut(b), "Was not able to solve (maybe not invertible?).");
}

pub fn optimize<T: Gradient>(gradient: &T, mut start: VectorN<f64, Dynamic>) -> VectorN<f64, Dynamic> {
    let rate = 1.0;
    // TODO: Add in more sophisticated stoping condition.
    loop {
        let mut grad = gradient.gradient(&start);
        if grad.norm() < 1e-2 {
            break;
        }
        println!("Gradient norm: {}", grad.norm());
        let hess = gradient.hessian(&start);
        solve_iter(hess, &mut grad);
        // let svd = hess.svd(true, true);
        // grad = svd.solve(&grad, 1e-6);
        start = start - rate * grad;
    }

    return start;
}
