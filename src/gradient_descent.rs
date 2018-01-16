use std::f64;
use nalgebra::{MatrixN, Dynamic, VectorN};

pub trait Gradient {
    fn gradient(&self, x: &VectorN<f64, Dynamic>) -> VectorN<f64, Dynamic>;
    fn hessian(&self, x: &VectorN<f64, Dynamic>) -> MatrixN<f64, Dynamic>;
}

type DynVec = VectorN<f64, Dynamic>;

pub fn solve_iter(a: MatrixN<f64, Dynamic>, b: &DynVec) -> DynVec {
    let mut copy = a.clone();
    for col in 0..a.nrows() {
        for row in 0..a.ncols() {
            copy[(row, col)] += 1e-3;
        }
    }

    return if let Some(res) = copy.lu().solve(&b) {
        res
    } else {
        a.svd(true, true).solve(b, 1e-3)
    };
}

pub fn optimize<T: Gradient>(gradient: &T, mut start: VectorN<f64, Dynamic>) -> VectorN<f64, Dynamic> {
    let rate = 1.0;
    let mut last_grad = f64::MAX;
    // TODO: Add in more sophisticated stoping condition.
    loop {
        let mut grad = gradient.gradient(&start);
        let norm = grad.norm();
        println!("Gradient norm: {}", norm);
        if last_grad - norm < 1e-2 {
            break;
        } else {
            last_grad = norm;
        }
        let hess = gradient.hessian(&start);
        grad = solve_iter(hess, &grad);
        // let svd = hess.svd(true, true);
        // grad = svd.solve(&grad, 1e-6);
        start = start - rate * grad;
    }

    return start;
}
