use gradient_descent;
use nalgebra::{MatrixN, Dynamic, VectorN};
use std::fmt::{Display, Formatter};
use std::collections::{HashMap, HashSet};
use std::fmt;

type DynMatrix = MatrixN<f64, Dynamic>;
type DynVector = VectorN<f64, Dynamic>;

#[derive(Debug, Clone)]
enum VariableType {
    BaseVariable{ var1: VarAndValue, var2: VarAndValue, lagrangians: Vec<usize>, neg_lags: Vec<usize> },
    Lagrangian(Vec<usize>),
    EquivalentSums(Vec<usize>, Vec<usize>)
}

use self::VariableType::{BaseVariable, Lagrangian, EquivalentSums};

pub struct OptimizationResult {
    distribution: HashMap<VarAndValue, f64>,
    varc: usize,
    k: usize
}

#[derive(PartialEq, Eq, Hash)]
struct PartialLagrangian {
    given: VarAndValue,
    free: usize
}

pub struct EntropyOptimizer {
    pub varc: usize,
    pub k: usize,
    pub contras: HashSet<EntropyConstraint>
}

impl Display for OptimizationResult {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        try!(write!(f, "{{\n"));
        for n in 0..self.varc {
            for k in 0..self.k {
                try!(write!(f, "\tPr[{}={}] = {}\n", n, k, self.var_prob(n, k)));
            }
        }
        write!(f, "}}\n")
    }
}

impl OptimizationResult {
    pub fn entropy(&self) -> f64 {
        let summ: f64 = self.distribution
            .values()
            .filter(|&&val| val > 0.0)
            .map(|val| val * val.log2())
            .sum();
        return -1.0 * summ;
    }

    pub fn var_prob(&self, var: usize, value: usize) -> f64 {
        let varval = VarAndValue{ var, value };
        let zero = 0.0;
        return *self.distribution.get(&varval).unwrap_or(&zero);
    }
}

impl EntropyOptimizer {
    fn required_joints(&self) -> Vec<(usize, usize)>  {
        // Remove joint probabilities where possible.
        let mut required_joints: Vec<(usize, usize)> = Vec::new();
        for &contra in self.contras.iter() {
            if let EntropyConstraint::DoubleNeq(first, second) = contra {
                required_joints.push((first, second));
            }
        }

        return required_joints;
    }

    pub fn optimize(&self) -> OptimizationResult {
        let mut var_meaning: Vec<VariableType> = Vec::new();
        let mut lagrangians: Vec<Vec<usize>> = Vec::new();
        let mut partials: HashMap<PartialLagrangian, Vec<usize>> = HashMap::new();
        let required_joints = self.required_joints();
        let mentioned: HashSet<usize> = required_joints.iter().flat_map(|&(one, two)| vec![one, two]).collect();
        let mut distribution: HashMap<VarAndValue, f64> = HashMap::new();

        fn add_partials(
            partials: &mut HashMap<PartialLagrangian, Vec<usize>>, 
            given: VarAndValue, 
            free: VarAndValue,
            current_pos: usize) 
        {
            partials
                .entry(PartialLagrangian{ given, free: free.var})
                .or_insert_with(|| Vec::new())
                .push(current_pos);
        }

        for &(n1, n2) in required_joints.iter() {
            let mut sum_to_one: Vec<usize> = Vec::new();
            for k1 in 0..self.k {
                for k2 in 0..self.k {
                    if !self.is_constrained(n1, k1, n2, k2) {
                        let var1 = VarAndValue{ var: n1, value: k1 };
                        let var2 = VarAndValue{ var: n2, value: k2 };
                        let current_pos = var_meaning.len();
                        add_partials(&mut partials, var1, var2, current_pos);
                        add_partials(&mut partials, var2, var1, current_pos);
                        sum_to_one.push(var_meaning.len());
                        var_meaning.push(BaseVariable{ var1, var2 , lagrangians: Vec::new(), neg_lags: Vec::new()});
                    }
                }
            }

            lagrangians.push(sum_to_one);
        }

        for lag in lagrangians {
            let lagind = var_meaning.len();
            for &i in lag.iter() {
                if let BaseVariable{ ref mut lagrangians, .. } = var_meaning[i] {
                    lagrangians.push(lagind);
                }
            }
            var_meaning.push(Lagrangian(lag));
        }

        // If no joints, can just assume uniform distribution!
        for n in 0..self.varc {
            if mentioned.contains(&n) {
                continue;
            }
            let mut variables: Vec<VarAndValue> = Vec::new();
            for k in 0..self.k {
                let varval = VarAndValue{ var: n, value: k };
                let contra = EntropyConstraint::SingleNeq(varval);
                if !self.contras.contains(&contra) {
                    variables.push(varval);
                }
            }

            let prob = 1.0 / (variables.len() as f64);
            for var in variables {
                distribution.insert(var, prob);
            }
        }

        // Only need to check mentioned values for equivalencies.
        for &n1 in mentioned.iter() {
            for k in 0..self.k {
                let mut to_eq: Vec<Vec<usize>> = Vec::new();
                let given = VarAndValue{ var: n1, value: k };
                for &n2 in mentioned.iter() {
                    if n1 == n2 {
                        continue;
                    }
                    let part = PartialLagrangian{ given, free: n2 };
                    if let Some(entry) = partials.get(&part) {
                        to_eq.push(entry.clone());
                    }
                }

                if to_eq.len() > 1 {
                    for i1 in 0..to_eq.len() {
                        for i2 in 0..to_eq.len() {
                            if i1 == i2 {
                                continue;
                            }
                            let lagind = var_meaning.len();
                            for &child in to_eq[i1].iter() {
                                if let BaseVariable{ ref mut lagrangians, .. } = var_meaning[child] {
                                    lagrangians.push(lagind);
                                } else {
                                    panic!("Unexpected variable type.")
                                }
                            }
                            for &child in to_eq[i2].iter() {
                                if let BaseVariable{ ref mut neg_lags, .. } = var_meaning[child] {
                                    neg_lags.push(lagind);
                                } else {
                                    panic!("Unexpected variable type.")
                                }
                            }
                            var_meaning.push(EquivalentSums(to_eq[i1].clone(), to_eq[i2].clone()))
                        }
                    }
                }
            }
        }

        let size = var_meaning.len();
        let gradient = EntropyGradient{ var_meaning: var_meaning.clone() };
        let start = DynVector::from_element(size, 0.5);
        let result = gradient_descent::optimize(&gradient, start);

        for n in 0..self.varc {
            for k in 0..self.k {
                let varval = VarAndValue{ var: n, value: k };
                if mentioned.contains(&n) {
                    let other = required_joints.iter()
                        .filter(|&&(first, second)| first == n || second == n)
                        .next()
                        .map(|&(first, second)| {
                            if first == n {
                                second
                            } else {
                                first
                            }
                        })
                        .expect("If mentioned, should be in required_joints.");
                    let probability: f64 = var_meaning.iter().enumerate()
                        .filter(|&(_, meaning)| {
                            if let &BaseVariable{ var1, var2, .. } = meaning {
                                (var1 == varval && var2.var == other) ||
                                (var2 == varval && var1.var == other)
                            } else {
                                false
                            }
                        })
                        .map(|(i, _)| result[i]).sum();
                    distribution.insert(varval, probability);
                }
            }
        }

        return OptimizationResult{ 
            distribution,
            k: self.k,
            varc: self.varc
        }
    }

    fn is_constrained(&self, var1: usize, val1: usize, var2: usize, val2: usize) -> bool {
        self.contras.iter().any(|contra| {
            match *contra {
                EntropyConstraint::DoubleNeq(test1, test2) => {
                    return val1 == val2 && ((var1 == test1 && var2 == test2) || (var1 == test2 && var2 == test1));
                },
                EntropyConstraint::SingleNeq(varval) => {
                    return (var1 == varval.var && val1 == varval.value) || (var2 == varval.var && val2 == varval.value);
                }
            }
        })
    }
}

const MULT: f64 = 1.0;

struct EntropyGradient {
    var_meaning: Vec<VariableType>
}

impl gradient_descent::Gradient for EntropyGradient {
    fn gradient(&self, x: &VectorN<f64, Dynamic>) -> VectorN<f64, Dynamic> {
        let mut result: VectorN<f64, Dynamic> = DynVector::from_element(x.len(), 0.0);
        for (i, var_type) in self.var_meaning.iter().enumerate() {
            match var_type {
                &BaseVariable{ ref lagrangians, ref neg_lags, .. } => {
                    let prob_part = MULT * (x[i].ln() + 1.0);
                    let lag_sum: f64 = lagrangians.iter().map(|&i2| x[i2]).sum();
                    let neg_sum: f64 = neg_lags.iter().map(|&i2| x[i2]).sum();
                    result[i] = prob_part + lag_sum - neg_sum;
                },
                &Lagrangian(ref sum_to_one ) => {
                    let sum: f64 = sum_to_one.iter().map(|&i2| x[i2]).sum();
                    result[i] = sum - 1.0;
                },
                &EquivalentSums(ref to_add, ref to_min) => {
                    let sum: f64 = to_add.iter().map(|&i2| x[i2]).sum();
                    let minus: f64 = to_min.iter().map(|&i2| x[i2]).sum();
                    result[i] = sum - minus;
                }
            }
        }

        return result;
    }

    fn hessian(&self, x: &VectorN<f64, Dynamic>) -> MatrixN<f64, Dynamic> {
        DynMatrix::from_fn(x.len(), x.len(), |row, column| {
            // Row is the first partial, column is the second one.
            match self.var_meaning.get(row).unwrap() {
                &BaseVariable{..} => {
                    match self.var_meaning.get(column).unwrap() {
                        &BaseVariable{ .. } => {
                            return if row == column {
                                MULT / x[row]
                            } else {
                                0.0
                            };
                        },
                        &Lagrangian(ref sum_to_one) => {
                            return if sum_to_one.binary_search(&row).is_ok() {
                                1.0
                            } else {
                                0.0
                            };
                        },
                        &EquivalentSums(ref pos, ref neg) => {
                            return if pos.binary_search(&row).is_ok() {
                                1.0
                            } else if neg.binary_search(&row).is_ok() {
                                -1.0
                            } else {
                                0.0
                            }
                        }
                    }
                },
                &Lagrangian{ .. } => {
                    match self.var_meaning.get(column).unwrap() {
                        &BaseVariable{ ref lagrangians, .. } => {
                            return if lagrangians.binary_search(&row).is_ok() {
                                1.0
                            } else {
                                0.0
                            };
                        },
                        _ => {
                            return 0.0;
                        }
                    }
                },
                &EquivalentSums(_, _) => {
                    match self.var_meaning.get(column).unwrap() {
                        &BaseVariable{ ref lagrangians, ref neg_lags, .. } => {
                            return if lagrangians.binary_search(&row).is_ok() {
                                1.0
                            } else if neg_lags.binary_search(&row).is_ok() {
                                -1.0
                            } else {
                                0.0
                            };
                        },
                        _ => {
                            return 0.0;
                        }
                    }
                }
            }
        })
    }
}

