use gradient_descent;
use nalgebra::{MatrixN, Dynamic, VectorN};
use std::fmt::{Display, Formatter};
use std::collections::{HashMap, HashSet};
use std::fmt;

type DynMatrix = MatrixN<f64, Dynamic>;
type DynVector = VectorN<f64, Dynamic>;

pub struct OptimizationResult {
    var_meaning: Vec<VariableType>,
    optimized: Vec<f64>,
    varc: usize,
    k: usize
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
        let mut result = 0.0;
        for n in 0..self.varc {
            for k in 0..self.k {
                let prob = self.var_prob(n, k);
                if prob > 0.0 {
                    result -= prob * prob.log2();
                }
            }
        }
        return result;
    }

    pub fn var_prob(&self, var: usize, value: usize) -> f64 {
        let varval = VarAndValue{ var, value };
        // let divamt = self.varc as f64;
        let target = if var == 0 {
            1
        } else {
            var - 1
        };
        let summ: f64 = self.var_meaning.iter()
            .zip(self.optimized.iter())
            .filter(|&(meaning, _)| {
                match *meaning {
                    VariableType::BaseVariable{ var1, var2, .. } => 
                    (var1==varval && var2.var == target) || (var2==varval && var1.var == target),
                    _ => false
                }
            })
            .map(|(_, prob)| prob)
            .sum();
        return summ; // Need to divide due to multiple universes considered.
    }
}

pub struct EntropyOptimizer {
    pub varc: usize,
    pub k: usize,
    pub contras: Vec<EntropyConstraint>
}

#[derive(PartialEq, Eq, Hash)]
struct PartialLagrangian {
    given: VarAndValue,
    free: usize
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

        for &(n1, n2) in required_joints.iter() {
            let mut sum_to_one: Vec<usize> = Vec::new();
            for k1 in 0..self.k {
                for k2 in 0..self.k {
                    if !self.is_constrained(n1, k1, n2, k2) {
                        let var1 = VarAndValue{ var: n1, value: k1 };
                        let var2 = VarAndValue{ var: n2, value: k2 };
                        let current_pos = var_meaning.len();
                        partials.entry(PartialLagrangian{ given: var1, free: var2.var})
                            .or_insert_with(|| Vec::new())
                            .push(current_pos);
                        partials.entry(PartialLagrangian{ given: var2, free: var1.var})
                            .or_insert_with(|| Vec::new())
                            .push(current_pos);
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
            var_meaning.push(Lagrangian{ sum_to_one: lag });
        }

        for n in 0..self.varc {
            if !mentioned.contains(&n) {
                let variables: Vec<VarAndValue> = Vec::new();
                for k in 0..self.k {
                    let varval = VarAndValue{ var: n, value: k };
                    let contra = EntropyConstraint::SingleNeq(varval);
                    if !self.contras.contains(&contra) {
                        variables.push(varval);
                    }
                }

                let sum_to_one: Vec<usize> = Vec::new();
                let start = var_meaning.len();
                for i in start..(start+variables.len()) {
                    sum_to_one.push(i);
                }

                let lagind = start + variables.len();
                for var in variables {
                    var_meaning.push(SingleVariable{ var, lagrangian: lagind });
                }
                var_meaning.push(Lagrangian{ sum_to_one });
            }
        }

        for n1 in 0..self.varc {
            for k in 0..self.k {
                let mut to_eq: Vec<Vec<usize>> = Vec::new();
                let given = VarAndValue{ var: n1, value: k };
                for n2 in 0..self.varc {
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

        return OptimizationResult{ 
            var_meaning, 
            optimized: result.iter().map(|x| *x).collect(), 
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

#[derive(Debug, Clone)]
enum VariableType {
    BaseVariable{ var1: VarAndValue, var2: VarAndValue, lagrangians: Vec<usize>, neg_lags: Vec<usize> },
    SingleVariable{ var: VarAndValue, lagrangian: usize},
    Lagrangian{ sum_to_one: Vec<usize> },
    EquivalentSums(Vec<usize>, Vec<usize>)
}

use self::VariableType::{BaseVariable, Lagrangian, EquivalentSums, SingleVariable};

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
                &Lagrangian{ ref sum_to_one } => {
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
                        &Lagrangian{ ref sum_to_one } => {
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

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct VarAndValue {
    pub var: usize,
    pub value: usize
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum EntropyConstraint {
    DoubleNeq(usize, usize),
    SingleNeq(VarAndValue)
}
