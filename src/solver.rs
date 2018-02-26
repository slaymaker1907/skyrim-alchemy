use std::collections::{HashSet, HashMap};

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct VarAndValue {
    pub var: usize,
    pub value: usize
}

pub struct VarValIter {
    current: VarAndValue,
    end: VarAndValue
}

pub impl Iterator for VarValIter {
    type Item = VarAndValue;

    fn next(&mut self) -> Option<VarAndValue> {
        if self.current.var == self.end.var {
            return None;
        }

        let result = self.current;
        self.current.value += 1;
        if self.current.value == self.end.value {
            self.current.value = 0;
            self.current.var += 1;
        }

        return Some(result);
    }
}

pub impl VarAndValue {
    // varEnd is non-inclusive.
    pub fn enumerate(var_start: usize, var_end: usize, ksize: usize) -> VarValIter {
        let current = VarAndValue{var: var_start, value: 0};
        let end = VarAndValue{var: var_end, value: ksize};

        return VarValIter{
            current,
            end
        };
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum EntropyConstraint {
    DoubleNeq(usize, usize),
    SingleNeq(VarAndValue)
}


use self::EntropyConstraint{DoubleNeq, SingleNeq};

pub struct EntropyOptimizer {
    pub varc: usize,
    pub k: usize,
    pub contras: Vec<EntropyConstraint>
}

impl EntropyOptimizer {
    pub fn separate(&self) -> Vec<EntropySubProblem> {
        let mut vars: HashSet<usize> = HashSet::new();
        for var in 0..self.varc {
            vars.insert(var);
        }

        let mut result: Vec<EntropySubProblem> = Vec::new();
        while !vars.is_empty() {
            let mut var_map: HashMap<usize, usize> = HashMap::new();
            var_map.insert(vars.iter().next().unwrap(), 0);

            let mut need_update = true;
            while need_update {
                need_update = false;
                for &contra in self.contras.iter() {
                    if DoubleNeq(first, second) = contra {
                        if var_map.get(&first).is_some() != var_map.get(&second).is_some() {
                            need_update = true;
                            var_map.entry(&first).or_insert(var_map.len());
                            var_map.entry(&second).or_insert(var_map.len());
                        }
                    }
                }
            }

            let mut new_contras: Vec<EntropyConstraint> = Vec::new();
            let mut val_map: HashMap<usize, usize> = HashMap::new();
            for &contra in self.contras.iter() {
                match contra {
                    DoubleNeq(first, second) => {
                        let new_first = var_map.get(&first);
                        let new_second = var_map.get(&second);
                        if has_first.is_some() {
                            new_contras.push(DoubleNeq(new_first.unwrap(), new_second.unwrap()));
                        }
                    },
                    SingleNeq(VarAndValue{var, value}) => {
                        let has_var = var_map.get(&var);
                        if has_var.is_some() {
                            let new_value = val_map.entry(&value).or_insert(val_map.len());
                            new_contras.push(SingleNeq(VarAndValue{var: has_var.unwrap(), value: new_value}));
                        }
                    }
                }
            }

            for &to_remove in var_map.keys() {
                vars.remove(&to_remove);
            }

            let not_mentioned: Vec<usize> = (0..self.k).filter(|k| val_map.get(&k).is_none()).collect();
            let mut val_arr: Vec<Vec<usize>> = vec![Vec::new(); val_map.len()];

            for &(old_val, new_val) in val_map.iter() {
                val_arr[new_val].push(old_val);
            }

            val_arr.push(not_mentioned);

            let mut var_arr: Vec<usize> = vec![0; var_map.len()];
            for &(old_val, new_val) in var_map.iter() {
                var_arr[new_val] = old_val;
            }

            result.push(EntropySubProblem{
                varMap: var_arr,
                kMap: val_arr,
                contras: new_contras
            });
        }

        return result;
    }
}

pub struct EntropySubProblem {
    pub varcMap: Vec<usize>, // varc of problem is varcMap.length() and each element is a map back to original variable.
    pub kMap: Vec<Vec<usize>>, // Each element is the k values from original problem.
    pub contras: Vec<EntropyConstraint>
}

// Add together entropy from all subproblems for total entropy.
impl EntropySubProblem {
    pub fn varc(&self) -> usize {
        return varcMap.len();
    }

    pub fn k(&self) -> usize {
        return kMap.len();
    }

    pub fn kcount(&self, kval: usize) -> usize {
        return kMap.get(kval).unwrap().len();
    }

    pub fn entropy<Solution: ProblemSolution>(&self, solution: &Solution) -> f64 {
        let mut result: f64 = 0.0;
        for var1 in VarAndValue::enumerate(0, self.varc(), self.k()) {
            let count1 = self.kcount(var.value);
            for var2 in VarAndValue::enumerate(var1.var+1, self.varc(), self.k()) {
                // full_count is the number of events represented uniformly under the given probability.
                let full_count = count1 * self.kcount(var2.value);
                let prob = solution.prob(var1, var2) / full_count;
                result += -prob * (prob / full_count).log2(); // Divide by full_count since uniformly distributed over multiple events.
            }
        }

        return result;
    }
}
