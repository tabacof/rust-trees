use std::cmp::Ordering::Equal;

pub fn sort_two_vectors(a: &[f32], b: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let mut pairs: Vec<(&f32, &f32)> = a.iter().zip(b).collect();
    pairs.sort_by(|&a, &b| a.0.partial_cmp(b.0).unwrap_or(Equal));

    pairs.into_iter().map(|(x, y)| (*x, *y)).unzip()
}

pub fn float_avg(x: &[f32]) -> f32 {
    x.iter().sum::<f32>() / x.len() as f32
}

pub fn classification_threshold(x: &[f32], clf_threshold: f32) -> Vec<f32> {
    x.iter()
        .map(|&x| if x >= clf_threshold { 1.0 } else { 0.0 })
        .collect()
}

pub fn r2(x_true: &[f32], x_pred: &[f32]) -> f32 {
    let mse: f32 = x_true
        .iter()
        .zip(x_pred)
        .map(|(xt, xp)| (xt - xp).powf(2.0))
        .sum();

    let avg = float_avg(x_true);
    let var: f32 = x_true.iter().map(|x| (x - avg).powf(2.0)).sum();

    1.0 - mse / var
}

pub fn accuracy(x_true: &[f32], x_pred: &[f32]) -> f32 {
    x_true
        .iter()
        .zip(x_pred)
        .map(|(xt, xp)| ((xt == xp) as i32 as f32))
        .sum::<f32>()
        / x_true.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_float_avg() {
        let vector = vec![1.0, 2.0, 3.0];
        let expect = 2.0;
        assert_eq!(expect, float_avg(&vector));
    }

    #[test]
    fn test_sort_two_vectors() {
        let vec1 = vec![2.0, 3.0, 1.0];
        let vec2 = vec![6.0, 5.0, 4.0];

        let expect = (vec![1.0, 2.0, 3.0], vec![4.0, 6.0, 5.0]);

        let got = sort_two_vectors(&vec1, &vec2);
        assert_eq!(expect, got);
    }
}
