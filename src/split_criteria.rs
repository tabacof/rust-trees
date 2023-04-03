use crate::utils;

pub(crate) type SplitFunction = fn(
    col_index: usize,
    feature_name: &str,
    min_samples_leaf: i32,
    feature: &[f32],
    target: &[f32],
) -> SplitResult;

//pub(crate) trait SplitCriteria {
//    fn split_feature(col_index: usize, feature: &[f32], target: &[f32]) -> SplitResult;
//}

#[derive(Debug, PartialEq)]
pub(crate) struct SplitResult {
    pub(crate) col_index: usize,
    pub(crate) feature_name: String,
    pub(crate) row_index: usize,
    pub(crate) split: f32,
    pub(crate) prediction: f32,
    pub(crate) loss: f32,
}

//pub(crate) struct MeanSquaredError;

//impl SplitCriteria for MeanSquaredError {
pub(crate) fn mean_squared_error_split_feature(
    col_index: usize,
    feature_name: &str,
    min_samples_leaf: i32,
    feature: &[f32],
    target: &[f32],
) -> SplitResult {
    let (sorted_feature, sorted_target) = utils::sort_two_vectors(feature, target);

    let mut row_index = 1;
    let mut min_mse = f32::MAX;
    let mut last = sorted_feature[0];

    let square: Vec<f32> = sorted_target
        .iter()
        .map(|x| x * x)
        .scan(0.0, |state, x| {
            *state += x;
            Some(*state)
        })
        .collect();
    let sum: Vec<f32> = sorted_target
        .iter()
        .scan(0.0, |state, x| {
            *state += x;
            Some(*state)
        })
        .collect();

    for i in min_samples_leaf as usize..((sorted_feature.len() - min_samples_leaf as usize) + 1) {
        if sorted_feature[i] > last {
            //    var = \sum_i^n (y_i - y_bar) ** 2
            //           = (\sum_i^n y_i ** 2) - n_samples * y_bar ** 2
            //
            let left_square_sum = square[i - 1];
            let right_square_sum = square[square.len() - 1] - square[i - 1];

            let left_avg = sum[i - 1] / (i as f32);
            let right_avg = (sum[sum.len() - 1] - sum[i - 1]) / (sum.len() - i) as f32;

            let right_mse = right_square_sum - (sum.len() - i) as f32 * right_avg * right_avg;
            let left_mse = left_square_sum - i as f32 * left_avg * left_avg;
            let mse = left_mse + right_mse;

            if mse < min_mse {
                row_index = i;
                min_mse = mse;
            }

            last = sorted_feature[i];
        }
    }

    SplitResult {
        col_index,
        feature_name: feature_name.to_string(),
        row_index,
        split: (sorted_feature[row_index] + sorted_feature[row_index - 1]) / 2.0,
        prediction: utils::float_avg(target),
        loss: min_mse,
    }
}
//}

//pub(crate) struct GiniCoefficient;

//impl SplitCriteria for GiniCoefficient {
pub(crate) fn gini_coefficient_split_feature(
    col_index: usize,
    feature_name: &str,
    min_samples_leaf: i32,
    feature: &[f32],
    target: &[f32],
) -> SplitResult {
    let (sorted_feature, sorted_target) = utils::sort_two_vectors(feature, target);

    let mut row_index = 1;
    let mut min_gini = f32::MAX;
    let mut last = sorted_feature[0];

    let cumsum: Vec<f32> = sorted_target
        .iter()
        .scan(0.0, |state, x| {
            *state += x;
            Some(*state)
        })
        .collect();

    for i in min_samples_leaf as usize..((sorted_feature.len() - min_samples_leaf as usize) + 1) {
        if sorted_feature[i] > last {
            let left_cumsum = cumsum[i - 1] / (i as f32);
            let right_cumsum =
                (cumsum[cumsum.len() - 1] - cumsum[i - 1]) / (cumsum.len() - i) as f32;

            let gini_left =
                1. - ((left_cumsum * left_cumsum) + ((1.0 - left_cumsum) * (1.0 - left_cumsum)));

            let gini_right = 1.
                - ((right_cumsum * right_cumsum) + ((1.0 - right_cumsum) * (1.0 - right_cumsum)));

            let gini = gini_left * (i as f32) + gini_right * (cumsum.len() - i) as f32;

            let gini = gini / cumsum.len() as f32;

            if gini < min_gini {
                row_index = i;
                min_gini = gini;
            }

            last = sorted_feature[i];
        }
    }

    SplitResult {
        col_index,
        feature_name: feature_name.to_string(),
        row_index,
        split: (sorted_feature[row_index] + sorted_feature[row_index - 1]) / 2.0,
        prediction: utils::float_avg(target),
        loss: min_gini,
    }
}
//}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_gini_coefficient_split_feature() {
        assert_eq!(
            mean_squared_error_split_feature(
                1,
                "feature_a",
                1,
                &vec![2.0, 0.0, 1.0],
                &vec![-1.0, 0.0, 1.0]
            ),
            SplitResult {
                col_index: 1,
                feature_name: "feature_a".to_string(),
                row_index: 2,
                split: 1.5, // takes the average between the value to split on and the previous
                prediction: 0.0,
                loss: 0.5,
            }
        );
    }

    #[test]
    fn test_mean_squared_error() {
        assert_eq!(
            mean_squared_error_split_feature(
                1,
                "feature_a",
                1,
                &vec![2.0, 0.0, 1.0],
                &vec![-1.0, 0.0, 1.0]
            ),
            SplitResult {
                col_index: 1,
                feature_name: "feature_a".to_string(),
                row_index: 2,
                split: 1.5, // takes the average between the value to split on and the previous
                prediction: 0.0,
                loss: 0.5,
            }
        );
    }
}
