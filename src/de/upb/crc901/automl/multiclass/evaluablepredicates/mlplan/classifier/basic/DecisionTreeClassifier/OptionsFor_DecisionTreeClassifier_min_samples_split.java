
package de.upb.crc901.automl.multiclass.evaluablepredicates.mlplan.classifier.basic.DecisionTreeClassifier;
/*
    min_samples_split samples.

min_samples_split : int, float, optional (default=2)
    The minimum number of samples required to split an internal node:

    - If int, then consider `min_samples_split` as the minimum number.
    - If float, then `min_samples_split` is a percentage and
      `ceil(min_samples_split * n_samples)` are the minimum
      number of samples for each split.

    .. versionchanged:: 0.18
       Added float values for percentages.


*/

import de.upb.crc901.automl.multiclass.evaluablepredicates.mlplan.NumericRangeOptionPredicate;

public class OptionsFor_DecisionTreeClassifier_min_samples_split extends NumericRangeOptionPredicate {

  @Override
  protected double getMin() {
    return 3;
  }

  @Override
  protected double getMax() {
    return 10;
  }

  @Override
  protected int getSteps() {
    return 8;
  }

  @Override
  protected boolean needsIntegers() {
    return true;
  }
}
