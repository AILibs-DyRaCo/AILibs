
package de.upb.crc901.automl.multiclass.evaluablepredicates.mlplan.classifier.basic.ExtraTreeClassifier;
/*
    min_samples_leaf : int, float, optional (default=1)
    The minimum number of samples required to be at a leaf node:

    - If int, then consider `min_samples_leaf` as the minimum number.
    - If float, then `min_samples_leaf` is a percentage and
      `ceil(min_samples_leaf * n_samples)` are the minimum
      number of samples for each node.

    .. versionchanged:: 0.18
       Added float values for percentages.


*/

import de.upb.crc901.automl.multiclass.evaluablepredicates.mlplan.NumericRangeOptionPredicate;

public class OptionsFor_ExtraTreeClassifier_min_samples_leaf extends NumericRangeOptionPredicate {

  @Override
  protected double getMin() {
    return 2;
  }

  @Override
  protected double getMax() {
    return 10;
  }

  @Override
  protected int getSteps() {
    return 10;
  }

  @Override
  protected boolean needsIntegers() {
    return true;
  }
}
