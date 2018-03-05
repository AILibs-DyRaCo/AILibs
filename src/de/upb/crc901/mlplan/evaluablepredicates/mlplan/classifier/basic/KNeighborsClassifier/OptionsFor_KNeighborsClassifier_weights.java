package de.upb.crc901.mlplan.evaluablepredicates.mlplan.classifier.basic.KNeighborsClassifier;

import de.upb.crc901.mlplan.evaluablepredicates.mlplan.OptionsPredicate;

import java.util.Arrays;
import java.util.List;

/*
    weights : str or callable, optional (default = 'uniform')
    weight function used in prediction.  Possible values:

    - 'uniform' : uniform weights.  All points in each neighborhood
      are weighted equally.
    - 'distance' : weight points by the inverse of their distance.
      in this case, closer neighbors of a query point will have a
      greater influence than neighbors which are further away.
    - [callable] : a user-defined function which accepts an
      array of distances, and returns an array of the same shape
      containing the weights.


*/
public class OptionsFor_KNeighborsClassifier_weights extends OptionsPredicate {

  private static List<Object> validValues = Arrays.asList(new Object[] { "distance" });

  @Override
  protected List<? extends Object> getValidValues() {
    return validValues;
  }
}
