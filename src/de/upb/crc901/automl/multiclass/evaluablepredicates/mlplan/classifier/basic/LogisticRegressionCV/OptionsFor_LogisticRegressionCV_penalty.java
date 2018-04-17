package de.upb.crc901.automl.multiclass.evaluablepredicates.mlplan.classifier.basic.LogisticRegressionCV;

import java.util.Arrays;
import java.util.List;

import de.upb.crc901.automl.multiclass.evaluablepredicates.mlplan.OptionsPredicate;

/*
    penalty : str, 'l1' or 'l2'
    Used to specify the norm used in the penalization. The 'newton-cg',
    'sag' and 'lbfgs' solvers support only l2 penalties.


*/
public class OptionsFor_LogisticRegressionCV_penalty extends OptionsPredicate {

  // FIXME remove the default value from the list
  private static List<Object> validValues = Arrays.asList(new Object[] { "l1", "l2" });

  @Override
  protected List<? extends Object> getValidValues() {
    return validValues;
  }
}
