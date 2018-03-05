package de.upb.crc901.mlplan.evaluablepredicates.mlplan.classifier.basic.LogisticRegressionCV;

import de.upb.crc901.mlplan.evaluablepredicates.mlplan.OptionsPredicate;

import java.util.Arrays;
import java.util.List;

/*
    fit_intercept : bool, default: True
    Specifies if a constant (a.k.a. bias or intercept) should be
    added to the decision function.


*/
public class OptionsFor_LogisticRegressionCV_fit_intercept extends OptionsPredicate {

  private static List<Object> validValues = Arrays.asList(new Object[] { "false" });

  @Override
  protected List<? extends Object> getValidValues() {
    return validValues;
  }
}
