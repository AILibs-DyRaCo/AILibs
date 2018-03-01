package de.upb.crc901.mlplan.evaluablepredicates.mlplan.classifier.basic.SVC;

    import java.util.Arrays;
    import java.util.List;

    import de.upb.crc901.mlplan.evaluablepredicates.mlplan.OptionsPredicate;
    /*
        probability : boolean, optional (default=False)
        Whether to enable probability estimates. This must be enabled prior
        to calling `fit`, and will slow down that method.


    */
    public class OptionsFor_SVC_probability extends OptionsPredicate {
        
        private static List<Object> validValues = Arrays.asList(new Object[]{"true", "false"});

        @Override
        protected List<? extends Object> getValidValues() {
            return validValues;
        }
    }
    
