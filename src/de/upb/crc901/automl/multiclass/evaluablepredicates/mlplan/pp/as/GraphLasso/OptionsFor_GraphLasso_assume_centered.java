package de.upb.crc901.automl.multiclass.evaluablepredicates.mlplan.pp.as.GraphLasso;

    import java.util.Arrays;
    import java.util.List;

import de.upb.crc901.automl.multiclass.evaluablepredicates.mlplan.OptionsPredicate;
    /*
        assume_centered : boolean, default False
        If True, data are not centered before computation.
        Useful when working with data whose mean is almost, but not exactly
        zero.
        If False, data are centered before computation.

    Attributes
    
    */
    public class OptionsFor_GraphLasso_assume_centered extends OptionsPredicate {
        
        private static List<Object> validValues = Arrays.asList(new Object[]{"true"});

        @Override
        protected List<? extends Object> getValidValues() {
            return validValues;
        }
    }
    
