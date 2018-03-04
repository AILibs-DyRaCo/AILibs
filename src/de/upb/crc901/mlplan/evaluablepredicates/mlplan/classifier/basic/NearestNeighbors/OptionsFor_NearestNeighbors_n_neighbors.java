
    package de.upb.crc901.mlplan.evaluablepredicates.mlplan.classifier.basic.NearestNeighbors;
    /*
        n_neighbors : int, optional (default = 5)
        Number of neighbors to use by default for :meth:`kneighbors` queries.


    */

    import java.util.Arrays;
import java.util.List;

import de.upb.crc901.mlplan.evaluablepredicates.mlplan.OptionsPredicate;

    public class OptionsFor_NearestNeighbors_n_neighbors extends OptionsPredicate {
        
        private static List<Integer> validValues = Arrays.asList(new Integer[]{1, 3, 5, 11, 21, 51, 101});

        @Override
        protected List<? extends Object> getValidValues() {
            return validValues;
        }
    }
    
