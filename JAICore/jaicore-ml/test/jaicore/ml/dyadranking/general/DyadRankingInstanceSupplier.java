package jaicore.ml.dyadranking.general;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;

import de.upb.isys.linearalgebra.Vector;
import jaicore.ml.dyadranking.Dyad;
import jaicore.ml.dyadranking.dataset.DyadRankingDataset;
import jaicore.ml.dyadranking.dataset.DyadRankingInstance;

/**
 * Creates simple rankings for testing purposes.
 * 
 * @author Jonas Hanselle
 *
 */
public class DyadRankingInstanceSupplier {

	public static DyadRankingInstance getDyadRankingInstance(int maxLength) {
		List<Dyad> dyads = new ArrayList<Dyad>();
		int actualLength = ThreadLocalRandom.current().nextInt(2,maxLength+1);
		for(int i = 0; i < actualLength; i++) {
			Dyad dyad = DyadSupplier.getRandomDyad(2, 2);
			dyads.add(dyad);
		}
		Comparator<Dyad> comparator = new Comparator<Dyad>() {
			@Override
			public int compare(Dyad d1, Dyad d2) {
				Vector scoreVecI = d1.getInstance();
				Vector scoreVecA = d1.getAlternative();
				Vector scoreVecI2 = d2.getInstance();
				Vector scoreVecA2 = d2.getAlternative();
				double score1 = Math.pow(scoreVecI.getValue(0), 2) + Math.pow(scoreVecI.getValue(1), 2) - Math.pow(scoreVecA.getValue(0), 2) - Math.pow(scoreVecA.getValue(1), 2);
				double score2 = Math.pow(scoreVecI2.getValue(0), 2) + Math.pow(scoreVecI2.getValue(1), 2) - Math.pow(scoreVecA2.getValue(0), 2) - Math.pow(scoreVecA2.getValue(1), 2);
				return score1 - score2 == 0 ? 0 : (score1 - score2 > 0 ? 1 : -1);
			}
		};
		Collections.sort(dyads, comparator);
		return new DyadRankingInstance(dyads);
	}
	
	public static DyadRankingDataset getDyadRankingDataset(int maxLengthDyadRankingInstance, int length) {
		DyadRankingDataset dataset = new DyadRankingDataset();
		for(int i = 0; i < length; i++) {
			dataset.add(DyadRankingInstanceSupplier.getDyadRankingInstance(maxLengthDyadRankingInstance));
		}
		return dataset;
	}

}
