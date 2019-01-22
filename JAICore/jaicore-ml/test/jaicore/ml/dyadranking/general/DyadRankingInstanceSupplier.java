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
 * @author Jonas Hanselle, Mirko Jürgens, Helena Graf, Michael Braun
 *
 */
public class DyadRankingInstanceSupplier {

	/**
	 * Creates a random {@link jaicore.ml.dyadranking.dataset.DyadRankingInstance}
	 * consisting of (with 2 alternatives and 2 instances)
	 * 
	 * @param maxLength
	 *            the amount of dyads
	 * @return random dyad ranking instance of length at most maxLength
	 */
	public static DyadRankingInstance getDyadRankingInstance(int maxLength, int seed) {
		List<Dyad> dyads = new ArrayList<Dyad>();
		int actualLength = ThreadLocalRandom.current().nextInt(2, maxLength + 1);

		for (int i = 0; i < actualLength; i++) {
			Dyad dyad = DyadSupplier.getRandomDyad(30, 2);
			dyads.add(dyad);
		}
		Comparator<Dyad> comparator = complexDyadRanker();
		Collections.sort(dyads, comparator);
		return new DyadRankingInstance(dyads);
	}

	/**
	 * Creates a comparator for {@link jaicore.ml.dyadranking.Dyad} (with 2
	 * instances x_1, x_2 and 2 alternatives y_1,y_2). A pair of dyads (d_i, d_j) is
	 * then ranked by the rule d_i >= d_j iff x_i1^2 + x_i2^2 - y_i1^2 - y_i2^2 >
	 * x_j1^2 + x_j2^2 - y_j1^2 - y_j2^2
	 * 
	 * @return Comparator for dyads with 2 instances and 2 alternatives
	 */
	public static Comparator<Dyad> complexDyadRanker() {
		Comparator<Dyad> comparator = new Comparator<Dyad>() {
			@Override
			public int compare(Dyad d1, Dyad d2) {
				Vector scoreVecI = d1.getInstance();
				Vector scoreVecA = d1.getAlternative();
				Vector scoreVecI2 = d2.getInstance();
				Vector scoreVecA2 = d2.getAlternative();
				double score1 = bilinFunc(scoreVecI, scoreVecA);
				double score2 = bilinFunc(scoreVecI2, scoreVecA2);
				return score1 - score2 == 0 ? 0 : (score1 - score2 > 0 ? 1 : -1);
			}
		};
		return comparator;
	}
	/**
	 * A simple function that can be learned by a bilinear feature transform:
	 * <code>
	 * f((x_1,y_1) , (x_2, y_2)) = x1*y1 + x2*y2 + x1*y2 + x2*y1
	 * </code>
	 * @param scoreVec1 (x_1, y_1)
	 * @param scoreVec2 (x_2, y_2)
	 * @return
	 */
	private static final double bilinFunc(Vector scoreVec1, Vector scoreVec2) {
		double score = scoreVec1.getValue(0) * scoreVec2.getValue(0) + scoreVec1.getValue(1) * scoreVec2.getValue(1)
				+ scoreVec1.getValue(0) * scoreVec2.getValue(1) + scoreVec1.getValue(1) * scoreVec2.getValue(0);
		return Math.exp(score);
	}

	/**
	 * 
	 * @param maxLengthDyadRankingInstance
	 *            Maximum length of an individual dyad ranking instance in the
	 *            dataset
	 * @param size
	 *            Number of dyad ranking instances in the dataset
	 * @return A dyad ranking dataset with random dyads ((with 2 alternatives and 2
	 *         instances) that are ranked by the ranking function implemented by the
	 *         {@link Comparator} returned by {@link #complexDyadRanker()}
	 */
	public static DyadRankingDataset getDyadRankingDataset(int maxLengthDyadRankingInstance, int size, int seed) {
		DyadRankingDataset dataset = new DyadRankingDataset();
		for (int i = 0; i < size; i++) {
			dataset.add(DyadRankingInstanceSupplier.getDyadRankingInstance(maxLengthDyadRankingInstance, seed));
		}
		return dataset;
	}

}