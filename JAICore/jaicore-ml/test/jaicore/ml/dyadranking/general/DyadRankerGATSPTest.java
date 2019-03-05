package jaicore.ml.dyadranking.general;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.junit.Before;
import org.junit.Test;

import com.google.common.collect.Lists;

import de.upb.isys.linearalgebra.DenseDoubleVector;
import de.upb.isys.linearalgebra.Vector;
import jaicore.ml.core.dataset.IInstance;
import jaicore.ml.core.exception.PredictionException;
import jaicore.ml.core.exception.TrainingException;
import jaicore.ml.dyadranking.Dyad;
import jaicore.ml.dyadranking.algorithm.ADyadRanker;
import jaicore.ml.dyadranking.algorithm.PLNetDyadRanker;
import jaicore.ml.dyadranking.algorithm.featuretransform.FeatureTransformPLDyadRanker;
import jaicore.ml.dyadranking.dataset.DyadRankingDataset;
import jaicore.ml.dyadranking.dataset.DyadRankingInstance;
import jaicore.ml.dyadranking.dataset.IDyadRankingInstance;
import jaicore.ml.dyadranking.dataset.SparseDyadRankingInstance;
import jaicore.ml.dyadranking.loss.DyadRankingLossUtil;
import jaicore.ml.dyadranking.loss.KendallsTauDyadRankingLoss;

/**
 * This is a test based on Dirk Schäfers dyad ranking dataset based on
 * performance data of genetic algorithms on traveling salesman problem
 * instances https://github.com/disc5/ga-tsp-dataset which was used in [1] for
 * evaluation.
 * 
 * [1] Schäfer, D., & Hüllermeier, E. (2015). Dyad Ranking using a Bilinear
 * {P}lackett-{L}uce Model. In Proceedings ECML/PKDD--2015, European Conference
 * on Machine Learning and Knowledge Discovery in Databases (pp. 227–242).
 * Porto, Portugal: Springer.
 * 
 * @author Jonas Hanselle
 *
 */
public class DyadRankerGATSPTest {

	private static final String XXL_FILE = "testsrc/ml/dyadranking/ga-tsp/data_meta/GAMeta72-LR.txt";
	private static final String ALTERNATIVES_FEATURE_FILE = "testsrc/ml/dyadranking/ga-tsp/data_meta/GAMeta72-labeldescriptions.csv";
	private static final String ORDERINGS_FILE = "testsrc/ml/dyadranking/ga-tsp/data_meta/orderings.csv";

	// M = average ranking length
	private static final int M = 30;
	// N = number of training instances
	private static final int N = 90;
	// seed for shuffling the dataset
	private static final long seed = 15;

	ADyadRanker ranker;
	DyadRankingDataset dataset;

	public DyadRankerGATSPTest(ADyadRanker ranker) {
		this.ranker = ranker;
	}

	@Before
	public void init() {
		// load dataset
		dataset = loadDatasetFromXXLAndCSV(XXL_FILE, ALTERNATIVES_FEATURE_FILE);
		System.out.println(dataset);
		// TODO differenct rankers
		ranker = new PLNetDyadRanker();
	}

	@Test
	public void test() {

		System.out.println(dataset);
		
		Collections.shuffle(dataset, new Random(seed));

		// trim rankings
		dataset = randomlyTrimSparseDyadRankingInstances(dataset, M);
		
		// split data
		DyadRankingDataset trainData = new DyadRankingDataset(dataset.subList(0, N));
		DyadRankingDataset testData = new DyadRankingDataset(dataset.subList(N, dataset.size()));

		// trim dyad ranking instances for train data
		trainData = randomlyTrimSparseDyadRankingInstances(trainData, M);

		// standardize data
		DyadStandardScaler scaler = new DyadStandardScaler();
		scaler.fit(trainData);
		scaler.transformInstances(trainData);
		scaler.transformInstances(testData);
		
		try {

			// train the ranker
			ranker.train(trainData);

			double avgKendallTau = 0.0d;

			System.out.println();

			// compute average rank correlation
			for (int testIndex = 0; testIndex < testData.size(); testIndex++) {
				IDyadRankingInstance testInstance = (IDyadRankingInstance) testData.get(testIndex);
				List<Dyad> shuffleContainer = Lists.newArrayList(testInstance.iterator());
				shuffleContainer = Lists.reverse(shuffleContainer);
				IDyadRankingInstance shuffledInstance = new DyadRankingInstance(shuffleContainer);
				IDyadRankingInstance predictionInstance = (IDyadRankingInstance) ranker.predict(shuffledInstance);

//				System.out.println("Test instance");
//				for (Dyad dyad : testInstance) {
//					System.out.println(dyad.getAlternative());
//				}
//				System.out.println("\nPrediction ");
//				for (Dyad dyad : predictionInstance) {
//					System.out.println(dyad.getAlternative());
//				}
//				System.out.println("\n\n");

//				System.out.println("prediction: " + predictionInstance.toString());
//				System.out.println("test instance: " + testInstance.toString());
//				System.out.println();

				int dyadRankingLength = testInstance.length();
				int nConc = 0;
				int nDisc = 0;

				// check for all pairs of dyads in the test instance whether their order is the
				// same in the prediction, assumes that both DyadRankingInstances have the same
				// dyads and that these dyads are pairwise distinct
				for (int i = 1; i < dyadRankingLength; i++) {
					for (int j = 0; j < i; j++) {
						Dyad followingDyad = testInstance.getDyadAtPosition(i);
						Dyad leadingDyad = testInstance.getDyadAtPosition(j);
						int posOfFollowingInPrediction = positionOfDyad(predictionInstance, followingDyad);
						int posOfLeadingInPrediction = positionOfDyad(predictionInstance, leadingDyad);
						if (posOfFollowingInPrediction >= posOfLeadingInPrediction) {
							nConc++;
						} else {
							nDisc++;
						}
					}
				}
				double kendallTau = 2.0 * (nConc - nDisc) / (dyadRankingLength * (dyadRankingLength - 1));
				avgKendallTau += kendallTau;
			}

			avgKendallTau /= testData.size();

			System.out.println("Average Kendall's tau: " + avgKendallTau);

		} catch (TrainingException | PredictionException e) {
			e.printStackTrace();
		}

	}

	/**
	 * Loads the dataset from an xxl and a csv file.
	 * 
	 * @param filePathXXL                 xxl file containing the label ranking
	 * @param filePathAlternativeFeatures csv file containing the features for the
	 *                                    alternatives (labels)
	 * @return {@link DyadRankingDataset} constructed of the instances and
	 *         alternatives in the corresponding files
	 */
	private static DyadRankingDataset loadDatasetFromXXLAndCSV(String filePathXXL, String filePathAlternativeFeatures) {

		DyadRankingDataset dataset = new DyadRankingDataset();

		List<Vector> alternativeFeatures = new ArrayList<Vector>(100);

		// parse the file containing the features of the alternatives
		File alternativeFile = new File(filePathAlternativeFeatures);
		try {
			BufferedReader reader = new BufferedReader(new FileReader(alternativeFile));
			String line;
			while ((line = reader.readLine()) != null) {
				String[] tokens = line.split(",");
				DenseDoubleVector vector = new DenseDoubleVector(tokens.length);
				for (int i = 0; i < vector.length(); i++) {
					vector.setValue(i, Double.parseDouble(tokens[i]));
				}
				alternativeFeatures.add(vector);
			}
			reader.close();

		} catch (IOException e) {
			e.printStackTrace();
		}

		// parse XXL file
		File xxlFile = new File(filePathXXL);
		int numAttributes = 0;
		int numLabels = 0;
		try {
			BufferedReader reader = new BufferedReader(new FileReader(xxlFile));
			// read the first line and setup counters accordingly
			String line = reader.readLine();
			String[] tokens = line.split("\t");
			for (String token : tokens) {
				switch (token.charAt(0)) {
				case 'A':
					numAttributes++;
					break;
				case 'L':
					numLabels++;
					break;
				}
			}
			System.out.println("numAttributes: " + numAttributes);
			System.out.println("numLabels: " + numLabels);

			// skip two lines
			reader.readLine();
			reader.readLine();

			List<Vector> instanceFeatures = new ArrayList<Vector>(246);
			List<ArrayList<Vector>> alternativesList = new ArrayList<ArrayList<Vector>>(246);
			DescriptiveStatistics[] stats = new DescriptiveStatistics[numAttributes];
			for(int i = 0; i < stats.length; i++) {
				stats[i] = new DescriptiveStatistics();
			} 
			int lineIndex = 0;
			while ((line = reader.readLine()) != null) {
				tokens = line.split("\t");
				Vector instance = new DenseDoubleVector(numAttributes);
				ArrayList<Vector> alternatives = new ArrayList<Vector>(numLabels);

				// add the instances to the dyad ranking instance
				for (int i = 0; i < numAttributes; i++) {
					double val =  Double.parseDouble(tokens[i]);
					instance.setValue(i,val);
					stats[i].addValue(val);
				}

				// add the alternatives to the dyad ranking instance
				for (int i = numAttributes; i < tokens.length; i++) {
					int index = Integer.parseInt(tokens[i]) - 1;
					alternatives.add(alternativeFeatures.get(index));
				}
				instanceFeatures.add(instance);
				alternativesList.add(alternatives);
				
//				SparseDyadRankingInstance drInstance = new SparseDyadRankingInstance(instance, alternatives);
//				List<Dyad> dyadList = new LinkedList<Dyad>();
//				for(Dyad dyad : drInstance) {
//					dyadList.add(dyad);
//				}
//				DyadRankingInstance drDenseInstance = new DyadRankingInstance(dyadList);
//				System.out.println(dyadList);
//				dataset.add(drInstance);
			}
			
			double[] means = new double[numAttributes];
			for(int i = 0; i < means.length; i++) {
				means[i] = stats[i].getMean();
			}
			double[] stds = new double[numAttributes];
			for(int i = 0; i < stds.length; i++) {
				stds[i] = stats[i].getStandardDeviation();
			}
			Vector meanVec = new DenseDoubleVector(means);
			Vector stdVec = new DenseDoubleVector(stds);
			
			for(Vector instVec : instanceFeatures) {
				instVec.subtractVector(meanVec);
				instVec.divideByVectorPairwise(stds);
			}
			
			for(int i = 0; i < instanceFeatures.size(); i++) {
				dataset.add(new SparseDyadRankingInstance(instanceFeatures.get(i), alternativesList.get(i)));
			}
			
			reader.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		return dataset;
	}

	/**
	 * Trims the sparse dyad ranking instances by randomly selecting alternatives
	 * from each dyad ranking instance.
	 * 
	 * @param dataset
	 * @param dyadRankingLength the length of the trimmed dyad ranking instances
	 * @param seed
	 * @return
	 */
	private static DyadRankingDataset randomlyTrimSparseDyadRankingInstances(DyadRankingDataset dataset,
			int dyadRankingLength) {
		DyadRankingDataset trimmedDataset = new DyadRankingDataset();
		for (IInstance instance : dataset) {
			IDyadRankingInstance drInstance = (IDyadRankingInstance) instance;
			if (drInstance.length() < dyadRankingLength)
				continue;
			ArrayList<Boolean> flagVector = new ArrayList<Boolean>(drInstance.length());
			for (int i = 0; i < dyadRankingLength; i++) {
				flagVector.add(Boolean.TRUE);
			}
			for (int i = dyadRankingLength; i < drInstance.length(); i++) {
				flagVector.add(Boolean.FALSE);
			}
			Collections.shuffle(flagVector);
			List<Vector> trimmedAlternatives = new ArrayList<Vector>(dyadRankingLength);
			for (int i = 0; i < drInstance.length(); i++) {
				if (flagVector.get(i))
					trimmedAlternatives.add(drInstance.getDyadAtPosition(i).getAlternative());
			}
			SparseDyadRankingInstance trimmedDRInstance = new SparseDyadRankingInstance(
					drInstance.getDyadAtPosition(0).getInstance(), trimmedAlternatives);
			trimmedDataset.add(trimmedDRInstance);
			System.out.println("original: " + drInstance);
			System.out.println("trimmed: " + trimmedDRInstance + "\n\n");
		}
		return trimmedDataset;
	}

	/**
	 * Return the position of the first occurence of a {@link Dyad} in a
	 * {@link DyadRankingInstance}. Returns -1 if the dyad is not contained in the
	 * ranking.
	 * 
	 * @param drInstance
	 * @param dyad
	 * @return Position of the dyad in the ranking. -1 if it is not contained in the
	 *         ranking.
	 */
	private static int positionOfDyad(IDyadRankingInstance drInstance, Dyad dyad) {
		int pos = 0;
		while (!drInstance.getDyadAtPosition(pos).equals(dyad)) {
			if (pos >= drInstance.length())
				return -1;
			pos++;
		}
		return pos;
	}
	
}
