package de.upb.crc901.mlplan.metamining.activelearning;

import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Random;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

import de.upb.isys.linearalgebra.DenseDoubleVector;
import de.upb.isys.linearalgebra.Vector;
import jaicore.basic.SQLAdapter;
import jaicore.basic.sets.SetUtil.Pair;
import jaicore.ml.activelearning.IActiveLearningPoolProvider;
import jaicore.ml.core.dataset.IInstance;
import jaicore.ml.dyadranking.Dyad;
import jaicore.ml.dyadranking.dataset.DyadRankingDataset;
import jaicore.ml.dyadranking.dataset.IDyadRankingInstance;
import jaicore.ml.dyadranking.dataset.SparseDyadRankingInstance;
import jaicore.ml.dyadranking.util.DyadStandardScaler;

/**
 * An {@link IActiveLearningPoolProvider} that generates
 * {@link IDyadRankingInstance}s from AutoML metamining data in a SQL database.
 * 
 * @author Jonas Hanselle, Mirko Jürgens
 *
 */
public class SQLMetafeatureDyadPoolProvider implements IActiveLearningPoolProvider {

	private enum X_METRIC {
		X_LANDMARKERS("X_LANDMARKERS"), X_OTHERS("X_OTHERS");

		X_METRIC(String dbColumn) {
			this.dbColumn = dbColumn;
		}

		protected String dbColumn;

		@Override
		public String toString() {
			return dbColumn;
		}
	}

	private final int DATASET_COUNT = 3421;

	private final int[] allowedDatasetIds = { 12, 14, 16, 18, 20, 21, 22, 23, 24, 26, 28, 3, 30, 32, 36, 38, 44, 46, 5,
			6 };

	private final String dyadTable = "dyad_dataset_new";

	private final String datasetMetaFeatureTable = "dataset_metafeatures_mirror";
	private final Pattern arrayDeserializer = Pattern.compile(" ");
	
	private SQLAdapter adapter;
	
	/**
	 * this {@link HashMap} contains a {@link SparseDyadRankingInstance} for each dataset.
	 */
	private HashMap<Vector, SparseDyadRankingInstance> sampleSpace;
	
	
	private X_METRIC xMetric = X_METRIC.X_LANDMARKERS;

	public SQLMetafeatureDyadPoolProvider(String dbHost, String dbUser, String dbPassword, String dbName) {
		adapter = new SQLAdapter(dbHost, dbUser, dbPassword, dbName);
		DyadRankingDataset trainDataset = new DyadRankingDataset();
		try {
			trainDataset = getSparseDyadDataset(42, 400, 10, adapter);
		} catch (SQLException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		DyadStandardScaler scaler = new DyadStandardScaler();
		scaler.fitTransform(trainDataset);
	}

	/**
	 * Queries the DB to extract the dyad and its' perfomance score.
	 * 
	 * @param id specifies the db entry
	 * @return the dyad
	 * @throws SQLException
	 */
	private Pair<Dyad, Double> getDyadAndScoreWithId(int id, SQLAdapter adapter) throws SQLException {
		ResultSet res = adapter.getResultsOfQuery("SELECT " + xMetric + ", score FROM " + dyadTable + " NATURAL JOIN "
				+ datasetMetaFeatureTable + " WHERE id=" + id);
		if (res.wasNull())
			throw new IllegalArgumentException("No entry with id " + id);

		res.first();

		ResultSet res_y = adapter.getResultsOfQuery("SELECT y FROM " + dyadTable + " WHERE id=" + id);
		res_y.first();

		String serializedY = res_y.getString(1);
		String serializedX = res.getString(1);
		Double score = res.getDouble(2);

		double[] xArray = arrayDeserializer.splitAsStream(serializedX).mapToDouble(Double::parseDouble).toArray();
		double[] yArray = arrayDeserializer.splitAsStream(serializedY).mapToDouble(Double::parseDouble).toArray();

		Dyad dyad = new Dyad(new DenseDoubleVector(xArray), new DenseDoubleVector(yArray));
		return new Pair<Dyad, Double>(dyad, score);
	}

	/**
	 * Generates a {@link SparseDyadRankingInstance} in the following manner: <code>
	 * while there aren't enough dyads
	 *   collect all dyads with the specified dataset id
	 *   draw a random dyad using the seed
	 * sort the dyads
	 * return the sparse instance
	 * </code>
	 * 
	 * @return
	 * @throws SQLException
	 */
	private SparseDyadRankingInstance getSparseDyadInstance(int datasetId, int seed, int length, SQLAdapter adapter)
			throws SQLException {
		// get all indices that have the correct dataset id
		// count the datasets
		ResultSet res = adapter
				.getResultsOfQuery("SELECT COUNT(id) FROM " + dyadTable + " WHERE dataset = " + datasetId);
		res.first();
		int indicesAmount = res.getInt(1);
		if (indicesAmount == 0)
			throw new IllegalArgumentException("No performance samples for for the dataset-id: " + datasetId);
		int[] dyadIndicesWithDataset = new int[indicesAmount];
		// collect the indices
		res = adapter.getResultsOfQuery("SELECT id FROM " + dyadTable + " WHERE dataset = " + datasetId);
		int counter = 0;
		while (res.next()) {
			dyadIndicesWithDataset[counter++] = res.getInt(1);
		}

		// now draw the dyads
		List<Pair<Dyad, Double>> dyads = new ArrayList<>(length);
		Random random = new Random(seed);

		for (int i = 0; i < length; i++) {
			int randomIndexOfArray = random.nextInt(indicesAmount);
			int randomIndexInDb = dyadIndicesWithDataset[randomIndexOfArray];
			dyads.add(getDyadAndScoreWithId(randomIndexInDb, adapter));
		}

		// sort the dyads and extract the sparse instance
		Vector singleX = dyads.iterator().next().getX().getInstance();
		List<Vector> sortedAlternatives = dyads.stream()
				.sorted((pair1, pair2) -> Double.compare(pair1.getY(), pair2.getY())).map(Pair::getX)
				.map(Dyad::getAlternative).collect(Collectors.toList());
		return new SparseDyadRankingInstance(singleX, sortedAlternatives);
	}

	public DyadRankingDataset getSparseDyadDataset(int seed, int amountOfDyadInstances, int alternativeLength,
			SQLAdapter adapter) throws SQLException {

		List<IDyadRankingInstance> sparseDyadRankingInstances = new ArrayList<>();
		for (int i = 0; i < amountOfDyadInstances; i++) {
			int intermediateSeed = seed + i;
			int randomDataset = getRandomDatasetId(intermediateSeed);
			sparseDyadRankingInstances
					.add(getSparseDyadInstance(randomDataset, intermediateSeed, alternativeLength, adapter));
		}
		return new DyadRankingDataset(sparseDyadRankingInstances);
	}

	private int getRandomDatasetId(int intermediateSeed) {
		Random random = new Random(intermediateSeed);
		int index = random.nextInt(allowedDatasetIds.length);
		return allowedDatasetIds[index];
	}

	@Override
	public Collection<IInstance> getPool() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public IInstance query(IInstance queryInstance) {
		// TODO Auto-generated method stub
		return null;
	}

}
