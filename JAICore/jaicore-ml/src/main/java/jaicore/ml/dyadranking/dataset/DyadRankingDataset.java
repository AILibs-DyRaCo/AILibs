package jaicore.ml.dyadranking.dataset;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.LinkedList;
import java.util.List;

import org.apache.commons.io.IOUtils;
import org.apache.commons.io.LineIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import de.upb.isys.linearalgebra.DenseDoubleVector;
import de.upb.isys.linearalgebra.Vector;
import jaicore.ml.core.dataset.IDataset;
import jaicore.ml.core.dataset.IInstance;
import jaicore.ml.core.dataset.attribute.IAttributeType;
import jaicore.ml.dyadranking.Dyad;

/**
 * A dataset representation for dyad ranking. Contains
 * {@link IDyadRankingInstance}s. In particular, this dataset is just an
 * extension to the {@link ArrayList} implementation with typecasts to
 * {@link IDyadRankingInstance}.
 * 
 * @author Helena Graf, Mirko Jürgens, Michael Braun, Jonas Hanselle
 *
 */
public class DyadRankingDataset extends ArrayList<IInstance> implements IDataset<IInstance> {
	
	private Logger logger = LoggerFactory.getLogger(DyadRankingDataset.class);

	private static final long serialVersionUID = -1102494546233523992L;

	/**
	 * Creates an empty dyad ranking dataset.
	 */
	public DyadRankingDataset() {
		super();
	}

	/**
	 * Creates a dyad ranking dataset containing all elements in the given
	 * {@link Collection} in the order specified by the collections iterator.
	 * 
	 * @param c {@link Collection} containing {@link IInstance} objects
	 */
	public DyadRankingDataset(Collection<IInstance> c) {
		super(c);
	}

	/**
	 * Creates an empty dyad ranking dataset with the given initial capacity.
	 * 
	 * @param initialCapacity initial capacity of the dyad ranking dataset
	 */
	public DyadRankingDataset(int initialCapacity) {
		super(initialCapacity);
	}

	public DyadRankingDataset(List<IDyadRankingInstance> dyadRankingInstances) {
		super(dyadRankingInstances);
	}

	@Override
	public IAttributeType<?> getTargetType() {
		throw new UnsupportedOperationException("Dyad rankings have no target type.");
	}

	@Override
	public List<IAttributeType<?>> getAttributeTypes() {
		throw new UnsupportedOperationException();
	}

	@Override
	public int getNumberOfAttributes() {
		if (this.size() == 0) {
			return 0;
		}
		Dyad firstDyad = this.get(0).getDyadAtPosition(0);
		return firstDyad.getInstance().length() + firstDyad.getAlternative().length();
	}

	public void serialize(OutputStream out) {
		// currently, this always creates a dense dyad representation of the dyad
		// ranking dataset
		try {
			for (IInstance instance : this) {
				IDyadRankingInstance drInstance = (IDyadRankingInstance) instance;
				for (Dyad dyad : drInstance) {
					out.write(dyad.getInstance().toString().getBytes());
					out.write(";".getBytes());
					out.write(dyad.getAlternative().toString().getBytes());
					out.write("|".getBytes());
				}
				out.write("\n".getBytes());
			}
		} catch (IOException e) {
			logger.warn(e.getMessage());
		}
	}

	public void deserialize(InputStream in) {
		// currently, this always creates a dense dyad ranking dataset
		this.clear();
		try {
			LineIterator input = IOUtils.lineIterator(in, StandardCharsets.UTF_8);
			while(input.hasNext()) {
				String row = input.next();
			if (row.isEmpty())
					break;
				List<Dyad> dyads = new LinkedList<>();
				String[] dyadTokens = row.split("\\|");
				for (String dyadString : dyadTokens) {
					String[] values = dyadString.split(";");
					if (values[0].length() > 1 && values[1].length() > 1) {
						String[] instanceValues = values[0].substring(1, values[0].length() - 1).split(",");
						String[] alternativeValues = values[1].substring(1, values[1].length() - 1).split(",");
						Vector instance = new DenseDoubleVector(instanceValues.length);
						for (int i = 0; i < instanceValues.length; i++) {
							instance.setValue(i, Double.parseDouble(instanceValues[i]));
						}

						Vector alternative = new DenseDoubleVector(alternativeValues.length);
						for (int i = 0; i < alternativeValues.length; i++) {
							alternative.setValue(i, Double.parseDouble(alternativeValues[i]));
						}
						Dyad dyad = new Dyad(instance, alternative);
						dyads.add(dyad);
					}
				}
				this.add(new DyadRankingInstance(dyads));
			}
		} catch (IOException e) {
			logger.warn(e.getMessage());
		}
	}

	@Override
	public boolean equals(Object o) {
		if (!(o instanceof DyadRankingDataset)) {
			return false;
		}
		DyadRankingDataset dataset = (DyadRankingDataset) o;

		if (dataset.size() != this.size()) {
			return false;
		}

		for (int i = 0; i < dataset.size(); i++) {
			IDyadRankingInstance i1 = this.get(i);
			IDyadRankingInstance i2 = dataset.get(i);
			if (!i1.equals(i2)) {
				return false;
			}
		}
		return true;
	}
	
	@Override
	public int hashCode() {
		int result = 17;
		
		for (IInstance instance : this) {
			result = result * 31 + instance.hashCode();
		}
		
		return result;
	}

	@Override
	public IDyadRankingInstance get(int index) {
		return (IDyadRankingInstance) super.get(index);
	}

	@Override
	public <T> IAttributeType<T> getTargetType(Class<T> clazz) {
		throw new UnsupportedOperationException("Dyad rankings have no target type.");
	}
	
	/**
	 * Converts this data set to a list of ND4j {@link INDArray}s. 
	 * Each dyad ranking is represented by a 2D-matrix where a row is a dyad.
	 * @return
	 */
	public List<INDArray> toND4j() {
		List<INDArray> ndList = new ArrayList<>();
		for (IInstance instance : this) {
			IDyadRankingInstance drInstance = (IDyadRankingInstance) instance;
			ndList.add(dyadRankingToMatrix(drInstance));
		}
		return ndList;	
	}
	
	/**
	 * Converts a dyad to a {@link INDArray} row vector consisting of a
	 * concatenation of the instance and alternative features.
	 * 
	 * @param dyad The dyad to convert.
	 * @return The dyad in {@link INDArray} row vector form.
	 */
	private INDArray dyadToVector(Dyad dyad) {
		INDArray instanceOfDyad = Nd4j.create(dyad.getInstance().asArray());
		INDArray alternativeOfDyad = Nd4j.create(dyad.getAlternative().asArray());
		return Nd4j.hstack(instanceOfDyad, alternativeOfDyad);
	}
	
	public static DyadRankingDataset fromOrderedDyadList(List<Dyad> orderedDyad) {
		List<IDyadRankingInstance> dyadRankingInstance = Arrays.asList(new DyadRankingInstance(orderedDyad));
		return new DyadRankingDataset(dyadRankingInstance);
	}

	/**
	 * Converts a dyad ranking to a {@link INDArray} matrix where each row
	 * corresponds to a dyad.
	 * 
	 * @param drInstance The dyad ranking to convert to a matrix.
	 * @return The dyad ranking in {@link INDArray} matrix form.
	 */
	private INDArray dyadRankingToMatrix(IDyadRankingInstance drInstance) {
		List<INDArray> dyadList = new ArrayList<>(drInstance.length());
		for (Dyad dyad : drInstance) {
			INDArray dyadVector = dyadToVector(dyad);
			dyadList.add(dyadVector);
		}
		INDArray dyadMatrix;
		dyadMatrix = Nd4j.vstack(dyadList);
		return dyadMatrix;
	}
}
