package jaicore.ml.dyadranking.dataset;

import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;

import jaicore.ml.core.dataset.IDataset;
import jaicore.ml.core.dataset.IInstance;
import jaicore.ml.core.dataset.attribute.IAttributeType;

/**
 * A dataset representation for dyad ranking. Contains
 * {@link IDyadRankingInstance}s.
 * 
 * @author Helena Graf
 *
 */
public class DyadRankingDataset extends ArrayList<IInstance> implements IDataset {

	private static final long serialVersionUID = -1102494546233523992L;

	@Override
	public <T> IAttributeType<T> getTargetType(Class<? extends T> clazz) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public List<IAttributeType<?>> getAttributeTypes() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public int getNumberOfAttributes() {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public void serialize(OutputStream out) {
		// TODO Auto-generated method stub

	}

	@Override
	public void deserialize(InputStream in) {
		// TODO Auto-generated method stub

	}

}
