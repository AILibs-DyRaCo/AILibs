package jaicore.ml.dyadranking.activelearning;

import jaicore.ml.dyadranking.algorithm.PLNetDyadRanker;

/**
 * Abstract description of a pool-based active learning strategy for dyad
 * ranking.
 * 
 * @author Jonas Hanselle
 *
 */
public abstract class ActiveDyadRanker {

	protected PLNetDyadRanker ranker;
	protected IDyadRankingPoolProvider poolProvider;

	/**
	 * 
	 * @param ranker       The {@link PLNetDyadRanker} that is actively trained.
	 * @param poolProvider The {@link IDyadRankingPoolProvider} that provides a pool
	 *                     for pool-based selective sampling
	 */
	public ActiveDyadRanker(PLNetDyadRanker ranker, IDyadRankingPoolProvider poolProvider) {
		this.ranker = ranker;
		this.poolProvider = poolProvider;
	}

	/**
	 * Actively trains the ranker for a certain number of queries.
	 * 
	 * @param numberOfQueries Number of queries the ranker conducts
	 */
	public abstract void activelyTrain(int numberOfQueries);

	public PLNetDyadRanker getRanker() {
		return ranker;
	}

	public void setRanker(PLNetDyadRanker ranker) {
		this.ranker = ranker;
	}

	public IDyadRankingPoolProvider getPoolProvider() {
		return poolProvider;
	}

	public void setPoolProvider(IDyadRankingPoolProvider poolProvider) {
		this.poolProvider = poolProvider;
	}
}
