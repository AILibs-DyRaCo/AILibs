package de.upb.crc901.mlplan.classifiers;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import de.upb.crc901.mlplan.core.MLPipeline;
import de.upb.crc901.mlplan.core.MLUtil;
import de.upb.crc901.mlplan.core.Preprocessor;
import de.upb.crc901.mlplan.core.SolutionEvaluator;
import de.upb.crc901.mlplan.search.evaluators.MonteCarloCrossValidationEvaluator;
import de.upb.crc901.mlplan.search.evaluators.RandomCompletionEvaluator;
import jaicore.ml.WekaUtil;
import jaicore.planning.graphgenerators.task.tfd.TFDNode;
import jaicore.search.algorithms.interfaces.IObservableORGraphSearch;
import jaicore.search.algorithms.parallel.parallelevaluation.local.bestfirst.ParallelizedBestFirst;
import jaicore.search.algorithms.parallel.parallelexploration.distributed.interfaces.SerializableGraphGenerator;
import weka.attributeSelection.ASSearch;
import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.PrincipalComponents;
import weka.attributeSelection.Ranker;
import weka.core.Instances;

public class ShrinkingPipelineOptimizer extends TwoPhasePipelineSearcher {
	private Preprocessor preprocessor;
	
	public ShrinkingPipelineOptimizer() throws IOException {
		this (MLUtil.getGraphGenerator(new File("testrsc/automl2.testset"), null), null, 0, 0, 20, 50, false);
	}
	
	public ShrinkingPipelineOptimizer(SerializableGraphGenerator<TFDNode, String> graphGenerator, Random random, int timeoutTotal, int timeoutPerNodeFComputation, int numberOfSolutions, int selectionDepth, boolean showGraph) {
		super(graphGenerator, random, timeoutTotal, timeoutPerNodeFComputation, numberOfSolutions, selectionDepth, showGraph);
	}
	
	@Override
	protected IObservableORGraphSearch<TFDNode, String, Integer> getSearch(Instances data) throws IOException {
		SolutionEvaluator solutionEvaluator = new MonteCarloCrossValidationEvaluator(5, 0.7f);
		RandomCompletionEvaluator rce = new RandomCompletionEvaluator(getRandom(), 3, solutionEvaluator);
		rce.setGenerator(getGraphGenerator());
		
		/* recude considered data if necessary */
		int originalNumberOfExamples = data.size();
		int maximumAcceptedExamples = data.numClasses() * 250;
		double relevantShare = maximumAcceptedExamples * 1f / data.size();
		if (originalNumberOfExamples > maximumAcceptedExamples) {
			data = WekaUtil.getStratifiedSplit(data, getRandom(), relevantShare).get(0);
			System.out.println("Reduced number of examples from " + originalNumberOfExamples + " to " + data.size());
		}
		else
			System.out.println(originalNumberOfExamples + "/" + maximumAcceptedExamples + " permitted examples. No reduction necessary");
		
		/* if there are too many features, reduce to at most */
		int specifiedFeatures = data.numAttributes();
		int maximumAcceptedFeatures = data.numClasses() * 5;
		if (specifiedFeatures > maximumAcceptedFeatures) {
			System.out.println(specifiedFeatures + "/" + maximumAcceptedFeatures + " permitted features. Reduction required.");
			PrincipalComponents pca = new PrincipalComponents();
			pca.setMaximumAttributeNames(maximumAcceptedFeatures / 2);
			ASSearch r = new Ranker();
			AttributeSelection as = new AttributeSelection();
			as.setEvaluator(pca);
			as.setSearch(r);
			preprocessor = new Preprocessor(r, pca, as);
			try {
				System.out.print("Applying " + pca.getClass().getName() + " ...");
				as.SelectAttributes(data);
				data = as.reduceDimensionality(data);
				System.out.println(" done");
			} catch (Exception e) {
				e.printStackTrace();
			}
			System.out.println("Reduced number of features from " + specifiedFeatures + " to " + data.numAttributes());
		}
		rce.setData(data);
		
		/* we allow CPUs-1 threads for node evaluation. Setting the timeout evaluator to null means to really prune all those */
		return new ParallelizedBestFirst<>(getGraphGenerator(), rce, getNumberOfCPUs() - 1, n -> null, getTimeoutPerNodeFComputation());
	}
	
	@Override
	protected MLPipeline modifyPipeline(MLPipeline mlp) {
		List<Preprocessor> preprocessors = new ArrayList<>(mlp.getPreprocessors());
		preprocessors.add(0, preprocessor);
		return new MLPipeline(null, preprocessors, mlp.getBaseClassifier());
	}
}

