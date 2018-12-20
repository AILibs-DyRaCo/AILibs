package mlplan.dyadrankingevaluation.performancesamplegenerator;

import java.io.File;
import java.util.concurrent.ThreadLocalRandom;

import org.aeonbits.owner.ConfigFactory;

import de.upb.crc901.mlpipeline_evaluation.CacheEvaluatorMeasureBridge;
import de.upb.crc901.mlpipeline_evaluation.ConsistentMLPipelineEvaluator;
import de.upb.crc901.mlpipeline_evaluation.PerformanceDBAdapter;
import de.upb.crc901.mlplan.multiclass.wekamlplan.MLPlanWekaBuilder;
import de.upb.crc901.mlplan.multiclass.wekamlplan.MLPlanWekaClassifier;
import de.upb.crc901.mlplan.multiclass.wekamlplan.weka.WEKAPipelineFactory;
import de.upb.crc901.mlplan.multiclass.wekamlplan.weka.WekaMLPlanWekaClassifier;
import de.upb.crc901.mlplan.multiclass.wekamlplan.weka.model.MLPipeline;
import hasco.core.Util;
import hasco.model.ComponentInstance;
import jaicore.basic.SQLAdapter;
import jaicore.ml.cache.ReproducibleInstances;
import jaicore.ml.core.evaluation.measure.singlelabel.MultiClassPerformanceMeasure;
import jaicore.ml.core.evaluation.measure.singlelabel.ZeroOneLoss;
import jaicore.ml.evaluation.evaluators.weka.MonteCarloCrossValidationEvaluator;
import jaicore.planning.graphgenerators.task.tfd.TFDNode;
import jaicore.search.algorithms.standard.bestfirst.nodeevaluation.INodeEvaluator;
import jaicore.search.algorithms.standard.random.RandomSearch;
import jaicore.search.core.interfaces.GraphGenerator;
import jaicore.search.model.other.SearchGraphPath;
import jaicore.search.model.probleminputs.GraphSearchInput;
import jaicore.search.model.travesaltree.Node;

/**
 * Simple class that can be used to generate performance samples from the
 * ML-Plan search space for datasets from OpenML. Therefore a
 * {@link RandomSearch} is applied to the search space graph return by ML-Plan.
 * 
 * @author Jonas Hanselle
 *
 */
public class PerformanceSampleGenerator {

	private IPerformanceSampleGeneratorConfig config;

	public PerformanceSampleGenerator() {
		this.config = ConfigFactory.create(IPerformanceSampleGeneratorConfig.class);
	}

	public void evaluate(String openmlID) {

		try {
			SQLAdapter adapter = new SQLAdapter(config.getDBHost(), config.getDBUsername(), config.getDBPassword(),
					config.getDBDatabaseName());
			PerformanceDBAdapter performanceDBAdapter = new PerformanceDBAdapter(adapter, config.getDBTableName());

			ZeroOneLoss lossFunction = new ZeroOneLoss();
			CacheEvaluatorMeasureBridge bridge = new CacheEvaluatorMeasureBridge(lossFunction, performanceDBAdapter);
			MLPlanWekaClassifier mlplan = new WekaMLPlanWekaClassifier();
			ReproducibleInstances data = ReproducibleInstances.fromOpenML(openmlID, config.getOpenMLKey());
			data.setCacheLookup(true);
			data.setCacheStorage(true);
			data.setClassIndex(data.numAttributes() - 1);
			mlplan.setLoggerName("mlplan");
			mlplan.setData(data);
			GraphGenerator gg = mlplan.getGraphGenerator();
			GraphSearchInput gsi = new GraphSearchInput(gg);
			RandomSearch rs = new RandomSearch(gsi, config.getRandomSearchSeed());
			while (rs.hasNext()) {
				SearchGraphPath sgp = rs.nextSolution();
				TFDNode goalNode = (TFDNode) sgp.getNodes().get(sgp.getNodes().size() - 1);
				WEKAPipelineFactory factory = new WEKAPipelineFactory();
				ComponentInstance ci = Util.getSolutionCompositionFromState(mlplan.getComponents(), goalNode.getState(),
						true);
				try {
					MLPipeline mlp = factory.getComponentInstantiation(ci);
					CacheEvaluatorMeasureBridge shallowBridge = ((CacheEvaluatorMeasureBridge) bridge)
							.getShallowCopy(ci);
					MonteCarloCrossValidationEvaluator evaluator = new MonteCarloCrossValidationEvaluator(shallowBridge,
							config.getNumberMCCVRepeats(), data, config.getMCCVTrainRatio(), config.getMCCVSeed());
					double score = evaluator.evaluate(mlp);
					System.out.println(mlp);
					System.out.println("score: " + score);
				} catch (Exception e) {
					e.printStackTrace();
				}
			}
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	public static void main(String args[]) {
		String openmlID = args[0];
		PerformanceSampleGenerator psg = new PerformanceSampleGenerator();
		psg.evaluate(openmlID);
	}
}
