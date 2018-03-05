package de.upb.crc901.mlplan.scc2018;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.aeonbits.owner.ConfigCache;

import de.upb.crc901.mlplan.classifiers.TwoPhaseHTNBasedPipelineSearcher;
import de.upb.crc901.mlplan.core.MLUtil;
import de.upb.crc901.mlplan.core.MySQLMLPlanExperimentLogger;
import de.upb.crc901.mlplan.search.evaluators.BalancedRandomCompletionEvaluator;
import de.upb.crc901.mlplan.search.evaluators.MonteCarloCrossValidationEvaluator;
import de.upb.crc901.mlplan.search.evaluators.MulticlassEvaluator;
import de.upb.crc901.services.core.HttpServiceServer;
import jaicore.graphvisualizer.SimpleGraphVisualizationWindow;
import jaicore.ml.experiments.MultiClassClassificationExperimentRunner;
import jaicore.ml.measures.PMMulticlass;
import jaicore.planning.graphgenerators.task.tfd.TFDNode;
import jaicore.planning.graphgenerators.task.tfd.TFDTooltipGenerator;
import jaicore.search.algorithms.standard.bestfirst.BestFirst;
import jaicore.search.algorithms.standard.core.ORGraphSearch;
import weka.classifiers.Classifier;

public class SCC2018MLTest extends MultiClassClassificationExperimentRunner {
	
	private static IPipelineEvaluationConf conf;
	private static int[] timeouts;
	private static int seeds;
	private static float trainingPortion;

	private static int numCPUs;
	private static int memoryInMB;
	
	static {
		conf = ConfigCache.getOrCreate(IPipelineEvaluationConf.class);
		timeouts = new int[] {conf.getTimeoutTotal()};
		seeds = conf.getNumberOfRuns();
		trainingPortion = (conf.getTrainingPortion() / 100f);
		numCPUs = conf.getNumberOfAllowedCPUs();
		memoryInMB = conf.getMemoryLimitinMB();
	}
	
	private final MySQLMLPlanExperimentLogger logger; // we want to have the logger, because we also send 
	
	protected static String[] getClassifierNames() {
		return new String[] { "MLS-Plan" };
	}

	protected static Map<String,String[]> getSetupNames() {
		Map<String,String[]> algoModes = new HashMap<>();
		algoModes.put("MLS-Plan", new String[] { conf.getValidationAlgorithm() });
		return algoModes;
	}
	
	public SCC2018MLTest(File datasetFolder) throws IOException {
		super(datasetFolder, getClassifierNames(), getSetupNames(), timeouts, seeds, trainingPortion, numCPUs, memoryInMB, PMMulticlass.errorRate, new MySQLMLPlanExperimentLogger("isys-db.cs.upb.de", "mlplan", "UMJXI4WlNqbS968X", "mlplan_results_test"));
		this.logger = (MySQLMLPlanExperimentLogger)getLogger();
	}

	@Override
	protected Classifier getConfiguredClassifier(int seed, String algoName, String algoMode, int timeoutInSeconds, int numberOfCPUs, int memoryInMB, PMMulticlass performanceMeasure) {
		try {
			switch (algoName) {
			case "MLS-Plan": {

				File evaluablePredicatFile = new File("testrsc/services/automl.evaluablepredicates");
				File searchSpaceFile = new File("testrsc/services/automl-services.searchspace");
				TwoPhaseHTNBasedPipelineSearcher<Double> bs = new TwoPhaseHTNBasedPipelineSearcher<>();
				
//				logicalDerivationTree(searchSpaceFile, evaluablePredicatFile);
				
				Random random = new Random(seed);
				bs.setHtnSearchSpaceFile(searchSpaceFile);
				//bs.setHtnSearchSpaceFile(new File("testrsc/automl3.testset"));
				bs.setEvaluablePredicateFile(evaluablePredicatFile);
				bs.setRandom(random);
				bs.setTimeout(1000 * timeoutInSeconds);
				bs.setNumberOfCPUs(numberOfCPUs);
				bs.setMemory(memoryInMB);
				MulticlassEvaluator baseEvaluator = new MulticlassEvaluator(random);
				Pattern p = Pattern.compile("([\\d]+)-([\\d]+)-MCCV");
				Matcher m = p.matcher(conf.getValidationAlgorithm());
				if (!m.find())
					throw new IllegalArgumentException("Cannot find validator " + conf.getValidationAlgorithm() + ". Need something that matches " + p.pattern());
				int mccvRepeats = Integer.valueOf(m.group(1));
				float mccvPortion = Integer.valueOf(m.group(2)) / 100f;
				
				bs.setSolutionEvaluatorFactory4Search(() -> new MonteCarloCrossValidationEvaluator(baseEvaluator, mccvRepeats, mccvPortion)); 
				bs.setSolutionEvaluatorFactory4Selection(() -> new MonteCarloCrossValidationEvaluator(baseEvaluator, mccvRepeats, mccvPortion));
				int numberOfSamples = conf.getNumberOfSamplesInFValueComputation();
				System.out.println("Samples: " + numberOfSamples);
				bs.setRce(new BalancedRandomCompletionEvaluator(random, numberOfSamples, new MonteCarloCrossValidationEvaluator(baseEvaluator, mccvRepeats, mccvPortion)));
//				bs.setTimeoutPerNodeFComputation(1000 * (timeoutInSeconds == 60 ? 15 : 300));
				bs.setTimeoutPerNodeFComputation(1000 * conf.getTimeoutPerCandidate());
//				bs.setTooltipGenerator(new TFDTooltipGenerator<>());
				bs.setPortionOfDataForPhase2(conf.getPortionOfDataForPhase2());
				
				bs.setExperimentLogger(logger);
				baseEvaluator.getMeasurementEventBus().register(logger);
				return bs;
			}
			}
		} catch (Throwable e) {
			e.printStackTrace();
		}
		return null;
	}

	private void logicalDerivationTree(File searchSpaceFile, File evaluablePredicatFile) throws IOException {

		 ORGraphSearch<TFDNode, String, Double> bf = new BestFirst<>(MLUtil.getGraphGenerator(searchSpaceFile, evaluablePredicatFile, null, null), n
		 -> 0.0);
		 new SimpleGraphVisualizationWindow<>(bf.getEventBus()).getPanel().setTooltipGenerator(new TFDTooltipGenerator<>());;
		
		 while(bf.nextSolution() != null);
	}
	
	public static void main(String[] args) throws Exception {
		HttpServiceServer server = HttpServiceServer.TEST_SERVER();
		try {
			
			File folder = new File(args[0]);
			SCC2018MLTest runner = new SCC2018MLTest(folder);
			runner.runAny();
		} finally {
			server.shutdown();
		}
	}
}
