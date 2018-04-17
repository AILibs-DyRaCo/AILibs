package de.upb.crc901.mlplan.scc2018;

import de.upb.crc901.mlplan.classifiers.TwoPhaseHTNBasedPipelineSearcher;
import de.upb.crc901.mlplan.core.DummyMLPlanExperimentLogger;
import de.upb.crc901.mlplan.core.MLUtil;
import de.upb.crc901.mlplan.search.evaluators.BalancedRandomCompletionEvaluator;
import de.upb.crc901.mlplan.search.evaluators.MonteCarloCrossValidationEvaluator;
import de.upb.crc901.mlplan.search.evaluators.MulticlassEvaluator;
import de.upb.crc901.mlplan.services.MLPipelinePlan;
import de.upb.crc901.services.core.HttpServiceClient;
import de.upb.crc901.services.core.HttpServiceServer;

import jaicore.graphvisualizer.SimpleGraphVisualizationWindow;
import jaicore.ml.experiments.IMultiClassClassificationExperimentDatabase;
import jaicore.ml.experiments.MultiClassClassificationExperimentRunner;
import jaicore.ml.measures.PMMulticlass;
import jaicore.planning.graphgenerators.task.tfd.TFDNode;
import jaicore.planning.graphgenerators.task.tfd.TFDTooltipGenerator;
import jaicore.search.algorithms.standard.bestfirst.BestFirst;
import jaicore.search.algorithms.standard.core.ORGraphSearch;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.aeonbits.owner.ConfigCache;

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
    timeouts = new int[] { conf.getTimeoutTotal() };
    seeds = conf.getNumberOfRuns();
    trainingPortion = (conf.getTrainingPortion() / 100f);
    numCPUs = conf.getNumberOfAllowedCPUs();
    memoryInMB = conf.getMemoryLimitinMB();
  }

  private final IMultiClassClassificationExperimentDatabase logger; // we want to have the logger, because we also send

  protected static String[] getClassifierNames() {
    return new String[] { "MLS-Plan" };
  }

  protected static Map<String, String[]> getSetupNames() {
    Map<String, String[]> algoModes = new HashMap<>();
    algoModes.put("MLS-Plan", new String[] { conf.getValidationAlgorithm() });
    return algoModes;
  }

  public SCC2018MLTest(final File datasetFolder, final String hostPase, final String hostJase) throws IOException {
    // super(datasetFolder, getClassifierNames(), getSetupNames(), timeouts, seeds, trainingPortion,
    // numCPUs, memoryInMB, PMMulticlass.errorRate, new MySQLMLPlanExperimentLogger(conf.getDBHost(),
    // conf.getDBUsername(), conf.getDBPassword(), conf.getDBDatabaseName()));
    // this.logger = (MySQLMLPlanExperimentLogger)getLogger();
    super(datasetFolder, getClassifierNames(), getSetupNames(), timeouts, seeds, trainingPortion, numCPUs, memoryInMB, PMMulticlass.errorRate, new DummyMLPlanExperimentLogger());
    this.logger = this.getLogger();
    MLPipelinePlan.hostPASE = hostPase;
    MLPipelinePlan.hostJASE = hostJase;
    String[] jaseParts = hostJase.split(":");
    HttpServiceClient.hostForCancelation = jaseParts[0] + ":" + (Integer.parseInt(jaseParts[1]) + 1000);
  }

  @Override
  protected Classifier getConfiguredClassifier(final int seed, final String algoName, final String algoMode, final int timeoutInSeconds, final int numberOfCPUs,
      final int memoryInMB, final PMMulticlass performanceMeasure) {
    try {
      switch (algoName) {
        case "MLS-Plan": {

          File evaluablePredicatFile = new File("testrsc/services/automl.evaluablepredicates");
          File searchSpaceFile = new File("testrsc/services/automl-services.searchspace");
          TwoPhaseHTNBasedPipelineSearcher<Double> bs = new TwoPhaseHTNBasedPipelineSearcher<>();

          Random random = new Random(seed);
          bs.setHtnSearchSpaceFile(searchSpaceFile);
          // bs.setHtnSearchSpaceFile(new File("testrsc/automl3.testset"));
          bs.setEvaluablePredicateFile(evaluablePredicatFile);
          bs.setRandom(random);
          bs.setTimeout(1000 * timeoutInSeconds);
          bs.setNumberOfCPUs(numberOfCPUs);
          bs.setMemory(memoryInMB);
          MulticlassEvaluator baseEvaluator = new MulticlassEvaluator(random);
          Pattern p = Pattern.compile("([\\d]+)-([\\d]+)-MCCV");
          Matcher m = p.matcher(conf.getValidationAlgorithm());
          if (!m.find()) {
            throw new IllegalArgumentException("Cannot find validator " + conf.getValidationAlgorithm() + ". Need something that matches " + p.pattern());
          }
          int mccvRepeats = Integer.valueOf(m.group(1));
          float mccvPortion = Integer.valueOf(m.group(2)) / 100f;

          bs.setSolutionEvaluatorFactory4Search(() -> new MonteCarloCrossValidationEvaluator(baseEvaluator, mccvRepeats, mccvPortion));
          bs.setSolutionEvaluatorFactory4Selection(() -> new MonteCarloCrossValidationEvaluator(baseEvaluator, mccvRepeats, mccvPortion * .85f));
          int numberOfSamples = conf.getNumberOfSamplesInFValueComputation();
          System.out.println("Samples: " + numberOfSamples);
          bs.setRce(new BalancedRandomCompletionEvaluator(random, numberOfSamples, new MonteCarloCrossValidationEvaluator(baseEvaluator, mccvRepeats, mccvPortion)));
          // bs.setTimeoutPerNodeFComputation(1000 * (timeoutInSeconds == 60 ? 15 : 300));
          bs.setTimeoutPerNodeFComputation(1000 * conf.getTimeoutPerCandidate());
          bs.setTooltipGenerator(new TFDTooltipGenerator<>());
          bs.setNumberOfMCIterationsPerSolutionInSelectionPhase(conf.getNumberOfIterationsInSelectionPhase());
          bs.setPortionOfDataForPhase2(conf.getPortionOfDataForPhase2());
          // bs.setExperimentLogger(logger);
          baseEvaluator.getMeasurementEventBus().register(this.logger);
          return bs;
        }
      }
    } catch (Throwable e) {
      e.printStackTrace();
    }
    return null;
  }

  private void logicalDerivationTree(final File searchSpaceFile, final File evaluablePredicatFile) throws IOException {

    ORGraphSearch<TFDNode, String, Double> bf = new BestFirst<>(MLUtil.getGraphGenerator(searchSpaceFile, evaluablePredicatFile, null, null), n -> 0.0);
    new SimpleGraphVisualizationWindow<>(bf.getEventBus()).getPanel().setTooltipGenerator(new TFDTooltipGenerator<>());
    ;

    while (bf.nextSolution() != null) {
      ;
    }
  }

  public static void main(final String[] args) throws Exception {
    File folder = new File(args[0]);
    String hostPase = args[1];
    String hostJase = args[2];

    /* start JASE server */
    HttpServiceServer server = null;
    if (hostJase.split(":")[0].equals("localhost")) {
      System.out.println("Host for JASE: " + hostJase);
      server = new HttpServiceServer(Integer.parseInt(hostJase.split(":")[1]), "testrsc/conf/classifiers.json", "testrsc/conf/preprocessors.json", "testrsc/conf/others.json");
    } else {
    }
    try {
      SCC2018MLTest runner = new SCC2018MLTest(folder, hostPase, hostJase);
      runner.runSpecific(0);
    } finally {
      if (server != null) {
        server.shutdown();
      }
    }
  }
}
