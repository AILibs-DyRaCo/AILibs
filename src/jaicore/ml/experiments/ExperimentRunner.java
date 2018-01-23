package jaicore.ml.experiments;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import jaicore.basic.MathExt;
import jaicore.ml.WekaUtil;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Instances;

public abstract class ExperimentRunner {

	private final File datasetFolder;
	
	public ExperimentRunner(File datasetFolder) {
		super();
		this.datasetFolder = datasetFolder;
	}
	
	protected abstract String[] getClassifierNames();
	protected abstract String[] getSetupNames();
	protected abstract int getNumberOfRunsPerExperiment();
	protected abstract float getTrainingPortion();
	protected abstract Classifier getConfiguredClassifier(int seed, String setupName, String algoName);
	

	public void run(int k) throws Exception {
		
		/* get classifiers */
		String[] classifiers = getClassifierNames();
		String[] setups = getSetupNames();
		
		/* read data sets */
		List<File> availableDatasets = getAvailableDatasets(datasetFolder);
		System.out.println("Available datasets: ");
		final AtomicInteger i = new AtomicInteger();
		availableDatasets.stream().forEach(ds -> System.out.println("\t" + (i.getAndIncrement()) + ": " + ds.getName()));
		System.out.println("Available algorithms: ");
		i.set(0);
		Arrays.asList(classifiers).stream().forEach(c -> System.out.println("\t" + (i.getAndIncrement()) + ": " + c.getClass().getName()));
		int numberOfDatasets = availableDatasets.size();
		int numberOfClassifiers = classifiers.length;
		int numberOfSetups = setups.length;
		int numberOfSeeds = getNumberOfRunsPerExperiment();
		System.out.println("Number of runs (seeds) per dataset/algo-combination: " + numberOfSeeds);
		int totalExperimentSize = numberOfClassifiers * numberOfDatasets * numberOfSetups * numberOfSeeds;
		int frameSizeForSeed = totalExperimentSize / numberOfSeeds;
		int frameSizeForSetup = frameSizeForSeed / numberOfSetups;
		int frameSizeForDataset = frameSizeForSetup / numberOfDatasets;
		
		/* determine exact experiment */
		int seedId = (int) Math.floor(k / frameSizeForSeed * 1f);
		int indexWithinSeed = k % frameSizeForSeed;
		int setupId = (int) Math.floor(indexWithinSeed / frameSizeForSetup * 1f);
		int indexWithinSetup = indexWithinSeed % frameSizeForSetup;
		int datasetId = (int) Math.floor(indexWithinSetup / frameSizeForDataset * 1f);
		int indexWithinDataset = indexWithinSetup % frameSizeForDataset;
		int algoId = indexWithinDataset % frameSizeForSeed;
		
		/* read dataset */
		String datasetName = getAvailableDatasets(datasetFolder).get(datasetId).getName();
		datasetName = datasetName.substring(0, datasetName.lastIndexOf("."));

		System.out.println("Running experiment " + k + "/" + totalExperimentSize + ". The setup is: " + seedId + "/" +  setupId + "/" + datasetId + "/" + algoId + "(seed/setup/dataset/algo)");
		
		/* create random object */
		Random r = new Random(seedId);
		
		/* create actual dataset */
		Instances data = getKthInstances(datasetFolder, datasetId);
		data.setClassIndex(data.numAttributes() - 1);
		List<Instances> overallSplit = WekaUtil.getStratifiedSplit(data, r, getTrainingPortion());
		Instances internalData = overallSplit.get(0);
		Instances testData = overallSplit.get(1);
		System.out.println("Data were split into " + internalData.size() + "/" + testData.size());
		
		/* create actual classifier */
		Classifier c = getConfiguredClassifier(seedId, setups[setupId],classifiers[algoId]);
		
		/* determine result file */
		File resultFolder = new File("results" + File.separator + setups[setupId]);
		if (!resultFolder.exists())
			resultFolder.mkdirs();
		File resultFile = new File(resultFolder + File.separator + c.getClass().getName() + "-" + datasetName + ".csv");
		
		/* now search for the best pipeline */
		long start = System.currentTimeMillis();
		System.out.println("Invoking " + getExperimentDescription(datasetFolder, datasetId, c, seedId) + " with setup " + setups[setupId] + "s");
		c.buildClassifier(internalData);
		long end = System.currentTimeMillis();
		System.out.println("Search has finished. Runtime: " + (end - start) / 1000f + " s");

		/* check performance of the pipeline */
		Evaluation eval = new Evaluation(internalData);
		eval.evaluateModel(c, testData);
		double error = MathExt.round((eval.pctIncorrect() + eval.pctUnclassified()) / 100f, 4);
		
		
		
		System.out.println("Wrote an error of " + error + " to stats file.");
	}
	
	public String getExperimentDescription(File folder, int datasetId, Classifier algorithm, int seed) {
		try {
			return algorithm.getClass().getName() + "-" + getAvailableDatasets(folder).get(datasetId).getName() + "-" + seed;
		} catch (IOException e) {
			e.printStackTrace();
		}
		return null;
	}

	public List<File> getAvailableDatasets(File folder) throws IOException {
		List<File> files = new ArrayList<>();
		try (Stream<Path> paths = Files.walk(folder.toPath())) {
			paths.filter(f -> f.getParent().toFile().equals(folder) && f.toFile().getAbsolutePath().endsWith(".arff")).forEach(f -> files.add(f.toFile()));
		}
		return files.stream().sorted().collect(Collectors.toList());
	}

	public Instances getKthInstances(File folder, int k) throws IOException {
		File f = getAvailableDatasets(folder).get(k);
		System.out.println("Selecting " + f);
		return new Instances(new BufferedReader(new FileReader(f)));
	}
}
