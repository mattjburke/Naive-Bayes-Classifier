import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Scanner;

/**
 * This class uses Naive Byes to classify news documents. It outputs a confusion
 * matrix for its performance on classifying each set of data.
 * 
 * The files input into the classifier are a processed version of the
 * 20Newsgroups data set. The 20 Newsgroups data set is a collection of
 * approximately 20,000 newsgroup documents, partitioned (nearly) evenly across
 * 20 different newsgroups. It was originally collected by Ken Lang, probably
 * for his "Newsweeder: Learning to filter netnews" paper, though he did not
 * explicitly mention this collection.
 * 
 * @author Matthew Burke
 */
public class NaiveBayes {
	private static final int NUM_CATS = 20;
	private File vocab;
	private File map;

	private double[] trainLabelArr;
	private double[] testLabelArr;

	private double[][] trainDataArr;
	public double[][] testDataArr;

	private double numVocab;

	private int[][] trainDocIDRanges;
	private int[][] testDocIDRanges;

	private int[][] trainDocRanges;
	private int[][] testDocRanges;

	private int[][] trainDataRanges;
	private int[][] testDataRanges;

	private double[] priors;

	private double[][] numOccurances;
	private double[] totNumWordsInCat;

	private double[][] postsBE;
	private double[][] postsMLE;

	private int[][] confusionMatrix;
	private double[] classAccs;

	/**
	 * Constructs a Naive Bayes classifier from the specified files.
	 * 
	 * @param vocab
	 *            A list of all of the words contained in the set of news documents.
	 *            The word ID is the integer corresponding a word's order in this
	 *            list.
	 * @param map
	 *            Indexes each category as an integer
	 * @param trainLabel
	 *            Each index in this file corresponds to a document ID in trainData,
	 *            and the value at each index is the category that the document
	 *            belongs in.
	 * @param trainData
	 *            A CSV file with each row formatted as Document ID, word ID, number
	 *            of occurances of the word in the document.
	 * @param testLabel
	 *            The same as trainLabel, but not used for training the model. Used
	 *            to evaluate the model.
	 * @param testData
	 *            The same as trainData, but not used for training the model. Used
	 *            to evaluate the model.
	 */
	public NaiveBayes(File vocab, File map, File trainLabel, File trainData, File testLabel, File testData)
			throws IOException {
		this.vocab = vocab;
		this.map = map;

		this.trainLabelArr = fileToArr(trainLabel);
		this.testLabelArr = fileToArr(testLabel);

		this.trainDataArr = fileTo2DArr(trainData);
		this.testDataArr = fileTo2DArr(testData);

		this.numVocab = getNumVocab();

		storeDocIDRange(trainLabelArr, trainDataArr, "train");
		storeDocIDRange(testLabelArr, testDataArr, "test");

		storeDocsInCatRange(trainLabelArr);
		storeDocsInCatRange(testLabelArr);

		storeDataRange(trainDataArr);
		storeDataRange(testDataArr);

		storePriors();
		storeAllNumOccurAndTotNums();
		storePosteriorsBE();
		storePosteriorsMLE();
	}

	public File getMap() {
		return map;
	}

	public double[] fileToArr(File file) throws IOException {
		try (BufferedReader br = new BufferedReader(new FileReader(file))) {
			return br.lines().mapToDouble(Double::parseDouble).toArray();
		}
	}

	public double[][] fileTo2DArr(File file) throws IOException {
		BufferedReader bufferedReader = new BufferedReader(new FileReader(file));
		ArrayList<Double> ints = new ArrayList<Double>();
		String line;
		while ((line = bufferedReader.readLine()) != null) {
			String[] lineArr = line.split(",");
			ints.add(Double.parseDouble(lineArr[0]));
			ints.add(Double.parseDouble(lineArr[1]));
			ints.add(Double.parseDouble(lineArr[2]));
		}
		bufferedReader.close();

		int rows = ints.size() / 3;
		double[][] ret = new double[rows][3];
		for (int i = 0; i < ints.size(); i++) {
			ret[i / 3][0] = ints.get(i);
			i++;
			ret[i / 3][1] = ints.get(i);
			i++;
			ret[i / 3][2] = ints.get(i);
		}
		return ret;
	}

	// ***********************************************************************************

	public double getNumLines(File file) throws IOException {
		Scanner scan = new Scanner(file);
		double numLines = 0;
		while (scan.hasNextLine()) {
			numLines++;
			scan.nextLine();
		}
		scan.close();
		return numLines;
	}

	public double getNumDocs(double[] labelArr) throws IOException {
		return labelArr.length;
	}

	public double getNumVocab() throws IOException {
		return getNumLines(vocab);
	}

	// get P(w_j)
	public double calcPrior(int category) throws IOException {
		int[] range = getDocsInCatRange(category, trainDataArr);
		return (range[1] - range[0] + 1) / getNumDocs(trainLabelArr);
	}

	public void storePriors() throws IOException {
		priors = new double[NUM_CATS];
		for (int i = 1; i <= NUM_CATS; i++) {
			priors[i - 1] = calcPrior(i);
		}
	}

	public double getPrior(int category) {
		return priors[category - 1];
	}

	// *****************************************************************************************
	// These methods store ranges in the data to avoid looping through the entire
	// data set more than is necessary, reducinf the execution time.

	/**
	 * Stores the range of document IDs that belong to each category by looping
	 * through the labels array once.
	 */
	public void storeDocsInCatRange(double[] labelsArr) throws FileNotFoundException {

		if (labelsArr == testLabelArr) {
			testDocRanges = new int[NUM_CATS][2];

			int first = 1;
			int last = 1;
			for (int i = 1; i < labelsArr.length; i++) {
				int cat = (int) labelsArr[i];

				if (i == labelsArr.length - 1) {
					last = i + 1;
					testDocRanges[cat - 1][0] = first;
					testDocRanges[cat - 1][1] = last;
				} else if (cat != labelsArr[i + 1]) {
					last = i;
					testDocRanges[cat - 1][0] = first;
					testDocRanges[cat - 1][1] = last;
					first = i + 1;
				}

			}
		} else if (labelsArr == trainLabelArr) {
			trainDocRanges = new int[NUM_CATS][2];

			int first = 1;
			int last = 1;
			for (int i = 1; i < labelsArr.length; i++) {
				int cat = (int) labelsArr[i];

				if (i == labelsArr.length - 1) {
					last = i + 1;
					trainDocRanges[cat - 1][0] = first;
					trainDocRanges[cat - 1][1] = last;
				} else if (cat != labelsArr[i + 1]) {
					last = i;
					trainDocRanges[cat - 1][0] = first;
					trainDocRanges[cat - 1][1] = last;
					first = i + 1;
				}
			}
		}

	}

	/** Returns the range of document IDs that belong to each category */
	public int[] getDocsInCatRange(int category, double[][] dataArr) {
		if (dataArr == testDataArr) {
			return testDocRanges[category - 1];
		} else {
			return trainDocRanges[category - 1];
		}
	}

	/**
	 * stores the range of rows that belong in the same category in 2d data array
	 */
	public void storeDataRange(double[][] dataArr) {
		if (dataArr == testDataArr) {
			testDataRanges = new int[NUM_CATS][2];
			int begin = 0;
			int cat = 1;
			int[] range = getDocsInCatRange(cat, dataArr);
			for (int i = 0; i < testDataArr.length && cat <= NUM_CATS; i++) {
				testDataRanges[cat - 1][0] = begin;

				// should only execute once per category
				if (testDataArr[i][0] > range[1] && testDataArr[i][0] != testDataArr[i - 1][0]
						&& i != testDataArr.length - 1) {
					testDataRanges[cat - 1][1] = i - 1;
					cat++;
					range = getDocsInCatRange(cat, dataArr);
					begin = i;
				}

				// last row never enters if statement since never > range[1]
				testDataRanges[NUM_CATS - 1][1] = i;
			}
		} else {
			trainDataRanges = new int[NUM_CATS][2];
			int begin = 0;
			int cat = 1;
			int[] range = getDocsInCatRange(cat, dataArr);
			for (int i = 0; i < trainDataArr.length && cat <= NUM_CATS; i++) {
				trainDataRanges[cat - 1][0] = begin;

				// should only execute once per category
				if (trainDataArr[i][0] > range[1] && trainDataArr[i][0] != trainDataArr[i - 1][0]
						&& i != trainDataArr.length - 1) {
					trainDataRanges[cat - 1][1] = i - 1;
					cat++;
					range = getDocsInCatRange(cat, dataArr);
					begin = i;
				}

				// last row never enters if statement since never > range[1]
				trainDataRanges[NUM_CATS - 1][1] = i;
			}
		}

	}

	public int[] getDataRange(int category, double[][] dataArr) {
		if (dataArr == testDataArr) {
			return testDataRanges[category - 1];
		} else {
			return trainDataRanges[category - 1];
		}
	}

	/** Stores the range in trainDataArr corresponding to one doccument */
	public void storeDocIDRange(double[] labelArr, double[][] dataArr, String testOrTrain) {
		if (testOrTrain == "test") {
			testDocIDRanges = new int[labelArr.length][2];

			int first = 0;
			int last = 0;
			for (int i = 0; i < dataArr.length; i++) {
				int docid = (int) dataArr[i][0];

				if (i == dataArr.length - 1) {
					last = i;
					testDocIDRanges[docid - 1][0] = first;
					testDocIDRanges[docid - 1][1] = last;
				} else if (docid != dataArr[i + 1][0]) {
					last = i;
					testDocIDRanges[docid - 1][0] = first;
					testDocIDRanges[docid - 1][1] = last;
					first = i + 1;
				}

			}

		} else if (testOrTrain == "train") {
			trainDocIDRanges = new int[labelArr.length][2];

			int first = 0;
			int last = 0;
			for (int i = 0; i < dataArr.length; i++) {
				int docid = (int) dataArr[i][0];

				if (i == dataArr.length - 1) {
					last = i;
					trainDocIDRanges[docid - 1][0] = first;
					trainDocIDRanges[docid - 1][1] = last;
				} else if (docid != dataArr[i + 1][0]) {
					last = i;
					trainDocIDRanges[docid - 1][0] = first;
					trainDocIDRanges[docid - 1][1] = last;
					first = i + 1;
				}

			}
		}

	}

	public int[] getDocIDRange(int doc, double[][] dataArr) {
		if (dataArr == trainDataArr) {
			return trainDocIDRanges[doc - 1];
		} else {
			return testDocIDRanges[doc - 1];
		}
	}

	// *********************************************************************************

	// get n
	public double totNumWords(int category) throws FileNotFoundException {
		return totNumWordsInCat[category - 1];
	}

	// get n_k
	public double numOccurances(double word, int category) throws FileNotFoundException {
		return numOccurances[(int) word - 1][category - 1];
	}

	public void storeNumOccurAndTotNum(int category) throws FileNotFoundException {
		int[] range = getDataRange(category, trainDataArr);
		for (int i = range[0]; i <= range[1]; i++) {
			int word = (int) trainDataArr[i][1];
			// add count of words in doc at index corresponding to word
			numOccurances[word - 1][category - 1] += trainDataArr[i][2];
			// add count to total number of words in the category
			totNumWordsInCat[category - 1] += trainDataArr[i][2];
		}

	}

	public void storeAllNumOccurAndTotNums() throws FileNotFoundException {
		numOccurances = new double[(int) numVocab][NUM_CATS];
		totNumWordsInCat = new double[NUM_CATS];
		for (int cat = 1; cat <= NUM_CATS; cat++) {
			storeNumOccurAndTotNum(cat);
		}

	}

	// *********************************************************************************
	// These methods use the math formulas for naive bayes

	public double PosteriorMLE(double word, int category) throws FileNotFoundException {
		double prob = numOccurances(word, category) / totNumWords(category);
		return prob;
	}

	public double PosteriorBE(double word, int category) throws FileNotFoundException {
		double prob = (numOccurances(word, category) + 1) / (totNumWords(category) + numVocab);
		return prob;
	}

	/** Stores all beyesian estimated posterior probabilities for faster access */
	public void storePosteriorsBE() throws FileNotFoundException {
		postsBE = new double[(int) numVocab][NUM_CATS];
		for (int w = 1; w <= numVocab; w++) {
			for (int c = 1; c <= NUM_CATS; c++) {
				postsBE[w - 1][c - 1] = PosteriorBE(w, c);
			}
		}
	}

	/**
	 * Stores all maximum likelihood estimated posterior probabilities for faster
	 * access
	 */
	public void storePosteriorsMLE() throws FileNotFoundException {
		postsMLE = new double[(int) numVocab][NUM_CATS];
		for (int w = 1; w <= numVocab; w++) {
			for (int c = 1; c <= NUM_CATS; c++) {
				postsMLE[w - 1][c - 1] = PosteriorMLE(w, c);
			}
		}
	}

	public double getPostBE(double word, int category) {
		return postsBE[(int) (word - 1)][category - 1];
	}

	public double getPostMLE(double word, int category) {
		return postsMLE[(int) (word - 1)][category - 1];
	}

	/** Returns the log sum of posterior probabilities of all words in a category */
	public double sumLogPosterior(String MLEorBE, int doc, int category, double[][] dataArr)
			throws FileNotFoundException {
		double sum = 0;
		// range needs to be for only the rows that contain the certain docID
		int[] range = getDocIDRange(doc, dataArr);
		for (int i = range[0]; i <= range[1]; i++) {
			double post;
			if (MLEorBE == "MLE") {
				// find the posterior probability of the word occurring in that category
				post = getPostMLE(dataArr[i][1], category);
			} else {
				post = getPostBE(dataArr[i][1], category);
			}
			// multiply by word count to find total probability of word occurring each time
			// in category
			sum += Math.log(post) * dataArr[i][2];
		}
		return sum;
	}

	public double probOfCat(String MLEorBE, int doc, int category, double[][] dataArr) throws IOException {
		double prior = getPrior(category);
		double logPost = sumLogPosterior(MLEorBE, doc, category, dataArr);
		return Math.log(prior) + logPost;
	}

	/** @return the most likely category the document bolongs in */
	public double classify(String MLEorBE, int doc, double[][] dataArr) throws IOException {
		double max = probOfCat(MLEorBE, doc, 1, dataArr);
		double maxCat = 1;
		for (int i = 2; i <= NUM_CATS; i++) {
			double probi = probOfCat(MLEorBE, doc, i, dataArr);
			if (max < probi) {
				max = probi;
				maxCat = i;
			}
		}

		return maxCat;
	}

	// ***************************************************************************************
	// These methods print out information and statistics about the classifier's
	// performance

	public void printPriors() throws IOException {
		System.out.println("Class Priors:");
		for (int i = 1; i <= NUM_CATS; i++) {
			System.out.println("P(Omega = " + i + ") = " + getPrior(i));
		}
	}

	/** Creates a confusion matrix */
	public void createConfusionMatrix(String MLEorBE, double[] labelsArr, double[][] dataArr) throws IOException {
		confusionMatrix = new int[NUM_CATS][NUM_CATS];
		for (int i = 0; i < labelsArr.length; i++) {
			// there is no docId == 0
			double predicted = classify(MLEorBE, (i + 1), dataArr);
			double actual = labelsArr[i];
			confusionMatrix[(int) actual - 1][(int) predicted - 1] += 1;
		}
		storeAccuracies(labelsArr);
	}

	public void storeAccuracies(double[] labelArr) throws IOException {
		classAccs = new double[NUM_CATS + 1];
		int totalTP = 0;
		double n = getNumDocs(labelArr);

		// gets the total number of TP documents for overall accuracy
		for (int i = 0; i < NUM_CATS; i++) {
			totalTP += confusionMatrix[i][i];
		}
		classAccs[0] = totalTP / n;

		for (int j = 1; j <= NUM_CATS; j++) {
			double acc = getAccuracy(j, labelArr);
			classAccs[j] = acc;
		}

	}

	/**
	 * Finds the class accuracy for a category using the true positive and true
	 * negative classification rates from the confusion matrix. Class accuracy = (TP
	 * + TN)/n
	 */
	public double getAccuracy(int category, double[] labelArr) throws IOException {
		double n = getNumDocs(labelArr);
		int tp = confusionMatrix[category - 1][category - 1];

		int rowTotal = 0;
		for (int k = 0; k < confusionMatrix.length; k++) {
			rowTotal += confusionMatrix[category - 1][k];
		}
		int fn = rowTotal - tp;

		int colTotal = 0;
		for (int i = 0; i < confusionMatrix.length; i++) {
			colTotal += confusionMatrix[i][category - 1];
		}
		int fp = colTotal - tp;

		double tn = n - tp - fn - fp;
		return (tp + tn) / n;
	}

	public void printAccuracies(String MLEorBE, double[] labelsArr, double[][] dataArr) throws IOException {
		createConfusionMatrix(MLEorBE, labelsArr, dataArr);
		System.out.println("Overall Accuracy = " + classAccs[0] * 100 + " % ");
		System.out.println("Class Accuracy:");
		for (int i = 1; i <= NUM_CATS; i++) {
			System.out.printf("Group %2d:  " + classAccs[i] * 100 + " %% \n", i);
		}
	}

	public void printConfusionMatrix() {
		String[] actual = " , , , , , , ,A,c,t,u,a,l, , , , , , , ".split(",");
		System.out.println("Confusion Matrix:");

		System.out.printf("%80s %n", "Predicted");
		System.out.printf("%6s", " ");
		for (int rowidx = 1; rowidx <= confusionMatrix.length; rowidx++) {
			System.out.printf("%7s", "[" + rowidx + "]");
		}
		System.out.println();
		System.out.println();
		for (int row = 0; row < confusionMatrix.length; row++) {
			System.out.printf("%1s", actual[row]);
			System.out.printf("%6s", "[" + (row + 1) + "]");
			for (int col = 0; col < confusionMatrix[0].length; col++) {
				System.out.printf("%6d ", confusionMatrix[row][col]);
			}
			System.out.println();
		}
	}

	public void printAll() throws IOException {
		System.out.println("This program will output the following relevant statistics: \n" + "Class priors. \n"
				+ "Performance on training data (using Bayesian estimators): overall accuracy, class accuracy, and confusion matrix. \n"
				+ "Performance on testing data (using Bayesian estimators): overall accuracy, class accuracy, and confusion matrix. \n"
				+ "Performance on testing data (using Maximum Likelihood estimators): overall accuracy, class accuracy, and confusion matrix.\n");

		System.out.println("\n**************************************************************************\n");
		printPriors();

		System.out.println("\n**************************************************************************\n");
		System.out.println("Performance on training data using Bayesian estimators:");
		printAccuracies("BE", trainLabelArr, trainDataArr);
		printConfusionMatrix();

		System.out.println("\n**************************************************************************\n");
		System.out.println("Performance on testing data using Bayesian estimators:");
		printAccuracies("BE", testLabelArr, testDataArr);
		printConfusionMatrix();

		System.out.println("\n**************************************************************************\n");
		System.out.println("Performance on testing data using Maximum Likelihood estimators:");
		printAccuracies("MLE", testLabelArr, testDataArr);
		printConfusionMatrix();
		System.out.println("\n**************************************************************************\n");

		System.out.println("Finished!");
	}

}
