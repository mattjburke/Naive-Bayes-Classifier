import java.io.File;
import java.io.IOException;

/**
 * Constructs a Naive Bayes classifier and outputs all of the relevant
 * statistics about the classifier to the console.
 * 
 * @author Matthew Burke
 */
public class NaiveBayesPrintout {

	public static void main(String[] args) throws IOException {
		File vocab = new File("./vocabulary.txt");
		File map = new File("./map.csv");
		File trainLabel = new File("./train_label.csv");
		File trainData = new File("./train_data.csv");
		File testLabel = new File("./test_label.csv");
		File testData = new File("./test_data.csv");

		long start = System.currentTimeMillis();

		NaiveBayes nb = new NaiveBayes(vocab, map, trainLabel, trainData, testLabel, testData);
		nb.printAll();

		double time = System.currentTimeMillis() - start;
		time /= 1000;
		System.out.println("Total execution time: " + time + " seconds.");
	}

}
