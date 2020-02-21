import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.GainRatioAttributeEval;
import weka.attributeSelection.Ranker;
import weka.attributeSelection.ReliefFAttributeEval;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class GainRatio {

	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub
		
		System.out.print("Hello Long Dang");
		
		// Read the training ARFF file
				
		//String fileName = "/home/long/Desktop/Weka/data/weather.nominal.arff";
		String fileName = "Input/weather.nominal.arff";
		
		//Instances origData = new Instances(new BufferedReader(new FileReader(fileName)));
		
		DataSource source = new DataSource(fileName);
		
		Instances origData = source.getDataSet();
		
		if(origData.classIndex() == -1)
			//Make the last attribute be the class
			origData.setClassIndex(origData.numAttributes() - 1);
		
		int examples = origData.numInstances();
		
		//Print summary about the dataset
		
		//System.out.println(origData);
		
		System.out.println("\nNumber of examples " + examples);
		
		System.out.println("\nNumber of features " + origData.numAttributes());
		
		int size = origData.numAttributes() - 1;
		//return the top 10 ReliefF merit score features
		double[][] selectedFeatures = new double[size][2];
				
		selectedFeatures =	RankFeatures_GainRatio(origData, 5);
				
				for(int i = 0; i < size; i++) {
					// return the attributes's score using ReliefF's instance based approach
					int index = (int) selectedFeatures[i][0];
					System.out.printf("Gain Ratio - %.4f %d %s \n", selectedFeatures[i][1], (index + 1), origData.attribute(index).name());
				}
		
		

	}
	
	protected static double[][] RankFeatures_GainRatio(Instances trainData, int N1) throws Exception {
		
		int N = trainData.numAttributes();
		
		double[][] rankedAtt = new double[N1][2];
		
		AttributeSelection attsel = new AttributeSelection();
		
		// *******************Step1*********************************
				// Step1: Selecting Relevant Features: Uses Gain ratio filter to get feature
				// rankings and then select top N1 features
		
		// a. First get RelieF rankings
		GainRatioAttributeEval eval = new GainRatioAttributeEval();
		
		//Should we specify the number of features to retain?
		Ranker search = new Ranker();
		
		//String[] ReliefFOptions = {"-M", "-1", "-D", "1", "-K", "10"};
		
		//search.setOptions(ReliefFOptions);
		
		attsel.setEvaluator(eval);
		
		attsel.setSearch(search);
		
		attsel.SelectAttributes(trainData);
		
		rankedAtt = attsel.rankedAttributes();
		
		return rankedAtt;
	}

}
