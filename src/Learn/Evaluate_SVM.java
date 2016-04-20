package Learn;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Random;

import de.bwaldvogel.liblinear.Feature;
import de.bwaldvogel.liblinear.FeatureNode;
import de.bwaldvogel.liblinear.Linear;
import de.bwaldvogel.liblinear.Model;
import de.bwaldvogel.liblinear.Parameter;
import de.bwaldvogel.liblinear.Problem;
import de.bwaldvogel.liblinear.SolverType;

public class Evaluate_SVM {
	int dataNums;
	int dimensions;
	double Ws[][];
	int labels[];
    public static double C = 10;    // cost of constraints violation
    double eps = 0.1; // stopping criteria
	int test;
	int train;
	double limitRandom;
	Feature vectrain[][];
	Feature vectest[][];
	double trainattr[];
	double testattr[];
	int TestTrain[];
	File modelFile;
	String readLabelFiles="";
	/**
	 * Make a new SVM model.
	 * @param readLabelFile The category file of different source
	 * @param limitRandom The percentage of train set
	 * @param TestTrain The array to differentiate train and test set
	 * 
	 * */
	public Evaluate_SVM(int dataNum, int dimension, double W[][], String readLabelFile, double limitRandom, int[]TestTrain) throws Exception{
		this.dataNums=dataNum;
		this.dimensions=dimension;
		this.readLabelFiles=readLabelFile;
		this.limitRandom=limitRandom;
		this.TestTrain=TestTrain;
		//******
		this.countTestTrain();
		vectrain=new Feature[train][dimensions];
		vectest	=new Feature[test][dimensions];
		trainattr=new double[train];
		testattr =new double[test];
		//******
		this.Ws=W;
		labels=new int[dataNums];
		this.makeSource();
	}
	/**
	 * To count the number of users in two sets.
	 * 
	 */
	public void countTestTrain(){
		int countTest=0,countTrain=0;
		for(int i=0;i<dataNums;i++){
			if(TestTrain[i]==1){
				countTrain++;
			}
			else{
				countTest++;
			}
		}
		this.test=countTest;
		this.train=countTrain;
		System.out.println("test + train = "+(test+train));
	}
	/**
	 * To make train and test set to train SVM model
	 * 
	 * */
	public void makeSource() throws Exception{
		this.readLabel(readLabelFiles);
		this.prepTrainData();
		this.prepTestData();
		}
	/**
	 * Read in category file of different source.
	 * 
	 * */
	public void readLabel(String readLabelFile) throws Exception{
		BufferedReader trUsers = new BufferedReader(new InputStreamReader(new FileInputStream(readLabelFile)));
		String line = "";
		while((line = trUsers.readLine())!=null){
				String[] strs = line.split("\t");
				labels[Integer.valueOf(strs[0])]=Integer.valueOf(strs[1]);
		}
		trUsers.close();
	}
	
	public void prepTrainData() throws Exception{
		int index=0;
		boolean ss=true;
		for(int i=0;i<dataNums;i++){
			if(TestTrain[i]==0)
				continue;
			else{
			for(int j=0;j<dimensions;j++){
				this.vectrain[index][j]=new FeatureNode(j+1,Ws[i][j]);
			}
			this.trainattr[index]=labels[i];
			index++;
			}
		}
	}
	public void prepTestData(){
		int index=0;
		for(int i=0;i<dataNums;i++){
			if(TestTrain[i]==1)
				continue;
			else{
			for(int j=0;j<dimensions;j++){
				this.vectest[index][j]=new FeatureNode(j+1,Ws[i][j]);
			}
			this.testattr[index]=labels[i];
			index++;
			}
		}
	}
	/**
	 * Train SVM model. Return alpha and w matrix.
	 * 
	 * */
	public StoreAlphaWeight trainSvm(File saveModel) throws Exception{
		StoreAlphaWeight saww=new StoreAlphaWeight();
		this.modelFile=saveModel;
		Problem problem=new Problem();
		problem.l=train; 
		problem.n=dimensions;
		problem.x=vectrain;
		problem.y=trainattr;
		SolverType s=SolverType.MCSVM_CS;  
        Parameter parameter = new Parameter(s, C, eps);
        Model modelg = Linear.train(problem, parameter, saww);
        try {
			modelg.save(saveModel);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
        return saww;
	}
	public double[] evaluateSvm() throws Exception{
        	int right=0;
			Model model = Model.load(modelFile);
	        for(int t=0;t<test;t++){
	            double prediction = Linear.predict(model, vectest[t]);
	            if(prediction==testattr[t]){
	            	right++;
	            }
	          }
	        double precision=(double)right/test;
	        System.err.println("*************Precision = "+precision*100+"%*************");
	        double storeResult[]=new double[3];
	        storeResult[0]=right;
	        storeResult[1]=test;
	        storeResult[2]=precision;
	        return storeResult;
	}
	
	
}
