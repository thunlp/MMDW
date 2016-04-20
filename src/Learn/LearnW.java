package Learn;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Random;

import Learn.Evaluate_SVM;


public class LearnW extends Thread {
	double lambda;
	int flagNum;
	double alpha;
	double alphaBias;
	double limitRandom;
	boolean init=true;
	static String modelFile;
	static String linkGraphFile;
	static String labelFile;
	static String W_DW;
	static String source;
	boolean writeVec=true;
	static int alphaLevel;
	
	static int dataNum;//3312;//2708;
	static int dimension=200;//fixed
	int steps_init=40;//fixed
	int steps_after=40;
	static boolean test_switch;
	Random r = new Random(12345l);
	StoreAlphaWeight saw;
	
	double graph[][];
	double W[][];
	double H[][];
	double bias[][];
	int TestTrain[];
	public LearnW()throws Exception{
		BufferedReader trUsers = new BufferedReader(new InputStreamReader(new FileInputStream(labelFile)));
		int countDataNum=0;
		String line="";
		while((line = trUsers.readLine())!=null){
			countDataNum++;
		}
		trUsers.close();
		dataNum=countDataNum;
		graph=new double[dataNum][dataNum];
		W=new double[dataNum][dimension];
		H=new double[dimension][dataNum];
		bias=new double[dataNum][dimension];
		TestTrain=new int[dataNum];
	}
	/**
	 * Read in the normalized Category file.
	 * @param linkGraph The net file of source.
	 * 
	 * */
	public void makeGraph(File linkGraph) throws Exception{
    	String temp212;
    	int rowNum=0;
		BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(linkGraph)));
		while ((temp212 = br.readLine()) != null){
			String[] strs = temp212.split(" ");
			for(int i=0;i<strs.length;i++){
				graph[rowNum][i]=Double.parseDouble(strs[i]);
			}
			rowNum++;
		}
		for(int i=0;i<dataNum;i++){
			for(int j=0;j<dimension;j++){
				W[i][j]=r.nextDouble();
				H[j][i]=r.nextDouble();
			}
		}

		for(int i=0;i<dataNum;i++){
			double sumW=0;
			double sumH=0;
			for(int j=0;j<dimension;j++){
				sumW+=W[i][j]*W[i][j];
				sumH+=H[j][i]*H[j][i];
			}
			sumW=Math.sqrt(sumW);
			sumH=Math.sqrt(sumH);
			for(int j=0;j<dimension;j++){
				W[i][j]/=sumW;
				H[j][i]/=sumH;
			}
		}
        trainW();
        init = false;
	}
	/**
	 * Train user embedding, and add bias to embedding after the first iteration.
	 * In the first iteration, use matrix factorization to train the embedding, and
	 * After the first iteration, use bias to affect the embedding,
     * first add bias to W, then use W to effect H, and finally use
     * H to affect W.
     * 
	 * */
	public void trainW() throws Exception, IOException{
    	double last_loss=0;
    	if(test_switch){
        	String temp212;
        	int rowNum=0;
        	File f=new File(W_DW);
    		BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(f)));
    		while ((temp212 = br.readLine()) != null){
    			String[] strs = temp212.split(" ");
    			for(int i=0;i<strs.length;i++){
    				W[i][rowNum]=Double.parseDouble(strs[i]);
    			}
    			rowNum++;
    		}
    		test_switch=false;
    	}
    	else{
    	if(!init){	
    		preComputeBias();
    	}
    	/*
    	 * In the first iteration, use matrix factorization to train the embedding
    	 * */
    	if(init){
    	for(int step=0;step<steps_init;step++){
    		System.out.println(new Date().toLocaleString()+'\t'+"iter = "+step+'\t'+"Process for training"+'\t'+(step*100)/steps_init+"%");
        	double drv[][]=new double[dimension][dataNum];
        	double rt[][]=new double[dataNum*dimension][1];
        	double dt[][]=new double[dataNum*dimension][1];
        	double B[][]=H;
        	double Hess[][]=new double[dimension][dimension];
        	double store2BB[][]=new double[dimension][dimension];
        	double vecW[][]=new double[dataNum*dimension][1];
        	double vecH[][]=new double[dataNum*dimension][1];
        	for(int i=0;i<dimension;i++){
        		for(int j=0;j<dimension;j++)
        			for(int m=0;m<dataNum;m++)
        				store2BB[i][j]+=2*B[i][m]*B[j][m];
        	}
        	for(int i=0;i<dimension;i++){
        		for(int j=0;j<dataNum;j++)
        			for(int m=0;m<dimension;m++)
        				drv[i][j]+=store2BB[i][m]*W[j][m];
        	}
        	for(int i=0;i<dimension;i++){
        		for(int j=0;j<dataNum;j++)
        			for(int m=0;m<dataNum;m++)
        				drv[i][j]-=2*B[i][m]*graph[j][m];
        	}
        	for(int i=0;i<dimension;i++){
        		for(int j=0;j<dataNum;j++)
        			drv[i][j]+=lambda*W[j][i];
        	}
        	for(int i=0;i<dimension;i++){
        		for(int j=0;j<dimension;j++){
        			for(int m=0;m<dataNum;m++)
        				Hess[i][j]+=2*B[i][m]*B[j][m];
        			if(i==j)
        				Hess[i][j]+=lambda;
        		}
        	}
        	for(int i=0;i<dataNum;i++){
        		for(int j=0;j<dimension;j++)
        			rt[i*dimension+j][0]=-drv[j][i];
        	}
        	for(int i=0;i<dataNum;i++){
        		for(int j=0;j<dimension;j++)
        			drv[j][i]=0;
        	}
        	for(int i=0;i<dimension*dataNum;i++){
        		dt[i][0]=rt[i][0];
        	}
        	for(int i=0;i<dataNum;i++){
        		for(int j=0;j<dimension;j++)
        			vecW[i*dimension+j][0]=W[i][j];
        	}
        	int countW=0;
        	while(true){//update W
        		countW++;
        		double norm=0;
            	for(int i=0;i<dimension*dataNum;i++){
            		norm+=rt[i][0]*rt[i][0];
            	}
            	norm=Math.sqrt(norm);
            	if(countW>10){
            		System.out.println("W norm = "+norm);
            		break;
            	}
            	else{
                	double rtrt=0,dtHdt=0,rtmprtmp=0;
                	double at=0,bt=0;
            		double dtS[][]=new double[dimension][dataNum];
            		double Hdt[][]=new double[dataNum*dimension][1];
            		double storeHessdtS[][]=new double[dimension][dataNum];
            		double rtmp[][]=new double[dataNum*dimension][1];
                	for(int i=0;i<dataNum;i++){
                		for(int j=0;j<dimension;j++)
                			dtS[j][i]=dt[i*dimension+j][0];
                	}
                	for(int i=0;i<dimension;i++){
                		for(int j=0;j<dataNum;j++)
                			for(int m=0;m<dimension;m++)
                				storeHessdtS[i][j]+=Hess[i][m]*dtS[m][j];
                	}
                	for(int i=0;i<dataNum;i++){
                		for(int j=0;j<dimension;j++)
                			Hdt[i*dimension+j][0]=storeHessdtS[j][i];
                	}
                	for(int i=0;i<dimension*dataNum;i++){
                		rtrt+=rt[i][0]*rt[i][0];
                		dtHdt+=dt[i][0]*Hdt[i][0];
                	}
                	at=rtrt/dtHdt;
                	for(int i=0;i<dimension*dataNum;i++){
                		vecW[i][0]+=at*dt[i][0];
                		rtmp[i][0]=rt[i][0];
                	}
                	for(int i=0;i<dimension*dataNum;i++){
                		rt[i][0]-=at*Hdt[i][0];
                	}
                	rtmprtmp=rtrt;
                	for(int i=0;i<dimension*dataNum;i++){
                		rtrt+=rt[i][0]*rt[i][0];
                	}
                	bt=rtrt/rtmprtmp;
                	for(int i=0;i<dimension*dataNum;i++){
                		dt[i][0]=rt[i][0]+bt*dt[i][0];
                	}
            	}
        	}
        	for(int i=0;i<dataNum;i++){
        		for(int j=0;j<dimension;j++)
        			W[i][j]=vecW[i*dimension+j][0];
        	}
        	//update H
        	double storeWW[][]=new double[dimension][dimension];
        	double storeWWH[][]=new double[dimension][dataNum];
        	for(int i=0;i<dimension;i++){
        		for(int j=0;j<dimension;j++)
        			for(int m=0;m<dataNum;m++)
        				storeWW[i][j]+=W[m][i]*W[m][j];
        	}
        	for(int i=0;i<dimension;i++){
        		for(int j=0;j<dataNum;j++)
        			for(int m=0;m<dimension;m++)
        				storeWWH[i][j]+=storeWW[i][m]*H[m][j];
        	}
        	for(int i=0;i<dimension;i++){
        		for(int j=0;j<dataNum;j++){
        			double WMt=0;
        			for(int m=0;m<dataNum;m++)
        				WMt+=W[m][i]*graph[j][m];
        			storeWWH[i][j]-=WMt;
        		}
        	}
        	for(int i=0;i<dimension;i++){
        		for(int j=0;j<dataNum;j++)
        			drv[i][j]=storeWWH[i][j]+lambda*H[i][j];
        	}
        	for(int i=0;i<dataNum;i++){
        		for(int j=0;j<dimension;j++)
        			rt[i*dimension+j][0]=-drv[j][i];
        	}
        	for(int i=0;i<dimension*dataNum;i++){
        		dt[i][0]=rt[i][0];
        	}
        	for(int i=0;i<dataNum;i++){
        		for(int j=0;j<dimension;j++)
        			vecH[i*dimension+j][0]=H[j][i];
        	}
        	int countH=0;
        	while(true){
        		countH++;
        		double norm=0;
            	for(int i=0;i<dimension*dataNum;i++){
            		norm+=rt[i][0]*rt[i][0];
            	}
            	norm=Math.sqrt(norm);
            	if(countH>10){
            		System.out.println("H norm = "+norm);
            		break;
            	}
            	else{
            		double rtrt=0,dtHdt=0,rtmprtmp=0;
                	double at=0,bt=0;
            		double dtS[][]=new double[dimension][dataNum];
            		double Hdt[][]=new double[dataNum*dimension][1];
            		double storeHdt[][]=new double[dimension][dataNum];
            		double rtmp[][]=new double[dataNum*dimension][1];
                	for(int i=0;i<dataNum;i++){
                		for(int j=0;j<dimension;j++)
                			dtS[j][i]=dt[i*dimension+j][0];
                	}
                	for(int i=0;i<dimension;i++){
                		for(int j=0;j<dataNum;j++){
                			double aaa=0;
                			for(int m=0;m<dimension;m++)
                				aaa+=storeWW[i][m]*dtS[m][j];
                			storeHdt[i][j]=aaa+lambda*dtS[i][j];
                		}
                	}
                	for(int i=0;i<dataNum;i++){
                		for(int j=0;j<dimension;j++)
                			Hdt[i*dimension+j][0]=storeHdt[j][i];
                	}
                	for(int i=0;i<dimension*dataNum;i++){
                		rtrt+=rt[i][0]*rt[i][0];
                		dtHdt+=dt[i][0]*Hdt[i][0];
                	}
                	at=rtrt/dtHdt;
                	for(int i=0;i<dimension*dataNum;i++){
                		vecH[i][0]+=at*dt[i][0];
                		rtmp[i][0]=rt[i][0];
                	}
                	for(int i=0;i<dimension*dataNum;i++){
                		rt[i][0]-=at*Hdt[i][0];
                	}
                	rtmprtmp=rtrt;
                	for(int i=0;i<dimension*dataNum;i++){
                		rtrt+=rt[i][0]*rt[i][0];
                	}
                	bt=rtrt/rtmprtmp;
                	for(int i=0;i<dimension*dataNum;i++){
                		dt[i][0]=rt[i][0]+bt*dt[i][0];
                	}
            	}
        	}
        	for(int i=0;i<dataNum;i++){
        		for(int j=0;j<dimension;j++)
        			H[j][i]=vecH[i*dimension+j][0];
        	}
        	//******CALCULATE LOSS********
    		double loss=0;
    		for(int i=0;i<dataNum;i++){
    			for(int j=0;j<dataNum;j++){
    				if(graph[i][j]>0){
    					double error=0;
    					for(int k=0;k<dimension;k++)
    						error+=W[i][k]*H[k][j];
    					loss+=Math.pow(graph[i][j]-error, 2);
    					for(int k=0;k<dimension;k++){
    						loss+=(lambda/2)*(Math.pow(W[i][k], 2)+Math.pow(H[k][j], 2));
    					}
    				}
    			}
    		}
    		if(Math.abs(last_loss-loss)<0.0001){
    			System.out.println("BREAK!! Step:"+'\t'+step);
    			break;
    		}
    		last_loss=loss;
    		System.out.println("loss ="+'\t'+loss);
    	}

    	//*********Matrix factorization end
    	}
    	/*
    	 * After the first iteration, use bias to affect the embedding,
    	 * first add bias to W, then use W to effect H, and finally use
    	 * H to affect W.
    	 * */
    	else{
        	//ADD BIAS
        	for(int i=0;i<dataNum;i++){
           		for(int j=0;j<dimension;j++)
           			W[i][j]+=alphaBias*(bias[i][j]);
        	}
        	//CALCULATE H WITH BIAS AFFECTION
        	Random r1 = new Random(12345l);
    		for(int i=0;i<dataNum;i++){  //clear H
    			for(int j=0;j<dimension;j++){
    				H[j][i]=r1.nextDouble();
    			}
    		}

    		for(int i=0;i<dataNum;i++){
    			double sumH=0;
    			for(int j=0;j<dimension;j++){
    				sumH+=H[j][i]*H[j][i];
    			}
    			sumH=Math.sqrt(sumH);
    			for(int j=0;j<dimension;j++){
    				H[j][i]/=sumH;
    			}
    		}
        	for(int step=0;step<steps_after;step++){
        		System.out.println(new Date().toLocaleString()+'\t'+"iter = "+step+'\t'+"Process 2 for training"+'\t'+(step*100)/steps_after+"%");
            	double drv[][]=new double[dimension][dataNum];
            	double rt[][]=new double[dataNum*dimension][1];
            	double dt[][]=new double[dataNum*dimension][1];
            	double vecH[][]=new double[dataNum*dimension][1];
            	double storeWW[][]=new double[dimension][dimension];
            	double storeWWH[][]=new double[dimension][dataNum];
        	for(int i=0;i<dimension;i++){
        		for(int j=0;j<dimension;j++)
        			for(int m=0;m<dataNum;m++)
        				storeWW[i][j]+=W[m][i]*W[m][j];
        	}
        	for(int i=0;i<dimension;i++){
        		for(int j=0;j<dataNum;j++)
        			for(int m=0;m<dimension;m++)
        				storeWWH[i][j]+=storeWW[i][m]*H[m][j];
        	}
        	for(int i=0;i<dimension;i++){
        		for(int j=0;j<dataNum;j++){
        			double WMt=0;
        			for(int m=0;m<dataNum;m++)
        				WMt+=W[m][i]*graph[j][m];
        			storeWWH[i][j]-=WMt;
        		}
        	}
        	for(int i=0;i<dimension;i++){
        		for(int j=0;j<dataNum;j++)
        			drv[i][j]=storeWWH[i][j]+lambda*H[i][j];
        	}
        	for(int i=0;i<dataNum;i++){
        		for(int j=0;j<dimension;j++)
        			rt[i*dimension+j][0]=-drv[j][i];
        	}
        	for(int i=0;i<dimension*dataNum;i++){
        		dt[i][0]=rt[i][0];
        	}
        	for(int i=0;i<dataNum;i++){
        		for(int j=0;j<dimension;j++)
        			vecH[i*dimension+j][0]=H[j][i];
        	}
        	int countH=0;
        	while(true){
        		countH++;
        		double norm=0;
            	for(int i=0;i<dimension*dataNum;i++){
            		norm+=rt[i][0]*rt[i][0];
            	}
            	norm=Math.sqrt(norm);
            	if(countH>10){
                	System.out.println("H norm = "+norm);
            		break;
            	}
            	else{
            		double rtrt=0,dtHdt=0,rtmprtmp=0;
                	double at=0,bt=0;
            		double dtS[][]=new double[dimension][dataNum];
            		double Hdt[][]=new double[dataNum*dimension][1];
            		double storeHdt[][]=new double[dimension][dataNum];
            		double rtmp[][]=new double[dataNum*dimension][1];
                	for(int i=0;i<dataNum;i++){
                		for(int j=0;j<dimension;j++)
                			dtS[j][i]=dt[i*dimension+j][0];
                	}
                	for(int i=0;i<dimension;i++){
                		for(int j=0;j<dataNum;j++){
                			double aaa=0;
                			for(int m=0;m<dimension;m++)
                				aaa+=storeWW[i][m]*dtS[m][j];
                			storeHdt[i][j]=aaa+lambda*dtS[i][j];
                		}
                	}
                	for(int i=0;i<dataNum;i++){
                		for(int j=0;j<dimension;j++)
                			Hdt[i*dimension+j][0]=storeHdt[j][i];
                	}
                	for(int i=0;i<dimension*dataNum;i++){
                		rtrt+=rt[i][0]*rt[i][0];
                		dtHdt+=dt[i][0]*Hdt[i][0];
                	}
                	at=rtrt/dtHdt;
                	for(int i=0;i<dimension*dataNum;i++){
                		vecH[i][0]+=at*dt[i][0];
                		rtmp[i][0]=rt[i][0];
                	}
                	for(int i=0;i<dimension*dataNum;i++){
                		rt[i][0]-=at*Hdt[i][0];
                	}
                	rtmprtmp=rtrt;
                	for(int i=0;i<dimension*dataNum;i++){
                		rtrt+=rt[i][0]*rt[i][0];
                	}
                	bt=rtrt/rtmprtmp;
                	for(int i=0;i<dimension*dataNum;i++){
                		dt[i][0]=rt[i][0]+bt*dt[i][0];
                	}
            	}
        	}
        	for(int i=0;i<dataNum;i++){
        		for(int j=0;j<dimension;j++)
        			H[j][i]=vecH[i*dimension+j][0];
        	}
    		double loss=0;
    		for(int i=0;i<dataNum;i++){
    			for(int j=0;j<dataNum;j++){
    				if(graph[i][j]>0){
    					double error=0;
    					for(int k=0;k<dimension;k++)
    						error+=W[i][k]*H[k][j];
    					loss+=Math.pow(graph[i][j]-error, 2);
    					for(int k=0;k<dimension;k++){
    						loss+=(lambda/2)*(Math.pow(W[i][k], 2)+Math.pow(H[k][j], 2));
    					}
    				}
    			}
    		}
    		if(Math.abs(last_loss-loss)<0.0001){
    			System.out.println("BREAK!! Step:"+'\t'+step);
    			break;
    		}
    		last_loss=loss;
    		System.out.println("loss ="+'\t'+loss);
        	}
        	//H FIXED
        	//CALCULATE W
        	Random r2 = new Random(12345l);
    		for(int i=0;i<dataNum;i++){//CLEAR W
    			for(int j=0;j<dimension;j++){
    				W[i][j]=r2.nextDouble();
    			}
    		}
    		for(int i=0;i<dataNum;i++){
    			double sumW=0;
    			for(int j=0;j<dimension;j++){
    				sumW+=W[i][j]*W[i][j];
    			}
    			sumW=Math.sqrt(sumW);
    			for(int j=0;j<dimension;j++){
    				W[i][j]/=sumW;
    			}
    		}
        	for(int step2=0;step2<steps_after;step2++){
        		System.out.println(new Date().toLocaleString()+'\t'+"iter = "+step2+'\t'+"Process 2 for training"+'\t'+(step2*100)/steps_after+"%");
            	double B[][]=H;
            	double Hess[][]=new double[dimension][dimension];
            	double store2BB[][]=new double[dimension][dimension];
            	double vecW[][]=new double[dataNum*dimension][1];
            	double drv[][]=new double[dimension][dataNum];
            	double rt[][]=new double[dataNum*dimension][1];
            	double dt[][]=new double[dataNum*dimension][1];
        	for(int i=0;i<dimension;i++){
        		for(int j=0;j<dimension;j++)
        			for(int m=0;m<dataNum;m++)
        				store2BB[i][j]+=2*B[i][m]*B[j][m];
        	}
        	for(int i=0;i<dimension;i++){
        		for(int j=0;j<dataNum;j++)
        			for(int m=0;m<dimension;m++)
        				drv[i][j]+=store2BB[i][m]*W[j][m];
        	}
        	for(int i=0;i<dimension;i++){
        		for(int j=0;j<dataNum;j++)
        			for(int m=0;m<dataNum;m++)
        				drv[i][j]-=2*B[i][m]*graph[j][m];
        	}
        	for(int i=0;i<dimension;i++){
        		for(int j=0;j<dataNum;j++)
        			drv[i][j]+=lambda*W[j][i];
        	}
        	for(int i=0;i<dimension;i++){
        		for(int j=0;j<dimension;j++){
        			for(int m=0;m<dataNum;m++)
        				Hess[i][j]+=2*B[i][m]*B[j][m];
        			if(i==j)
        				Hess[i][j]+=lambda;
        		}
        	}
        	for(int i=0;i<dataNum;i++){
        		for(int j=0;j<dimension;j++)
        			rt[i*dimension+j][0]=-drv[j][i];
        	}
        	for(int i=0;i<dataNum;i++){
        		for(int j=0;j<dimension;j++)
        			drv[j][i]=0;
        	}
        	for(int i=0;i<dimension*dataNum;i++){
        		dt[i][0]=rt[i][0];
        	}
        	for(int i=0;i<dataNum;i++){
        		for(int j=0;j<dimension;j++)
        			vecW[i*dimension+j][0]=W[i][j];
        	}
        	int countW=0;
        	while(true){//update W
        		countW++;
        		double norm=0;
            	for(int i=0;i<dimension*dataNum;i++){
            		norm+=rt[i][0]*rt[i][0];
            	}
            	norm=Math.sqrt(norm);
            	if(countW>10){
                	System.out.println("W norm = "+norm);
            		break;
            	}
            	else{
                	double rtrt=0,dtHdt=0,rtmprtmp=0;
                	double at=0,bt=0;
            		double dtS[][]=new double[dimension][dataNum];
            		double Hdt[][]=new double[dataNum*dimension][1];
            		double storeHessdtS[][]=new double[dimension][dataNum];
            		double rtmp[][]=new double[dataNum*dimension][1];
                	for(int i=0;i<dataNum;i++){
                		for(int j=0;j<dimension;j++)
                			dtS[j][i]=dt[i*dimension+j][0];
                	}
                	for(int i=0;i<dimension;i++){
                		for(int j=0;j<dataNum;j++)
                			for(int m=0;m<dimension;m++)
                				storeHessdtS[i][j]+=Hess[i][m]*dtS[m][j];
                	}
                	for(int i=0;i<dataNum;i++){
                		for(int j=0;j<dimension;j++)
                			Hdt[i*dimension+j][0]=storeHessdtS[j][i];
                	}
                	for(int i=0;i<dimension*dataNum;i++){
                		rtrt+=rt[i][0]*rt[i][0];
                		dtHdt+=dt[i][0]*Hdt[i][0];
                	}
                	at=rtrt/dtHdt;
                	for(int i=0;i<dimension*dataNum;i++){
                		vecW[i][0]+=at*dt[i][0];
                		rtmp[i][0]=rt[i][0];
                	}
                	for(int i=0;i<dimension*dataNum;i++){
                		rt[i][0]-=at*Hdt[i][0];
                	}
                	rtmprtmp=rtrt;
                	for(int i=0;i<dimension*dataNum;i++){
                		rtrt+=rt[i][0]*rt[i][0];
                	}
                	bt=rtrt/rtmprtmp;
                	for(int i=0;i<dimension*dataNum;i++){
                		dt[i][0]=rt[i][0]+bt*dt[i][0];
                	}
            	}
        	}
        	for(int i=0;i<dataNum;i++){
        		for(int j=0;j<dimension;j++)
        			W[i][j]=vecW[i*dimension+j][0];
        	}
    		double loss=0;
    		for(int i=0;i<dataNum;i++){
    			for(int j=0;j<dataNum;j++){
    				if(graph[i][j]>0){
    					double error=0;
    					for(int k=0;k<dimension;k++)
    						error+=W[i][k]*H[k][j];
    					loss+=Math.pow(graph[i][j]-error, 2);
    					for(int k=0;k<dimension;k++){
    						loss+=(lambda/2)*(Math.pow(W[i][k], 2)+Math.pow(H[k][j], 2));
    					}
    				}
    			}
    		}
    		if(Math.abs(last_loss-loss)<0.0001){
    			System.out.println("BREAK!! Step:"+'\t'+step2);
    			break;
    		}
    		last_loss=loss;
    		System.out.println("loss ="+'\t'+loss);
        	}
    	}
    	}
	}
	/**
	 * Make a map to differentiate train and test set for training SVM in different percentage.
	 * 0 stands for user in test set and 1 stands for user in train set.
	 * 
	 * */
	public void makeTrainTest() throws Exception{
		int labels[]=new int[dataNum];
		int labelClassNum[]=new int[20];
		BufferedReader trUsers = new BufferedReader(new InputStreamReader(new FileInputStream(labelFile)));
		String line = "";
		while((line = trUsers.readLine())!=null){
				String[] strs = line.split("\t");
				labels[Integer.valueOf(strs[0])]=Integer.valueOf(strs[1]);
				labelClassNum[labels[Integer.valueOf(strs[0])]]++;
		}
		trUsers.close();
		int countClassNum=0;
		for(int i=0;i<labelClassNum.length;i++){
			if(labelClassNum[i]!=0)
				countClassNum++;
		}
		int labelClass[]=new int[countClassNum];
		Random r_classifier=new Random(123l);
		for(int i=0;i<dataNum;i++){
			double nextRandom=r_classifier.nextDouble();
			if(nextRandom<=limitRandom){
				TestTrain[i]=1; //train
				labelClass[labels[i]]++;
			}
		}
		for(int i=0;i<labelClass.length;i++){
			if(labelClass[i]==0){
				Random r_classifier2=new Random(123l);
				for(int j=0;j<dataNum;j++){
					if(labels[j]==i){
						double nextRandom2=r_classifier2.nextDouble();
						if(nextRandom2<=limitRandom){
							TestTrain[j]=1;//train
						}
					}
				}
			}
		}
		
	}
	/**
	 * Write the trained embedding to certain files.
	 * @param ftrain The file with train embedding.
	 * @param ftest  The file with test embedding.
	 * @param all	 The file with all embedding.
	 * @param trainLabel The file with label of train set users.
	 * 
	 * */
	public void writeVecor(File ftrain,File ftest,File all,File trainLabel) throws Exception{
		int labels[]=readLabels();
    	FileWriter fw=new FileWriter(ftrain);
    	FileWriter fwt=new FileWriter(ftest);
    	FileWriter flabel=new FileWriter(trainLabel);
    	for(int i=0;i<dataNum;i++){
    		if(TestTrain[i]==1){
    			flabel.write(String.valueOf(labels[i])+'\n');
    		for(int j=0;j<dimension;j++){
    			if(j!=dimension-1){
    				fw.write(String.valueOf(W[i][j])+" ");
    			}
    			else{
    				fw.write(String.valueOf(W[i][j])+'\n');
    			}
    		}
    	}
    		else{
        		for(int j=0;j<dimension;j++){
        			if(j!=dimension-1){
        				fwt.write(String.valueOf(W[i][j])+" ");
        			}
        			else{
        				fwt.write(String.valueOf(W[i][j])+'\n');
        			}
        		}
    		}
    	}
    	fw.close();
    	fwt.close();
    	flabel.close();
    	FileWriter fwall=new FileWriter(all);
    	for(int i=0;i<dataNum;i++){
    		for(int j=0;j<dimension;j++){
    			if(j!=dimension-1){
    				fwall.write(String.valueOf(W[i][j])+" ");
    			}
    			else{
    				fwall.write(String.valueOf(W[i][j])+'\n');
    			}
    		}
    	}
    	fwall.close();
	}
	/**
	 * Calculate bias for embedding in train set.
	 * This need the alpha and w matrix getting from training SVM model.
	 * 
	 * */
	public void preComputeBias() throws Exception{
		int labels[]=readLabels();
		System.out.println(saw.alphaB.length);
		int map[]=new int[saw.alphaB.length];
		int makeMap=0;
		for(int i=0;i<dataNum;i++){
			if(TestTrain[i]==1){
				map[makeMap]=i;
				makeMap++;
			}
		}
		double[][] alpha = saw.alphaB;
		FileWriter fw111=new FileWriter(new File(modelFile+"/Bias/"+source+"_alphaBiasLevel_"+alphaLevel+"_alpha_"+String.valueOf(flagNum)+"->"+String.valueOf(limitRandom*100)+"%"+".txt"));
		for(int i=0;i<alpha.length;i++){
			fw111.write(String.valueOf(labels[map[saw.indexSvm[i]]])+" ");
			for(int j=0;j<saw.nr_class;j++)
				fw111.write(String.valueOf(alpha[i][j])+" ");
			fw111.write('\n');
		}
		fw111.close();
    	double[][] weightSvm = saw.weightB;
    	for(int i = 0; i < saw.alphaB.length; i ++){
    		int id = map[saw.indexSvm[i]];
    		int label = labels[id];
    		//System.out.println(saw.nr_class);
    		for(int j = 0; j < saw.nr_class; j ++){
				double c = 0;
				if(j == label){
					c = Evaluate_SVM.C;
				}
    			for(int k = 0; k < dimension; k ++) 
    				bias[id][k] +=  (c-alpha[i][j])*(weightSvm[label][k]-weightSvm[j][k]);
    		}
    		double sum = 0;
    		for(int j = 0; j < dimension; j ++){
        		sum += bias[id][j]*bias[id][j];       	
        		}
    		sum = Math.sqrt(sum);
    		if(sum > 0){
	    		for(int j = 0; j < dimension; j ++){
	        		bias[id][j] /= sum;
	    		}
    		}
    	}
    	FileWriter twBias = new FileWriter(new File(modelFile+"/Bias/"+source+"_alphaBiasLevel_"+alphaLevel+"_Bias_"+String.valueOf(flagNum)+"->"+String.valueOf(limitRandom*100)+"%"+".txt"));
    	for(int i=0;i<dataNum;i++){
    		if(TestTrain[i]==1){
    			twBias.write(String.valueOf(i)+"\t"+String.valueOf(labels[i])+"\t");
    			for(int j = 0; j < dimension; j ++){
    				twBias.write(String.valueOf(bias[i][j])+"\t");
    			}
    			twBias.write("\n");
    		}
    	}
    	twBias.close();
	}
	/**
	 * Read label from category files of different source
	 * 
	 * */
	public int[] readLabels() throws Exception{
		int labels[]=new int[dataNum];
		BufferedReader trUsers = new BufferedReader(new InputStreamReader(new FileInputStream(labelFile)));
		String line = "";
		while((line = trUsers.readLine())!=null){
				String[] strs = line.split("\t");
				labels[Integer.valueOf(strs[0])]=Integer.valueOf(strs[1]);
		}
		trUsers.close();
		return labels;
	}
	
	public void run(){
		System.out.println("the "+flagNum+" alphaBias = "+alphaBias+'\t'+"lambda = "+lambda+'\t'+"alpha = "+alpha+'\t'+"limitRandom = "+limitRandom);
		int LoopSize = 10;
		for(int i=0;i<LoopSize;i++){
			 long start = System.currentTimeMillis() ; 
		        try{
		        	if(i == 0){
		        		makeGraph(new File(linkGraphFile)); 
		        		makeTrainTest();//TestTrain,test=0,train=1
		        	}else{
		        		trainW();  
		        	}
		        	if(writeVec){
		        		writeVecor(new File(modelFile+"/vector/"+source+"_alphaBiasLevel_"+alphaLevel+"_vectortrain_"+String.valueOf(flagNum)+"->"+String.valueOf(limitRandom*100)+"%"+".txt"),
		        				   new File(modelFile+"/vector/"+source+"_alphaBiasLevel_"+alphaLevel+"_vectortest_" +String.valueOf(flagNum)+"->"+String.valueOf(limitRandom*100)+"%"+".txt"),
		        				   new File(modelFile+"/vector/"+source+"_alphaBiasLevel_"+alphaLevel+"_vectorall_"+String.valueOf(flagNum)+"->"+String.valueOf(limitRandom*100)+"%"+".txt"),
		        				   new File(modelFile+"/vector/"+source+"_alphaBiasLevel_"+alphaLevel+"_trainlabel_"+String.valueOf(flagNum)+"->"+String.valueOf(limitRandom*100)+"%"+".txt"));
		        		//writeVec=false;
		        	}
		        	System.out.println(i+"\tTrain embeddings over! use time "+(System.currentTimeMillis()-start));
		        	/*
		        	 * First train DeepWalk,
		        	 * then train SVM 
		        	 * 
		        	 * */
		        	Evaluate_SVM svm=new Evaluate_SVM(dataNum,dimension,W,labelFile,limitRandom,TestTrain);
		        	saw=svm.trainSvm(new File(modelFile+"/svm_model/"+source+"_alphaBiasLevel_"+alphaLevel+"_model_" +String.valueOf(flagNum)+"->"+String.valueOf(limitRandom*100)+"%"+".txt"));
		        	double resultReceive[]=svm.evaluateSvm();
		        	FileWriter frw = new FileWriter(new File(modelFile+"/result/"+source+"_alphaBiasLevel_"+alphaLevel+"_Result_"+String.valueOf(flagNum)+"->"+String.valueOf(limitRandom*100)+"%"+".txt"),true);
		        	frw.write("**********************************************************************"+'\n');
		        	frw.write("Iter "+i+'\n');
		        	frw.write("Right = "+resultReceive[0]+"   "+"TestData = "+resultReceive[1]+"   "+"Precision = "+resultReceive[2]*100+"%"+'\n');
		        	frw.write("**********************************************************************"+'\n'+'\n'+'\n');
				frw.close();
		        }
		        catch(Exception e){
		        	e.printStackTrace();
		        }
		}
	}
	
	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub
		//********SWITCH**********
		test_switch=false;
		//********FIXED***********
		StoreAlphaWeight.dimensionForSVM=dimension;//fixed
		source=args[0];//1st input
		System.out.println("File type is : "+source);
		modelFile=args[1];//2nd input
		System.out.println("Folder of data is at : "+modelFile);
		alphaLevel=Integer.valueOf(args[2]);
		System.out.println("AlphaBias Level : "+alphaLevel);
		labelFile=modelFile+"/Category/"+source+"_category.txt";
		linkGraphFile=modelFile+"/Net/"+source+"_net.txt";
		W_DW=modelFile+"/W_DW/W_"+source+"_LINE.txt";
		//********THREAD**********
		List<LearnW> lls=new ArrayList<LearnW>();
    	for(int i=0;i<9;i++){
    		LearnW ls=new LearnW();
    		ls.lambda=0.3;
    		ls.flagNum=i;
    		ls.alpha=0.005;
    		ls.alphaBias=25*Math.pow(10, alphaLevel);
    		ls.limitRandom=0.1+0.1*i;//DIFFERENT PERCENTAGE OF TRAIN SET
    		lls.add(ls);
    	}
    	System.out.println("Number of data : "+dataNum);
    	for(int i=0;i<9;i++){
			LearnW lsls1=lls.get(i);
			lsls1.start();
    	}
	}

}
