import java.io.*;
import java.util.*;
import java.util.regex.Pattern;

import ilog.concert.*;
import ilog.cplex.*;

public class DataPreprocessing{
	private static int nRef = 100;
	
	private static ArrayList<Alternative> refList = new ArrayList<Alternative>();
	
	private static int nAlt = nRef;
	
	private static ArrayList<Alternative> altList = new ArrayList<Alternative>();
	
	private static int nCri = 2;
	
	private static double[] w = {0.35, 0.65};
	
	private static int nPoint = 1;
	
	private static int nCat = 3;
	
	private static double[] cardReq = {0.05, 0.15, 0.80};
	
	private static double outlierRate = 0.05;
	
	private static double outlierRatio = 10.0;
	
	private static double smallPositiveConstants = 0.00001;
	
	private static IloCplex cplex;
	
	private static int isFirstRead = 1;
		
	public static void main(String[] args) throws Exception {
		//System.setOut(new PrintStream(new FileOutputStream("C:/console.txt")));
		
		if(isFirstRead==0){
			generateData();
			
			dataProcessing();
		}else{
			secondDataProcessing();
		}
		
		ArrayList<ArrayList<Alternative>> finalAlternativeList = new ArrayList<ArrayList<Alternative>>();
		for(int iCat=1;iCat<=nCat;iCat++){
			ArrayList<Alternative> addedList = new ArrayList<Alternative>();
			for(Alternative a:refList)
				if(a.getRefAssignment()==iCat)
					addedList.add(a);
			finalAlternativeList.add(addedList);
		}
		double[][] returnArray = finalSoring(finalAlternativeList);
		
		double correctRate = evaluatePerformance(returnArray);
		
		System.out.println();
		System.out.println("The correct rate is "+correctRate);
	}
	
	public static void myPrint(Object o){
		if(o==null)
			System.out.println();
		else
			System.out.print(o);
	}
	
	public static void myPrintln(Object o){
		if(o==null)
			System.out.println();
		else
			System.out.println(o);
	}
	
	public static void generateData(){
		try {
			FileOutputStream fs = new FileOutputStream(new File("C:/rawData.txt"));
			PrintStream ps = new PrintStream(fs);
						
			for(int i=0;i<nRef+nAlt;i++){
				for(int j=0;j<nCri;j++){
					double eval = Math.random();
					if(eval<=0.01)
						eval = Math.random();
					ps.print(String.valueOf(eval).substring(0, 5)+" ");
				}
				ps.print("\r\n");
			}			
			
			ps.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	public static void dataProcessing(){
		try {
			InputStreamReader isr = new InputStreamReader(new FileInputStream(new File("C:/rawData.txt")));
			BufferedReader br = new BufferedReader(isr);
			
			boolean TF = true;
			int id = 0;
			String lineTxt = null;
	        while((lineTxt = br.readLine()) != null){    
	        	if(id==nRef){
	        		id=0;
	        		TF=false;
	        	}
	        	
	        	Alternative alt = new Alternative(id, null, null, -1, -1, -1);
	        	ArrayList<int[]> disturbMerge = new ArrayList<int[]>();
	        	alt.setDisturbMerge(disturbMerge);
	        		        	
	        	Pattern pattern = Pattern.compile(" ");
	        	String[] evalVecStr = pattern.split(lineTxt);
	        	double[] evalVec = new double[nCri];
	        	for(int i=0;i<nCri;i++)
	        		evalVec[i] = Double.valueOf(evalVecStr[i]);
	        	alt.setEvalVec(evalVec);
	        	
	        	int position = 0;
	        	double[] valueVec = new double[nCri*nPoint]; 
	        	for(int i=0;i<nCri;i++){
	        		int index = 0;
	        		while(evalVec[i]>((double)index)/((double)nPoint))
	        			index++;
	        		for(int j=0;j<index-1;j++){
	        			valueVec[position] = 1;
	        			position++;	        			
	        		}
	        		valueVec[position] = (evalVec[i]-((double)(index-1))/((double)nPoint))/(1.00/((double)nPoint));
	        		position++;
	        		for(int j=0;j<nPoint-index;j++){
	        			valueVec[position] = 0;
	        			position++;
	        		}
	        	}
	        	alt.setValueVec(valueVec);
	        	
	        	double utility = 0;
	        	for(int i=0;i<nCri;i++)
	        		utility+=w[i]*evalVec[i];
	        	alt.setRealValue(utility);
	        		        	
	        	if(TF)
	        		refList.add(alt);
	        	else
	        		altList.add(alt);
	        	
	        	id++;
	        }
	        br.close();
	       
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		Collections.sort(refList, valueComparator);
		int iAlt = 0;
		int iCat = 1;
		for(int i=0;i<cardReq.length;i++){
			int iCnt = 0;
			while(iCnt<((int)(cardReq[i]*nRef))){
				refList.get(iAlt).setActualSorting(iCat);
				refList.get(iAlt).setRefAssignment(iCat);
				iCnt++;
				iAlt++;
			}
			iCat++;
		}
		
	    double[] threshold = new double[nCat+1];
	    threshold[nCat] = 1.001;
	    threshold[0] = 0;
	    int pos = nCat-1;
	    for(int i=0;i<refList.size()-1;i++){
	    	if(refList.get(i).getActualSorting()!=refList.get(i+1).getActualSorting()){
	    		threshold[pos] = (refList.get(i).getRealValue()+refList.get(i+1).getRealValue())/2;
	    		pos--;
	    	}
	    }
	    /*for(int i=0;i<nCat+1;i++)
	    	System.out.println("Threshold is "+threshold[i]);*/
        
        /*for(Alternative a:refList)
        	System.out.println(a.getId()+" "+a.getRealValue()+" "+a.getActualSorting()+" "+a.getRefAssignment());*/
        
        ArrayList<Integer> outlierPositionList = new ArrayList<Integer>(); 
        for(int i=0;i<outlierRate*nRef;i++){
			int r = (int)(nRef * Math.random());
			while(outlierPositionList.contains(r))
				r = (int)(nRef * Math.random());
			outlierPositionList.add(r);
		}
        /*for(int i=0;i<outlierPositionList.size();i++)
        	System.out.print(outlierPositionList.get(i)+" ");
        System.out.println();*/
		for(int i=0;i<nRef;i++)
			if(outlierPositionList.contains(i)){
				int r = (int)(nCat*Math.random())+1;
				while(!(r>=1 && r<=nCat) || r==refList.get(i).getActualSorting())
					r = (int)(nCat*Math.random())+1;
				//System.out.print(refList.get(i).getId()+" "+refList.get(i).getActualSorting()+" "+refList.get(i).getRefAssignment()+" to ");
				refList.get(i).setRefAssignment(r);
				//System.out.println(refList.get(i).getId()+" "+refList.get(i).getActualSorting()+" "+refList.get(i).getRefAssignment());
	        }     
		
		
        Collections.sort(refList, idComparator);
//        for(Alternative a:refList)
//        	System.out.println(a.getId()+" "+a.getRealValue()+" "+a.getActualSorting()+" "+a.getRefAssignment());
        
        
		Collections.sort(altList, valueComparator);
		iAlt = 0;
		iCat = 1;
		for(int i=0;i<cardReq.length;i++){
			int iCnt = 0;
			while(iCnt<((int)(cardReq[i]*nAlt))){
				altList.get(iAlt).setActualSorting(iCat);
				altList.get(iAlt).setRefAssignment(iCat);
				iCnt++;
				iAlt++;
			}
			iCat++;
		}
	    Collections.sort(altList, idComparator);*/
        
        for(Alternative alt:altList)
        	for(int i=1;i<=nCat;i++)
        		if((alt.getRealValue()>=threshold[nCat-i])&&(alt.getRealValue()<threshold[nCat-i+1])){
        			alt.setActualSorting(i);
        			alt.setRefAssignment(i);
        		}
        
        
		myPrintln(refList.size());
		myPrintln(altList.size());
		
		int[] nRefOfCat = new int[nCat];
		for(int i=0;i<refList.size();i++)
			nRefOfCat[refList.get(i).getRefAssignment()-1]++;
		for(int i=0;i<nCat;i++)
			System.out.println(nRefOfCat[i]);
		
//		System.out.println("****************************************************************");		
//		for(int i=0;i<refList.size();i++){
//			myPrintln(refList.get(i).getId());
//			
////			double[] evalVec = refList.get(i).getEvalVec();
////			for(int j=0;j<nCri;j++){
////				myPrint(evalVec[j]+" ");
////				myPrintln(null);
////			}
////			
////			double[] valueVec = refList.get(i).getValueVec();
////			for(int j=0;j<nCri*nPoint;j++){
////				myPrint(valueVec[j]+" ");
////				myPrintln(null);
////			}
//						
//			myPrintln(refList.get(i).getClusterId());
//			myPrintln(refList.get(i).getActualSorting());
//			myPrintln(refList.get(i).getRefAssignment());
//			myPrintln(null);
//			System.out.println("****************************************************************");
//		}
//		
//		for(int i=0;i<altList.size();i++){
//			myPrintln(altList.get(i).getId());
//			
////			double[] evalVec = altList.get(i).getEvalVec();
////			for(int j=0;j<nCri;j++){
////				myPrint(evalVec[j]+" ");
////				myPrintln(null);
////			}
////			
////			double[] valueVec = altList.get(i).getValueVec();
////			for(int j=0;j<nCri*nPoint;j++){
////				myPrint(valueVec[j]+" ");
////				myPrintln(null);
////			}
//						
//			myPrintln(altList.get(i).getClusterId());
//			myPrintln(altList.get(i).getActualSorting());
//			myPrintln(altList.get(i).getRefAssignment());
//			myPrintln(null);
//			System.out.println("****************************************************************");
//		}
		
		try {
			FileOutputStream fs = new FileOutputStream(new File("C:/rawData.txt"));
			PrintStream ps = new PrintStream(fs);
			
			for(Alternative alt:refList){
				for(int j=0;j<nCri;j++)
					ps.print(String.valueOf(alt.getEvalVec()[j])+" ");
				ps.print(alt.getActualSorting()+" ");
				ps.print(alt.getRefAssignment()+" ");				
				ps.print("\r\n");
			}
						
			for(Alternative alt:altList){
				for(int j=0;j<nCri;j++)
					ps.print(String.valueOf(alt.getEvalVec()[j])+" ");
				ps.print(alt.getActualSorting()+" ");	
				ps.print(alt.getRefAssignment()+" ");	
				ps.print("\r\n");
			}			
			
			ps.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	
	public static void secondDataProcessing(){
		
		try {
			InputStreamReader isr = new InputStreamReader(new FileInputStream(new File("C:/rawData.txt")));
			BufferedReader br = new BufferedReader(isr);
			
			boolean TF = true;
			int id = 0;
			String lineTxt = null;
	        while((lineTxt = br.readLine()) != null){    
	        	if(id==nRef){
	        		id=0;
	        		TF=false;
	        	}
	        	
	        	Alternative alt = new Alternative(id, null, null, -1, -1, -1);
	        	ArrayList<int[]> disturbMerge = new ArrayList<int[]>();
	        	alt.setDisturbMerge(disturbMerge);
	        		        	
	        	Pattern pattern = Pattern.compile(" ");
	        	String[] evalVecStr = pattern.split(lineTxt);
	        	double[] evalVec = new double[nCri];
	        	for(int i=0;i<nCri;i++)
	        		evalVec[i] = Double.valueOf(evalVecStr[i]);
	        	alt.setEvalVec(evalVec);
	        	
	        	alt.setActualSorting(Integer.valueOf(evalVecStr[nCri]));
	        	alt.setRefAssignment(Integer.valueOf(evalVecStr[nCri+1]));
	        	
	        	int position = 0;
	        	double[] valueVec = new double[nCri*nPoint]; 
	        	for(int i=0;i<nCri;i++){
	        		int index = 0;
	        		while(evalVec[i]>((double)index)/((double)nPoint))
	        			index++;
	        		for(int j=0;j<index-1;j++){
	        			valueVec[position] = 1;
	        			position++;	        			
	        		}
	        		valueVec[position] = (evalVec[i]-((double)(index-1))/((double)nPoint))/(1.00/((double)nPoint));
	        		position++;
	        		for(int j=0;j<nPoint-index;j++){
	        			valueVec[position] = 0;
	        			position++;
	        		}
	        	}
	        	alt.setValueVec(valueVec);
	        	
	        	double utility = 0;
	        	for(int i=0;i<nCri;i++)
	        		utility+=w[i]*evalVec[i];
	        	alt.setRealValue(utility);
	        		        	
	        	if(TF)
	        		refList.add(alt);
	        	else
	        		altList.add(alt);
	        	
	        	id++;
	        }
	        br.close();
	       
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	static Comparator<Alternative> valueComparator = new Comparator<Alternative>(){
		public int compare(Alternative a1, Alternative a2){
			if(a1.getRealValue()<a2.getRealValue())
				return 1;
			else if(a1.getRealValue()==a2.getRealValue())
				return 0;		
			else
				return -1;
		}		
	};
	
	static Comparator<Alternative> idComparator = new Comparator<Alternative>(){
		public int compare(Alternative a1, Alternative a2){
			if(a1.getId()<a2.getId())
				return -1;
			else if(a1.getId()==a2.getId())
				return 0;		
			else
				return 1;
		}		
	};
		
	public static double[][] finalSoring(ArrayList<ArrayList<Alternative>> finalAlternativeList) throws Exception{
		int nFinalAlt = 0;
		for(ArrayList<Alternative> al:finalAlternativeList)
			nFinalAlt = nFinalAlt + al.size();
		
		cplex = new IloCplex();
		
		IloNumVar[] value = new IloNumVar[nFinalAlt];
		for(int k=0;k<nFinalAlt;k++)
			value[k] = cplex.numVar(0, 1);
		
		IloNumVar[] u = new IloNumVar[nCri*nPoint];
		for(int k=0;k<nCri*nPoint;k++)
			u[k] = cplex.numVar(0, 1);
		IloNumExpr expression = cplex.numExpr();
		for(int k=0;k<nCri*nPoint;k++)
			expression = cplex.sum(expression, u[k]);
		cplex.addEq(expression, 1);
		
		double epsilon = smallPositiveConstants;
		
		IloNumVar[] b = new IloNumVar[nCat+1];
		for(int k=0;k<nCat+1;k++)
			b[k] = cplex.numVar(0, 1+epsilon);
		for(int k=0;k<nCat;k++)
			cplex.addLe(cplex.sum(b[k], epsilon), b[k+1]);
		cplex.addEq(b[0], 0);
		cplex.addEq(b[nCat], 1+epsilon);
		
		IloNumVar[] delta1 = new IloNumVar[nFinalAlt];
		for(int k=0;k<nFinalAlt;k++)
			delta1[k] = cplex.numVar(0, 10);
		IloNumVar[] delta2 = new IloNumVar[nFinalAlt];
		for(int k=0;k<nFinalAlt;k++)
			delta2[k] = cplex.numVar(0, 10);
		
		int k=0;
		for(ArrayList<Alternative> al:finalAlternativeList)
			for(Alternative alt:al){
				cplex.addEq(value[k], cplex.scalProd(alt.getValueVec(), u));
				cplex.addGe(cplex.sum(value[k], delta1[k]), b[nCat+1-alt.getRefAssignment()-1]);
				cplex.addLe(cplex.sum(value[k], cplex.prod(-1, delta2[k])), cplex.sum(b[nCat+1-alt.getRefAssignment()], -epsilon));
				k++;
			}
			
		expression = cplex.numExpr();
		for(k=0;k<nFinalAlt;k++)
			expression = cplex.sum(expression, delta1[k], delta2[k]);
		
		cplex.addMinimize(expression);			
		cplex.solve();
		double minSumDelta = cplex.getValue(expression);
		
		
		System.out.println();
		System.out.println();
		System.out.println(minSumDelta);
		
		System.out.println();
		k=0;
		for(ArrayList<Alternative> al:finalAlternativeList)
			for(Alternative alt:al){
				System.out.println(alt.getRefAssignment()+" "+cplex.getValue(value[k])+" "+cplex.getValue(delta1[k])+" "+cplex.getValue(delta2[k]));
				k++;
			}
		
		double[] returnU = new double[nCri*nPoint];
		System.out.println();
		for(k=0;k<nCri*nPoint;k++){
			returnU[k] = cplex.getValue(u[k]);
			System.out.print(returnU[k]+" ");
		}
		System.out.println();
		double[] returnB = new double[nCat+1];
		System.out.println();
		for(k=0;k<nCat+1;k++){
			returnB[k] = cplex.getValue(b[k]);
			System.out.print(returnB[k]+" ");
		}
		System.out.println();
		
		cplex.clearModel();
		cplex.end();
		
		double[][] returnArray = new double[2][];
		returnArray[0] = returnU;
		returnArray[1] = returnB;
		
		return returnArray;
	}
	
	public static double evaluatePerformance(double[][] returnArray){
		double[] u = returnArray[0];
		double[] b = returnArray[1];
		
		double correctRate = 0;
		
		for(Alternative alt:altList){
			double value = 0;
			for(int k=0;k<nCri*nPoint;k++)
				value = value + u[k]*alt.getValueVec()[k];
			
			int pos = 0;
			while(value>=b[pos])
				pos++;
			
			int assignment = nCat+1-pos;
			
			if(assignment == alt.getActualSorting())
				correctRate = correctRate+1;
		}
		
		System.out.println(correctRate);
		
		correctRate = correctRate/altList.size();
		
		return correctRate;
	}
}