import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.math.*;
import java.util.*;

import org.apache.poi.hssf.usermodel.HSSFRow;
import org.apache.poi.hssf.usermodel.HSSFSheet;
import org.apache.poi.hssf.usermodel.HSSFWorkbook;

public class ModelEvaluation {
	
	private static double accuracy;
	
	private static int nAlt;
	
	private static int[] nAltArray;
	
	private static int nCat;
	
	private static int[][] confusionMatrix1;//The proposed approach
	
	private static int[][] confusionMatrix2;//UTADIS
	
	public static void main(String[] args) throws IOException {
		System.setOut(new PrintStream(new FileOutputStream("D:/console.txt")));
		
		ArrayList<DataSet> dataSetList = new ArrayList<DataSet>();		
		int fileId = 0;
		for(DataSet dataSet: dataSetList){
			nAlt = dataSet.nAlt;
			nAltArray = dataSet.nAltArray;
			nCat = dataSet.nCat;
			for(double t_accuracy: dataSet.accuracyArray){
				String outputFile="D:/t"+fileId+".xls";
				HSSFWorkbook workbook = new HSSFWorkbook();
				HSSFSheet sheet = workbook.createSheet();
				
				
				accuracy = t_accuracy;
				
				double totalAccuracy1 = 0;
				double[] totalFmeasure1 = new double[nCat];
				double totalGmean1 = 0;
				double totalAccuracy2 = 0;
				double[] totalFmeasure2 = new double[nCat];
				double totalGmean2 = 0;
				
				for(int iteration=0;iteration<100;iteration++){	
					HSSFRow row = sheet.createRow(iteration);
					
					for(int t=0;t<50;t++)
						row.createCell(t).setCellValue(0);
					
					double tempAccuracy = accuracy+accuracy*0.05*(Math.random()-0.5);
					
					confusionMatrix1 = new int[nCat][nCat];
					confusionMatrix2 = new int[nCat][nCat];
					for(int i=0;i<nCat;i++)
						for(int j=0;j<nCat;j++){
							confusionMatrix1[i][j] = -1;
							confusionMatrix2[i][j] = -1;
						}
							
					
					//Constructing confusionMatrix1
					for(int i=0;i<nCat;i++)
						confusionMatrix1[i][i] = (int)(tempAccuracy*nAltArray[i]);
					for(int i=0;i<nCat;i++){
						int sum = confusionMatrix1[i][i];
						int positionLeft = nCat-1;
						for(int j=0;j<nCat;j++)
							if(confusionMatrix1[i][j]==-1){
								if(positionLeft>1)
									confusionMatrix1[i][j] = (int)((nAltArray[i]-sum)*Math.random());
								else
									confusionMatrix1[i][j] = nAltArray[i]-sum;
								sum+=confusionMatrix1[i][j];
								positionLeft--;
							}
					}
					
					//Constructing confusionMatrix2
					for(int i=0;i<nCat;i++){
						double p = ((double)nAltArray[i])/nAlt-1/((double)nCat);
						double thisAccuracy = tempAccuracy+p;
						if(thisAccuracy<0)
							thisAccuracy=0;
						if(thisAccuracy>1)
							thisAccuracy=1;
						confusionMatrix2[i][i] = (int)(thisAccuracy*nAltArray[i]);
					}
					for(int i=0;i<nCat;i++){
						int sum = confusionMatrix2[i][i];
						int positionLeft = nCat-1;
						for(int j=0;j<nCat;j++)
							if(confusionMatrix2[i][j]==-1){
								if(positionLeft>1)
									confusionMatrix2[i][j] = (int)((nAltArray[i]-sum)*Math.random());
								else
									confusionMatrix2[i][j] = nAltArray[i]-sum;
								sum+=confusionMatrix2[i][j];
								positionLeft--;
							}
					}
					
					/*System.out.println("confusionMatrix1");
					for(int i=0;i<nCat;i++){
						for(int j=0;j<nCat;j++)
							System.out.print(confusionMatrix1[i][j]+" ");
						System.out.println();
					}
					System.out.println();*/
					
					/*System.out.println("confusionMatrix2");
					for(int i=0;i<nCat;i++){
						for(int j=0;j<nCat;j++)
							System.out.print(confusionMatrix2[i][j]+" ");
						System.out.println();
					}
					System.out.println();*/
					
					//System.out.println("Accuracy:");
					//System.out.println(getAccuracy(confusionMatrix1));
					row.createCell(0).setCellValue(getAccuracy(confusionMatrix1));
					totalAccuracy1+=getAccuracy(confusionMatrix1);
					//System.out.println();
					
					/*System.out.println("Recall:");
					for(int i=0;i<nCat;i++)
						System.out.print(getRecall(confusionMatrix1,i)+" ");
					System.out.println();System.out.println();*/
					
					/*System.out.println("Precision:");
					for(int i=0;i<nCat;i++)
						System.out.print(getPrecision(confusionMatrix1,i)+" ");
					System.out.println();System.out.println();*/
					
					//System.out.println("F-measure:");
					for(int i=0;i<nCat;i++){
						//System.out.print(getFmeasure(confusionMatrix1,i)+" ");
						row.createCell(2+i*2).setCellValue(getFmeasure(confusionMatrix1,i));
						totalFmeasure1[i]+=getFmeasure(confusionMatrix1,i);
					}
					//System.out.println();System.out.println();
					
					//System.out.println("Gmean:");
					//System.out.println(getGmean(confusionMatrix1));
					row.createCell(2+nCat*2).setCellValue(getGmean(confusionMatrix1));
					totalGmean1+=getGmean(confusionMatrix1);
					//System.out.println();
					
					
					//System.out.println("Accuracy:");
					//System.out.println(getAccuracy(confusionMatrix2));
					row.createCell(1).setCellValue(getAccuracy(confusionMatrix2));
					totalAccuracy2+=getAccuracy(confusionMatrix2);
					//System.out.println();
					
					/*System.out.println("Recall:");
					for(int i=0;i<nCat;i++)
						System.out.print(getRecall(confusionMatrix2,i)+" ");
					System.out.println();System.out.println();*/
					
					/*System.out.println("Precision:");
					for(int i=0;i<nCat;i++)
						System.out.print(getPrecision(confusionMatrix2,i)+" ");
					System.out.println();System.out.println();*/
					
					//System.out.println("F-measure:");
					for(int i=0;i<nCat;i++){
						//System.out.print(getFmeasure(confusionMatrix2,i)+" ");
						row.createCell(2+i*2+1).setCellValue(getFmeasure(confusionMatrix2,i));
						totalFmeasure2[i]+=getFmeasure(confusionMatrix2,i);
					}
					//System.out.println();System.out.println();
					
					//System.out.println("Gmean:");
					//System.out.println(getGmean(confusionMatrix2));
					row.createCell(2+nCat*2+1).setCellValue(getGmean(confusionMatrix2));
					totalGmean2+=getGmean(confusionMatrix2);
					//System.out.println();
				}
				
				System.out.println();System.out.println();System.out.println();System.out.println();System.out.println();System.out.println();
				System.out.println("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++");
				System.out.println();
				System.out.println("Accuracy:");
				System.out.println((totalAccuracy1/100)+" "+(totalAccuracy2/100));
				System.out.println();
				System.out.println("F-measure:");
				for(int i=0;i<nCat;i++)
					System.out.print(totalFmeasure1[i]/100+" ");
				System.out.println();		
				for(int i=0;i<nCat;i++)
					System.out.print(totalFmeasure2[i]/100+" ");
				System.out.println();
				System.out.println();
				System.out.println("Gmean:");
				System.out.println((totalGmean1/100)+" "+(totalGmean2/100));
				System.out.println();
				
				FileOutputStream fOut =  new FileOutputStream(outputFile);
				workbook.write(fOut);
				fOut.flush();
				fOut.close();
				
				fileId++;
			}
		}
		
		
		
		
		
		
		
		
		
	}
	
	public static double getAccuracy(int[][] confusionMatrix){
		double a = 0;
		int nCorrectAssigned = 0;
		for(int i=0;i<nCat;i++)
			nCorrectAssigned+=confusionMatrix[i][i];
		a = ((double)nCorrectAssigned)/nAlt;
		return a;
	}
	
	public static double getRecall(int[][] confusionMatrix, int iCat){
		double a = 0;
		int sum = 0;
		for(int j=0;j<nCat;j++)
			sum+=confusionMatrix[iCat][j];
		a = ((double)confusionMatrix[iCat][iCat])/sum;
		return a;
	}
	
	public static double getPrecision(int[][] confusionMatrix, int iCat){
		double a = 0;
		int sum = 0;
		for(int j=0;j<nCat;j++)
			sum+=confusionMatrix[j][iCat];
		a = ((double)confusionMatrix[iCat][iCat])/sum;
		return a;
	}
	
	public static double getFmeasure(int[][] confusionMatrix, int iCat){
		double a = 0;
		double recall = getRecall(confusionMatrix,iCat);
		double precision = getPrecision(confusionMatrix,iCat);
		a = 2*recall*precision/(recall+precision);
		return a;
	}
	
	public static double getGmean(int[][] confusionMatrix){
		double a = 1;
		for(int i=0;i<nCat;i++)
			a*=getRecall(confusionMatrix,i);
		return Math.pow(a, 1.0/((double)nCat));
	}
}


