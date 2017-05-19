/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package JavaWeka;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;

/**
 *
 * @author neomushlih
 */
public class SingleLabel {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws Exception {
        
        //dataset name
        String name = "diabetes";

        Instances train = null;  //this is train data if you define the test data
        Instances test = null;
        //change this directory into yours
        train = ReadFile("properties/diabetes-train.arff");
        test = ReadFile("properties/diabetes-test.arff");
        
        ////filtering purpose, but i think it's not used in this example
        //Instances newData = filtering(data);

        //declare the classifier, I use Naive Bayes as example
        Classifier scheme = new NaiveBayes();
        try {
            //if you want to use option
            // scheme.setOptions(weka.core.Utils.splitOptions("-C 1.0 -L 0.0010 -P 1.0E-12 -N 0 -V -1 -W 1 -K \"weka.classifiers.functions.supportVector.PolyKernel -C 250007 -E 1.0\""));     // set the options
            // build the classifier model using train data
             scheme.buildClassifier(train);   
        } catch (Exception ex) {
            System.err.println("Classification process is error :" + ex);
        }

        //just if you wanna see the output of scheme classification
        System.out.println(scheme.toString() + "\n\n");

        //evaluation methods, 1 for cross validation and 2 for training-testing data, further info see the function
        Evaluation eval = Evaluate(2,train,test,scheme);
        //see file result to take the output.txt (in folder properties)
    }

    //read your file, p.s. weka is only for single label learning, that is why you must set the class index
    private static Instances ReadFile(String fileAddress) {
        Instances data = null;
        
        weka.core.converters.ConverterUtils.DataSource source;
        try {
            source = new weka.core.converters.ConverterUtils.DataSource(fileAddress);
            data = source.getDataSet();
            
            // setting class attribute in the last of attribute
            if(data.classIndex() == -1)
                data.setClassIndex(data.numAttributes() - 1);
        } catch (Exception ex) {
            System.err.println("data can't be read because of this error :" + ex);
        }
        
        return data;
    }

    //if you want to filter the instances
    private static Instances filtering(Instances data) {
        Instances newData = null;
        String[] options = new String[2];
        options[0] = "-R";
        options[1] = "1";
        
        weka.filters.unsupervised.attribute.Remove remove = new weka.filters.unsupervised.attribute.Remove();
        try {
            remove.setOptions(options);
            remove.setInputFormat(data);
            newData = weka.filters.Filter.useFilter(data, remove);
        } catch (Exception ex) {
            System.err.println("filtering is failed, error :" + ex);
        }       
        
        return newData;
    }
        
    //cross validation method, choose your status to check with the cross validation or training and testing data
    private static Evaluation Evaluate(int status, Instances data, Instances test, Classifier tree) throws Exception{
        Evaluation eval = null;
        try {
            eval = new Evaluation(data);
        } catch (Exception ex) {
                System.err.println("data can't be evaluated, further error info : " + ex);
        }
        try {
            if(status == 1){    //if you use crossvalidation 
                eval.crossValidateModel(tree, data, 10, new java.util.Random(1));        
            }
            else{       //if you use train-test validation
                eval.evaluateModel(tree, test);
                for(int c=0;c< data.classAttribute().numValues();c++)
                    System.out.print(data.classAttribute().value(c) + ";");
                System.out.println(";truthLabel;prediction");
                for (int i = 0; i < test.numInstances(); i++) {
                  double pred = tree.classifyInstance(test.instance(i));
                  double[] conf = tree.distributionForInstance(test.instance(i));
                  for(int k=0;k<conf.length;k++)
                    System.out.print(String.format("%.6f", conf[k]) + ";");
                  System.out.print(";" + test.classAttribute().value((int) test.instance(i).classValue()) + ";" + test.classAttribute().value((int) pred));
                  System.out.println("");
                }
            }
        } catch (Exception ex) {
            System.err.println("evaluation process is failed\n" +  ex );
        }
        
        System.out.println(eval.toSummaryString("\nResults\n===========\n", false));
        return eval;
    }
}
