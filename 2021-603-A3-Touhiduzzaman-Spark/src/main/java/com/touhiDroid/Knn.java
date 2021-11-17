package com.touhiDroid;

import com.touhiDroid.models.ClassDistPair;
import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.util.MLUtils;

import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

public class Knn {
    private static final boolean toPrintDebugInfo = true;
    private static final int k = 3;
    // private static ArrayList<List<Float>> testDataSet;

    public static void main(String[] args) {
        SparkConf conf = new SparkConf().setAppName("IS-kNN").set("spark.master", "local[4]");
        SparkContext sc = new SparkContext(conf);

        // readTestDataSvm(args[1]);

        long startTimeMs = System.currentTimeMillis();

        // Initiate Spark Map operation
        JavaRDD<LabeledPoint> trainingData = MLUtils.loadLibSVMFile(sc, args[0]).toJavaRDD();
        JavaRDD<LabeledPoint> testData = MLUtils.loadLibSVMFile(sc, args[1]).toJavaRDD();

        trainingData.mapPartitions(
                (FlatMapFunction<Iterator<LabeledPoint>, Map.Entry<Integer, List<ClassDistPair>>>) labeledPointIterator -> {
                    // Algo-1 : Map function
                    List<LabeledPoint> trainSubSetJ = new ArrayList<>();
                    while (labeledPointIterator.hasNext())
                        trainSubSetJ.add(labeledPointIterator.next());

                    // CD(t,j) as a [testInst * k] sized list
                    Map<Integer, List<ClassDistPair>> resultJ = new HashMap<>();
                    AtomicInteger t = new AtomicInteger();
                    testData.foreach(testCaseI -> {
                        List<ClassDistPair> cdTJ = kNN(testCaseI, trainSubSetJ);
                        t.getAndIncrement();
                        resultJ.put(t.get(), cdTJ);
                    });
                    return resultJ.entrySet().iterator();
                });


        long endTimeMs = System.currentTimeMillis();
        long delta = endTimeMs - startTimeMs;
        long deltaMinutes = delta / 1000 / 60;
        float deltaSeconds = delta / 1000.0f - deltaMinutes * 60.0f;
        // print accuracy   -> done as the last action of the single reducer
        p("Total Time required: " + deltaMinutes + " minutes & " + deltaSeconds + "seconds", true);
    }

    static void p(String msg) {
        p(msg, false);
    }

    static void p(String msg, boolean mustPrint) {
        if (toPrintDebugInfo || mustPrint)
            System.out.println(msg);
    }

    private static double distance(double[] a, double[] b) {
        double sum = 0;
        int size = a.length;

        for (int i = 0; i < size; i++) {
            double diff = (a[i] - b[i]);
            sum += diff * diff;
        }
        return sum;
    }

    private static List<ClassDistPair> kNN(LabeledPoint testInstance, List<LabeledPoint> trainSubSet) {
        final List<ClassDistPair> kNeighbourList = new ArrayList<>(); // <- k-NNs are listed in ascending distance order

        for (int i = 0; i < Knn.k; i++)
            kNeighbourList.add(new ClassDistPair(0, Float.MAX_VALUE));

        for (LabeledPoint trainInstance : trainSubSet) {
            double[] trainItemFeatures = trainInstance.features().toArray();
            double[] testItemFeatures = testInstance.features().toArray();
            double distance = Knn.distance(testItemFeatures, trainItemFeatures);

            // insert kNNs as per distance's ascending order
            int sz = trainItemFeatures.length;
            for (int n = 0; n < Knn.k; n++) {
                if (distance < kNeighbourList.get(n).getDistance()) {
                    // move nth item downward & set this dist.+class as the nth item <- ascending order being guaranteed
                    for (int x = Knn.k - 1; x > n; x--)
                        kNeighbourList.set(x, kNeighbourList.get(x - 1));
                    kNeighbourList.set(n, new ClassDistPair((float) trainInstance.getLabel(), (float) distance));
                    break;
                }
            }
        }
        // the returned list contains the k-NN of the given test-case against the sub-set of
        // the training data as was given to this mapper by Hadoop
        return kNeighbourList;
    }

    /*private static void readTestDataSvm(String arg) {
        // read testDataSet line-by-line
        numAttrs = 0;
        testDataSet = new ArrayList<>();
        try (Stream<String> stream = Files.lines(Paths.get(arg))) {
            stream.forEach(testDataLine -> {
                String[] tokens = testDataLine.split("[\\s,]+");
                List<Float> featureList = new ArrayList<>();
                for (String s : tokens)
                    try {
                        if (s.contains(":")) s = s.split("[:]+")[1];
                        featureList.add(Float.parseFloat(s));
                    } catch (NumberFormatException nfe) {
                        featureList.add(0.0f); // any better fallback to keep the attrs+class count fixed?
                        nfe.printStackTrace();
                    }
                testDataSet.add(featureList);
            });
        } catch (IOException ie) {
            ie.printStackTrace();
        }
        p("readTestData -> Total test cases: " + testDataSet.size()
                + ", Number of Attributes: " + numAttrs);
    }*/
}
