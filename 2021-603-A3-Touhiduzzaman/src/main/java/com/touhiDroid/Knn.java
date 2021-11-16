package com.touhiDroid;

import com.touhiDroid.models.ClassDistPair;
import com.touhiDroid.models.ListOfClassDistPairList;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;

public class Knn {
    private static final boolean toPrintDebugInfo = false;
    private static final int k = 3;
    private static ArrayList<List<Float>> testDataSet;
    /**
     * contains the count combined as 1-class-value & (numAttrs-1) features
     */
    private static int numAttrs = 0;

    public static void main(String[] args) throws IOException, InterruptedException, ClassNotFoundException {
        // args: 0->train, 1->test, 2->out
        readTestData(args[1]);

        long startTimeMs = System.currentTimeMillis();
        Job jobKnn = setupKnnJob(args);
        if (!jobKnn.waitForCompletion(true)) // jobKnn was not successful
            System.exit(1);

        long endTimeMs = System.currentTimeMillis();
        long delta = endTimeMs - startTimeMs;
        long deltaMinutes = delta / 1000 / 60;
        float deltaSeconds = delta / 1000.0f - deltaMinutes * 60.0f;
        // print accuracy   -> done as the last action of the single reducer
        p("Total Time required: " + deltaMinutes + " minutes & " + deltaSeconds + "seconds", true);
    }

    private static Job setupKnnJob(String[] args) throws IOException {
        Configuration config = new Configuration();

        Job jobKnn = Job.getInstance(config, "kNN");
        jobKnn.setJarByClass(Knn.class);

        jobKnn.setMapperClass(KnnMapper.class);
        // jobKnn.setCombinerClass();
        jobKnn.setReducerClass(KnnReducer.class);
        jobKnn.setNumReduceTasks(1);

        jobKnn.setOutputKeyClass(LongWritable.class);
        jobKnn.setOutputValueClass(ListOfClassDistPairList.class);

        FileInputFormat.addInputPath(jobKnn, new Path(args[0]));
        FileOutputFormat.setOutputPath(jobKnn, new Path(args[2]));
        p("setupKnnJob -> DONE for args: 0->" + args[0] + ", 1->" + args[1] + ", 2->" + args[2]);
        return jobKnn;
    }

    private static void readTestData(String arg) {
        // read testDataSet line-by-line
        numAttrs = 0;
        testDataSet = new ArrayList<>();
        try (Stream<String> stream = Files.lines(Paths.get(arg))) {
            stream.forEach(testDataLine -> {
                if (testDataLine.startsWith("@")) {
                    if (testDataLine.startsWith("@attribute"))
                        numAttrs++;
                    return;
                }
                String[] tokens = testDataLine.split("[\\s,]+");
                List<Float> featureList = new ArrayList<>();
                for (String s : tokens)
                    try {
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
    }

    static void p(String msg) {
        p(msg, false);
    }

    static void p(String msg, boolean mustPrint) {
        if (toPrintDebugInfo || mustPrint)
            System.out.println(msg);
    }

    private static class KnnReducer extends Reducer<Object, ListOfClassDistPairList, Object, IntWritable> {
        private List<List<ClassDistPair>> cdReducer;
        private int testInstCount = 0;

        @Override
        protected void setup(Context context) throws IOException, InterruptedException {
            super.setup(context);

            testInstCount = testDataSet.size();
            cdReducer = new ArrayList<>();
            for (int i = 0; i < testInstCount; i++) {
                List<ClassDistPair> cdTemp = new ArrayList<>();
                for (int n = 0; n < k; n++)
                    cdTemp.add(new ClassDistPair(-1, Float.MAX_VALUE));
                cdReducer.add(cdTemp);
            }
            p("KnnReducer() -> CDReducer init. with size = " + cdReducer.size());
            p("KnnReducer() -> Test Instance Count = " + testInstCount);
        }

        @Override
        protected void reduce(Object key, Iterable<ListOfClassDistPairList> itCDj, Context context)
                throws IOException, InterruptedException {
            // List<List<MapOut>> cdj = new ArrayList<>();
            // for (List<List<MapOut>> list : itCDj) cdj.addAll(list); // all mapper outputs are merged into one CDj

            /* ** Reduce Operation : reducing all cdj into one cdReducer ** */
            for (ListOfClassDistPairList cdjObj : itCDj) {
                List<List<ClassDistPair>> cdj = cdjObj.getList();
                for (int i = 0; i < testInstCount; i++) {
                    int cont = 0;
                    for (int n = 0; n < k; n++) {
                        float d = cdj.get(i).get(cont).getDistance();
                        float d2 = cdReducer.get(i).get(n).getDistance();
                        if (d < d2) {
                            String inStr = i + ", " + n;
                            p("Updating distance (" + inStr + ") = " + d
                                    + " --- vs cdReducer(" + inStr + ") = " + d2
                                    + ", class=" + cdReducer.get(i).get(n).getClassValue());
                            cdReducer.get(i).set(n, cdj.get(i).get(cont));
                            cont++;
                        }
                    }
                }
            }
        }

        @Override
        protected void cleanup(Context context) throws IOException, InterruptedException {
            super.cleanup(context);
            /* ** Reduce cleanup ** */
            int correctPredictions = 0;
            // IntWritable[] predictedClasses = new IntWritable[testInstCount];
            for (int i = 0; i < testInstCount; i++) {
                // feeding majorityVoting() all k-NNs for the i'th case to get the predicted-class
                List<ClassDistPair> kNNs = cdReducer.get(i);
                int predictedClass = majorityVoting(kNNs);
                p("Voting for test#" + i + ": cdReducer[0,1,2] = ["
                        + kNNs.get(0) + ", " + kNNs.get(1) + ", " + kNNs.get(2) + "] -> " + predictedClass);
                context.write(i, new IntWritable(predictedClass));

                // for accuracy computation
                int originalClass = testDataSet.get(i).get(numAttrs - 1).intValue();
                if (predictedClass == originalClass)
                    correctPredictions++;
            }
            p("Correct Predictions: " + correctPredictions
                    + ", Total Test Cases: " + testInstCount
                    + ", Accuracy: " + ((correctPredictions * 100.0) / (testInstCount * 1.0)) + "%\n", true);
        }

        private int majorityVoting(List<ClassDistPair> reducedKNeighbourList) {
            // voting on class by k-NNs
            Map<Integer, Integer> classVoteMap = new HashMap<>();
            for (ClassDistPair n : reducedKNeighbourList) {
                classVoteMap.put(n.getClassValue(), classVoteMap.getOrDefault(n.getClassValue(), 0) + 1);
            }
            // find max-vote getter class
            Integer predictedClass = -1, maxVotes = -1;
            for (Map.Entry<Integer, Integer> e : classVoteMap.entrySet()) {
                if (e.getValue() > maxVotes) {
                    maxVotes = e.getValue();
                    predictedClass = e.getKey();
                }
            }
            return predictedClass;
        }
    }

    private static class KnnMapper extends Mapper<Object, Text, Object, ListOfClassDistPairList> {
        @Override
        protected void map(Object key, Text trainDataLinesParam, Context context)
                throws IOException, InterruptedException {
            String trainDataLines = trainDataLinesParam.toString();
            p("key: " + key + ", Value: " + trainDataLines);
            if (trainDataLines.startsWith("@")) {
                /*String attrIdentifier = "@attribute";
                int idx = 0, aiSz = attrIdentifier.length();
                while (trainDataLines.indexOf(attrIdentifier, idx) > -1) {
                    idx += aiSz;
                    numAttrs++;
                }*/
                return;
            }

            String[] trainTokens = trainDataLines.split("[\\s,]+");
            // int numTrainData = trainTokens.length / numAttrs; // since features & one class value denotes each train-set
            List<List<Float>> trainSubSet = new ArrayList<>();
            for (int i = 0; i < trainTokens.length; i++) {
                List<Float> trCase = new ArrayList<>();
                for (int j = 0; i < trainTokens.length && j < numAttrs; j++, i++)
                    try {
                        trCase.add(Float.parseFloat(trainTokens[i]));
                    } catch (NumberFormatException nfe) {
                        nfe.printStackTrace();
                    }
                trainSubSet.add(trCase);
            }
            p("KnnMapper.map() -> Read train-subset size = " + trainSubSet.size());

            // int classVal = Integer.parseInt(trainSubSet[t * (numAttrs+1)+numAttrs]); // <- the class of train-data #t
            // int testInstCount = testDataSet.size();
            List<List<ClassDistPair>> cdj = new ArrayList<>();
            // TO-DO Compute CDj
            for (List<Float> testInstance : testDataSet) {
                List<ClassDistPair> singleTestKnnList = kNN(testInstance, trainSubSet); // -> to implement this method, follow codes in sequential.cpp
                cdj.add(singleTestKnnList);
            }
            p("KnnMapper.map() -> Writing CDj of size: " + cdj.size());
            context.write(key, new ListOfClassDistPairList(cdj));
        }

        private List<ClassDistPair> kNN(List<Float> testInstance, List<List<Float>> trainSubSet) {
            final List<ClassDistPair> kNeighbourList = new ArrayList<>(); // <- k-NNs are listed in ascending distance order
            for (int i = 0; i < Knn.k; i++)
                kNeighbourList.add(new ClassDistPair(0, Float.MAX_VALUE));

            for (List<Float> trainInstance : trainSubSet) {
                float distance = distance(numAttrs, testInstance, trainInstance);

                // insert kNNs as per distance's ascending order
                int sz = trainInstance.size();
                for (int n = 0; n < Knn.k; n++) {
                    if (distance < kNeighbourList.get(n).getDistance()) {
                        // move nth item downward & set this dist.+class as the nth item <- ascending order being guaranteed
                        for (int x = Knn.k - 1; x > n; x--)
                            kNeighbourList.set(x, kNeighbourList.get(x - 1));
                        kNeighbourList.set(n, new ClassDistPair(trainInstance.get(sz - 1).intValue(), distance));
                        break;
                    }
                }
            }
            // the returned list contains the k-NN of the given test-case against the sub-set of
            // the training data as was given to this mapper by Hadoop
            return kNeighbourList;
        }

        private float distance(int size, List<Float> a, List<Float> b) {
            float sum = 0;
            for (int i = 0; i < size; i++) {
                float diff = (a.get(i) - b.get(i));
                sum += diff * diff;
            }
            return sum;
        }
    }
}
