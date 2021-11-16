package com.touhiDroid.models;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Writable;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class PredictionsWritable implements Writable {
    private List<IntWritable> predictions;

    public PredictionsWritable() {
        this.predictions = new ArrayList<>();
    }

    public PredictionsWritable(List<IntWritable> predictions) {
        this.predictions = predictions;
    }

    public PredictionsWritable(IntWritable[] predictedClasses) {
        int l = predictedClasses.length;
        this.predictions = new ArrayList<>(l);
        this.predictions.addAll(Arrays.asList(predictedClasses));
    }

    @Override
    public void write(DataOutput dataOutput) throws IOException {
        int size = predictions.size();
        dataOutput.writeInt(size);
        for (IntWritable prediction : predictions) {
            prediction.write(dataOutput);
        }
    }

    @Override
    public void readFields(DataInput dataInput) throws IOException {
        int size = dataInput.readInt();
        this.predictions = new ArrayList<>(size);
        for (int i = 0; i < size; i++) {
            IntWritable iw = new IntWritable();
            iw.readFields(dataInput);
            predictions.add(iw);
        }
    }

    public List<IntWritable> getPredictions() {
        return predictions;
    }

    public void setPredictions(List<IntWritable> predictions) {
        this.predictions = predictions;
    }

    @Override
    public String toString() {
        int size = this.predictions.size();
        StringBuilder s = new StringBuilder("PredictionsWritable{ predictions { size=" + size + " } = ");
        for (IntWritable prediction : this.predictions) {
            s.append(prediction.get()).append(", ");
        }
        s.append(" }");
        return s.toString();
    }
}
