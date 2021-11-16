package com.touhiDroid.models;

import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Writable;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

public class ClassDistPair implements Writable {
    private final IntWritable classValue;
    private final FloatWritable distance;

    public ClassDistPair() {
        this.classValue = new IntWritable(0);
        this.distance = new FloatWritable(Float.MAX_VALUE);
    }

    public ClassDistPair(int classValue, float distance) {
        this.classValue = new IntWritable(classValue);
        this.distance = new FloatWritable(distance);
    }

    public ClassDistPair(IntWritable classValue, FloatWritable distance) {
        this.classValue = classValue;
        this.distance = distance;
    }

    public int getClassValue() {
        return classValue.get();
    }

    public void setClassValue(int classValue) {
        this.classValue.set(classValue);
    }

    public float getDistance() {
        return distance.get();
    }

    public void setDistance(float distance) {
        this.distance.set(distance);
    }

    @Override
    public void write(DataOutput dataOutput) throws IOException {
        classValue.write(dataOutput);
        distance.write(dataOutput);
    }

    @Override
    public void readFields(DataInput dataInput) throws IOException {
        classValue.readFields(dataInput);
        distance.readFields(dataInput);
    }
}
