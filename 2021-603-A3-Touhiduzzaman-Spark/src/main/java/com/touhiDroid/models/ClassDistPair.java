package com.touhiDroid.models;


import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

public class ClassDistPair {// implements Parcel {
    private float classValue;
    private float distance;

    public ClassDistPair() {
        this.classValue = -1 ;// new IntWritable(-1);
        this.distance = Float.MAX_VALUE; // new FloatWritable(Float.MAX_VALUE);
    }

    public ClassDistPair(float classValue, float distance) {
        this.classValue = classValue; // new IntWritable(classValue);
        this.distance = distance; // new FloatWritable(distance);
    }

    /*public ClassDistPair(IntWritable classValue, FloatWritable distance) {
        this.classValue = classValue;
        this.distance = distance;
    }*/

    public float getClassValue() {
        return classValue;
    }

    public void setClassValue(float classValue) {
        this.classValue = (classValue);
    }

    public float getDistance() {
        return distance;
    }

    public void setDistance(float distance) {
        this.distance = (distance);
    }

    /*@Override
    public void write(DataOutput dataOutput) throws IOException {
        classValue.write(dataOutput);
        distance.write(dataOutput);
    }

    @Override
    public void readFields(DataInput dataInput) throws IOException {
        classValue.readFields(dataInput);
        distance.readFields(dataInput);
    }*/

    @Override
    public String toString() {
        return "ClassDistPair{" +
                "class=" + classValue +
                ", distance=" + distance +
                '}';
    }
}
