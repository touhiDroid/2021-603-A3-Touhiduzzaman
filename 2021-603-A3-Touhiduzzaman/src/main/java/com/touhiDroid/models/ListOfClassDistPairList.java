package com.touhiDroid.models;

import org.apache.hadoop.io.Writable;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class ListOfClassDistPairList implements Writable {
    private List<List<ClassDistPair>> list;


    public ListOfClassDistPairList() {
        this.list = new ArrayList<>();
    }

    public ListOfClassDistPairList(List<List<ClassDistPair>> cdj) {
        this.list = cdj;
    }

    @Override
    public void write(DataOutput dataOutput) throws IOException {
        int size = list.size();
        dataOutput.writeInt(size);
        for (int i = 0; i < size; i++) {
            int innerSize = list.get(i).size();
            dataOutput.writeInt(innerSize);
            for (int j = 0; j < innerSize; j++)
                list.get(i).get(j).write(dataOutput);
        }
    }

    @Override
    public void readFields(DataInput dataInput) throws IOException {
        int size = dataInput.readInt();
        this.list = new ArrayList<>(size);
        for (int i = 0; i < size; i++) {
            int innerSize = dataInput.readInt();
            List<ClassDistPair> innerList = new ArrayList<>(innerSize);
            for (int j = 0; j < innerSize; j++) {
                ClassDistPair cd = new ClassDistPair();
                cd.readFields(dataInput);
                innerList.add(cd);
            }
            this.list.add(innerList);
        }
    }

    public List<List<ClassDistPair>> getList() {
        return list;
    }

    public void setList(List<List<ClassDistPair>> list) {
        this.list = list;
    }
}
