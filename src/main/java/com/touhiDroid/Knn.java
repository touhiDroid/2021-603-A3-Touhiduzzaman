package com.touhiDroid;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;

import java.io.IOException;

public class Knn {

    private static class KnnMapper extends Mapper<Object, Object, Object, Object> {
        @Override
        protected void map(Object key, Object value, Mapper<Object, Object, Object, Object>.Context context)
                throws IOException, InterruptedException {
            super.map(key, value, context);
        }
    }


    public static void main(String[] args) throws IOException {
        Configuration config = new Configuration();
        Job jobKnn = Job.getInstance(config, "kNN");
        jobKnn.setJarByClass(Knn.class);
        jobKnn.setMapperClass(KnnMapper.class);
        // jobKnn.setReducerClass();
    }
}
