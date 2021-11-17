spark-submit --class com.touhiDroid.Knn --master local[4] target/Spark-ML-1.0.jar datasets/small-train.arff datasets/small-test.arff knnSparkOut
