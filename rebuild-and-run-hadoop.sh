mvn clean -f pom.xml
mvn package -f pom.xml
rm -R knnOut
hadoop jar target/2021-603-A3-Touhiduzzaman-1.0.0.jar com.touhiDroid.Knn datasets/small-train.arff datasets/small-test.arff knnOut -Dmapred.reduce.tasks=1
