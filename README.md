## Experiments

Here we provide instructions and code to repeat the experiments from *Distributed Matrix Factorization with MapReduce using a series of Broadcast-Joins*

### Datasets

 *Netflix* is no longer officially distributed, but can be found via bit-torrent

*R2 - Yahoo! Music User Ratings of Songs with Artist, Album, and Genre Meta Information, v. 1.0 (1.4 Gbyte & 1.1 Gbyte)* is available from [Yahoo Webscope](http://webscope.sandbox.yahoo.com/catalog.php?datatype=r)

*Bigflix* has to be created using the [Myriad data generation toolkit](myriad-toolkit.com) with our provided [configuration files](https://github.com/sscdotopen/recsys-als/blob/trunk/myriad-config.zip). 

In order to generate the 25M users version of Bigflix, run this command after installing myriad 

    bin/myriad-recsys-node -s 25000

### Setup

 Setup a cluster with an [Apache Hadoop](http://hadoop.apache.org) installation, we used version 1.0.4.

As we leverage [JBlas](http://mikiobraun.github.io/jblas/) for solving the linear equations involved in ALS, your cluster machines need to have the [necessary fortran libraries](https://github.com/mikiobraun/jblas/wiki/Missing-Libraries)  installed.

The datasets need to be converted to text files and copied to HDFS. Each line must have the following format:  
  
    userID[TAB]itemID[TAB]rating


Clone and compile the version of Mahout from this github repository. Create the jar containing the MapReduce code in *core/target/mahout-core-0.8-SNAPSHOT-job.jar* using:

    mvn -DskipTests clean package

### Compute a distributed factorization

 Prepare your dataset as following: 

    hadoop jar mahout-core-0.8-SNAPSHOT-job.jar   
      org.apache.mahout.cf.taste.hadoop.als.PrepareALSJob  
      --input /path/to/dataset 
      --output /preparedversion/of/dataset/

Finally, you can trigger the computation of a factorization as follows:

    hadoop jar mahout-core-0.8-SNAPSHOT-job.jar 
      org.apache.mahout.cf.taste.hadoop.als.ParallelALSFactorizationJob
      --input /preparedversion/of/dataset/
      --output /path/to/results
      --tempDir /some/temp/path
      --numFeatures <numFeatures>
      --numIterations <numIterations> 
      --lambda <lambda> 
      --numThreadsPerSolver <numCoresToUsePerMapper> 
      --skipPreprocessing true
      --numUsers <numUsersInDataset> 
      --numItems <numItemsInDataset>
