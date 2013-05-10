/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.mahout.cf.taste.hadoop.als;

import com.google.common.io.Closeables;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.map.MultithreadedMapper;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.common.iterator.sequencefile.PathFilters;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * <p>MapReduce implementation of the two factorization algorithms described in
 *
 * <p>"Large-scale Parallel Collaborative Filtering for the Netﬂix Prize" available at
 * http://www.hpl.hp.com/personal/Robert_Schreiber/papers/2008%20AAIM%20Netflix/netflix_aaim08(submitted).pdf.</p>
 *
 * "<p>Collaborative Filtering for Implicit Feedback Datasets" available at
 * http://research.yahoo.com/pub/2433</p>
 *
 * </p>
 * <p>Command line arguments specific to this class are:</p>
 *
 * <ol>
 * <li>--input (path): Directory containing one or more text files with the dataset</li>
 * <li>--output (path): path where output should go</li>
 * <li>--lambda (double): regularization parameter to avoid overfitting</li>
 * <li>--userFeatures (path): path to the user feature matrix</li>
 * <li>--itemFeatures (path): path to the item feature matrix</li>
 * <li>--numThreadsPerSolver (int): threads to use per solver mapper, (default: 1)</li>
 * </ol>
 */
public class ParallelALSFactorizationJob extends AbstractJob {

  private static final Logger log = LoggerFactory.getLogger(ParallelALSFactorizationJob.class);

  static final String NUM_FEATURES = ParallelALSFactorizationJob.class.getName() + ".numFeatures";
  static final String LAMBDA = ParallelALSFactorizationJob.class.getName() + ".lambda";
  static final String ALPHA = ParallelALSFactorizationJob.class.getName() + ".alpha";
  static final String FEATURE_MATRIX = ParallelALSFactorizationJob.class.getName() + ".featureMatrix";
  static final String NUM_ENTITIES = ParallelALSFactorizationJob.class.getName() + ".numEntities";

  private boolean implicitFeedback;
  private int numIterations;
  private int numFeatures;
  private double lambda;
  private double alpha;
  private int numThreadsPerSolver;

  private Path preprocessedInputPath;

  private int numUsers = -1;
  private int numItems = -1;

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new ParallelALSFactorizationJob(), args);
  }

  @Override
  public int run(String[] args) throws Exception {

    addInputOption();
    addOutputOption();
    addOption("lambda", null, "regularization parameter", true);
    addOption("implicitFeedback", null, "data consists of implicit feedback?", String.valueOf(false));
    addOption("alpha", null, "confidence parameter (only used on implicit feedback)", String.valueOf(40));
    addOption("numFeatures", null, "dimension of the feature space", true);
    addOption("numIterations", null, "number of iterations", true);
    addOption("numThreadsPerSolver", null, "threads per solver mapper", String.valueOf(1));

    addOption("skipPreprocessing", null, "", String.valueOf(false));

    addOption("numUsers", null, "", false);
    addOption("numItems", null, "", false);

    Map<String,List<String>> parsedArgs = parseArguments(args);
    if (parsedArgs == null) {
      return -1;
    }

    numFeatures = Integer.parseInt(getOption("numFeatures"));
    numIterations = Integer.parseInt(getOption("numIterations"));
    lambda = Double.parseDouble(getOption("lambda"));
    alpha = Double.parseDouble(getOption("alpha"));
    implicitFeedback = Boolean.parseBoolean(getOption("implicitFeedback"));

    numThreadsPerSolver = Integer.parseInt(getOption("numThreadsPerSolver"));

    boolean skipPreprocessing = Boolean.parseBoolean(getOption("skipPreprocessing"));

    if (hasOption("numUsers")) {
      numUsers = Integer.parseInt(getOption("numUsers"));
    }

    if (hasOption("numItems")) {
      numItems = Integer.parseInt(getOption("numItems"));
    }

    /*
    * compute the factorization A = U M'
    *
    * where A (users x items) is the matrix of known ratings
    *           U (users x features) is the representation of users in the feature space
    *           M (items x features) is the representation of items in the feature space
    */

    if (skipPreprocessing) {
      preprocessedInputPath = getInputPath();
    } else {
      preprocessedInputPath = getTempPath();
      ToolRunner.run(getConf(), new PrepareALSJob(), new String[] {
          "--input", getInputPath().toString(),
          "--output", preprocessedInputPath.toString() });
    }

    Vector averageRatings = ALS.readFirstRow(new Path(preprocessedInputPath, "averageRatings"), getConf());

    /* create an initial M */
    initializeM(averageRatings);

    for (int currentIteration = 0; currentIteration < numIterations; currentIteration++) {
      /* broadcast M, read A row-wise, recompute U row-wise */
      log.info("Recomputing U (iteration {}/{})", currentIteration, numIterations);
      runSolver(pathToUserRatings(), pathToU(currentIteration), pathToM(currentIteration - 1), currentIteration, "U",
                numItems);
      /* broadcast U, read A' row-wise, recompute M row-wise */
      log.info("Recomputing M (iteration {}/{})", currentIteration, numIterations);
      runSolver(pathToItemRatings(), pathToM(currentIteration), pathToU(currentIteration), currentIteration, "M",
                numUsers);
    }

    return 0;
  }

  private void initializeM(Vector averageRatings) throws IOException {
    Random random = RandomUtils.getRandom();

    FileSystem fs = FileSystem.get(pathToM(-1).toUri(), getConf());
    SequenceFile.Writer writer = null;
    try {
      writer = new SequenceFile.Writer(fs, getConf(), new Path(pathToM(-1), "part-m-00000"), IntWritable.class,
          VectorWritable.class);

      IntWritable index = new IntWritable();
      VectorWritable featureVector = new VectorWritable();

      Iterator<Vector.Element> averages = averageRatings.iterateNonZero();
      while (averages.hasNext()) {
        Vector.Element e = averages.next();
        Vector row = new DenseVector(numFeatures);
        row.setQuick(0, e.get());
        for (int m = 1; m < numFeatures; m++) {
          row.setQuick(m, random.nextDouble());
        }
        index.set(e.index());
        featureVector.set(row);
        writer.append(index, featureVector);
      }
    } finally {
      Closeables.closeQuietly(writer);
    }
  }



  private void runSolver(Path ratings, Path output, Path pathToUorM, int currentIteration, String matrixName,
    int numEntities) throws ClassNotFoundException, IOException, InterruptedException {

    int iterationNumber = currentIteration + 1;
    Class<? extends Mapper<IntWritable,VectorWritable,IntWritable,VectorWritable>> solverMapperClassInternal;
    String name;

    if (implicitFeedback) {
      solverMapperClassInternal = SolveImplicitFeedbackMapper.class;
      name = "Recompute " + matrixName + ", iteration (" + iterationNumber + "/" + numIterations + "), "
          + "(" + numThreadsPerSolver + " threads, implicit feedback)";
    } else {
      solverMapperClassInternal = SolveExplicitFeedbackMapper.class;
      name = "Recompute " + matrixName + ", iteration (" + iterationNumber + "/" + numIterations + "), "
          + "(" + numThreadsPerSolver + " threads, explicit feedback)";
    }

    Job solverForUorI = prepareJob(ratings, output, SequenceFileInputFormat.class, MultithreadedSharingMapper.class,
        IntWritable.class, VectorWritable.class, SequenceFileOutputFormat.class, name);
    Configuration solverConf = solverForUorI.getConfiguration();
    solverConf.set(LAMBDA, String.valueOf(lambda));
    solverConf.set(ALPHA, String.valueOf(alpha));
    solverConf.setInt(NUM_FEATURES, numFeatures);
    solverConf.setInt(NUM_ENTITIES, numEntities);
    solverConf.set(FEATURE_MATRIX, pathToUorM.toString());


    MultithreadedMapper.setMapperClass(solverForUorI, solverMapperClassInternal);
    MultithreadedMapper.setNumberOfThreads(solverForUorI, numThreadsPerSolver);

    boolean succeeded = solverForUorI.waitForCompletion(true);
    if (!succeeded) {
      throw new IllegalStateException("Job failed!");
    }
  }

  private void distributeViaCache(Path pathToUorM, Configuration conf) throws IOException {
    FileSystem fs = FileSystem.get(pathToUorM.toUri(), conf);
    FileStatus[] parts = fs.listStatus(pathToUorM, PathFilters.partFilter());
    for (FileStatus part : parts) {
      if (log.isDebugEnabled()) {
        log.debug("Adding {} to distributed cache", part.getPath().toString());
      }
      DistributedCache.addCacheFile(part.getPath().toUri(), conf);
    }
  }



  private Path pathToM(int iteration) {
    return iteration == numIterations - 1 ? getOutputPath("M") : getTempPath("M-" + iteration);
  }

  private Path pathToU(int iteration) {
    return iteration == numIterations - 1 ? getOutputPath("U") : getTempPath("U-" + iteration);
  }

  private Path pathToItemRatings() {
    return new Path(preprocessedInputPath, "itemRatings");
  }

  private Path pathToUserRatings() {
    return new Path(preprocessedInputPath, "userRatings");
  }
}
