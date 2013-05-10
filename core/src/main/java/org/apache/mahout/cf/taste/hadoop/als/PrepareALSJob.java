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

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.cf.taste.hadoop.TasteHadoopUtils;
import org.apache.mahout.cf.taste.impl.common.FullRunningAverage;
import org.apache.mahout.cf.taste.impl.common.RunningAverage;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.mapreduce.MergeVectorsCombiner;
import org.apache.mahout.common.mapreduce.MergeVectorsReducer;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;
import org.apache.mahout.math.function.Functions;

import java.io.IOException;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

public class PrepareALSJob extends AbstractJob {

  @Override
  public int run(String[] args) throws Exception {

    addInputOption();
    addOutputOption();

    Map<String,List<String>> parsedArgs = parseArguments(args);
    if (parsedArgs == null) {
      return -1;
    }

    /* create A' */
    Job itemRatings = prepareJob(getInputPath(), pathToItemRatings(),
        TextInputFormat.class, ItemRatingVectorsMapper.class, IntWritable.class,
        VectorWritable.class, VectorSumReducer.class, IntWritable.class,
        VectorWritable.class, SequenceFileOutputFormat.class);
    itemRatings.setCombinerClass(VectorSumCombiner.class);
    boolean succeeded = itemRatings.waitForCompletion(true);
    if (!succeeded) {
      return -1;
    }

    /* create A */
    Job userRatings = prepareJob(pathToItemRatings(), pathToUserRatings(),
        TransposeMapper.class, IntWritable.class, VectorWritable.class, MergeVectorsReducer.class, IntWritable.class,
        VectorWritable.class);
    userRatings.setCombinerClass(MergeVectorsCombiner.class);
    succeeded = userRatings.waitForCompletion(true);
    if (!succeeded) {
      return -1;
    }

    //TODO this could be fiddled into one of the upper jobs
    Job averageItemRatings = prepareJob(pathToItemRatings(), getOutputPath("averageRatings"),
        AverageRatingMapper.class, IntWritable.class, VectorWritable.class, MergeVectorsReducer.class,
        IntWritable.class, VectorWritable.class);
    averageItemRatings.setCombinerClass(MergeVectorsCombiner.class);
    succeeded = averageItemRatings.waitForCompletion(true);
    if (!succeeded) {
      return -1;
    }

    return 0;
  }

  private Path pathToItemRatings() {
    return getOutputPath("itemRatings");
  }

  private Path pathToUserRatings() {
    return getOutputPath("userRatings");
  }

  static class VectorSumCombiner
      extends Reducer<WritableComparable<?>, VectorWritable, WritableComparable<?>, VectorWritable> {

    private final VectorWritable vectorWritable = new VectorWritable();

    @Override
    protected void reduce(WritableComparable<?> key, Iterable<VectorWritable> values, Context ctx)
        throws IOException, InterruptedException {

      Iterator<VectorWritable> valueIterator = values.iterator();
      Vector vector = valueIterator.next().get();

      while (valueIterator.hasNext()) {
         vector.assign(valueIterator.next().get(), Functions.PLUS);
      }
      vectorWritable.set(vector);
      ctx.write(key, vectorWritable);
    }
  }

  static class VectorSumReducer
      extends Reducer<WritableComparable<?>, VectorWritable, WritableComparable<?>, VectorWritable> {

    private final VectorWritable vectorWritable = new VectorWritable();

    @Override
    protected void reduce(WritableComparable<?> key, Iterable<VectorWritable> values, Context ctx)
        throws IOException, InterruptedException {

      Iterator<VectorWritable> valueIterator = values.iterator();
      Vector vector = valueIterator.next().get();

      while (valueIterator.hasNext()) {
        vector.assign(valueIterator.next().get(), Functions.PLUS);
      }

      vectorWritable.set(new SequentialAccessSparseVector(vector));
      ctx.write(key, vectorWritable);
    }
  }

  static class TransposeMapper extends Mapper<IntWritable,VectorWritable,IntWritable,VectorWritable> {

    private final VectorWritable vectorWritable = new VectorWritable();

    @Override
    protected void map(IntWritable r, VectorWritable v, Context ctx) throws IOException, InterruptedException {

      RandomAccessSparseVector tmp = new RandomAccessSparseVector(Integer.MAX_VALUE, 1);

      int row = r.get();
      Iterator<Vector.Element> it = v.get().iterateNonZero();
      while (it.hasNext()) {
        Vector.Element e = it.next();

        tmp.setQuick(row, e.get());
        vectorWritable.set(tmp);
        r.set(e.index());
        ctx.write(r, vectorWritable);
        // prepare instance for reuse
        tmp.setQuick(row, 0.0);
      }
    }
  }

  static class ItemRatingVectorsMapper extends Mapper<LongWritable,Text,IntWritable,VectorWritable> {

    private final IntWritable itemIDWritable = new IntWritable();
    private final VectorWritable ratingsWritable = new VectorWritable(true);
    //private final Vector ratings = new SequentialAccessSparseVector(Integer.MAX_VALUE, 1);
    private final Vector ratings = new RandomAccessSparseVector(Integer.MAX_VALUE, 1);

    @Override
    protected void map(LongWritable offset, Text line, Context ctx) throws IOException, InterruptedException {
      String[] tokens = TasteHadoopUtils.splitPrefTokens(line.toString());
      int userID = Integer.parseInt(tokens[0]);
      int itemID = Integer.parseInt(tokens[1]);
      float rating = Float.parseFloat(tokens[2]);

      ratings.setQuick(userID, rating);

      itemIDWritable.set(itemID);
      ratingsWritable.set(ratings);

      ctx.write(itemIDWritable, ratingsWritable);

      // prepare instance for reuse
      ratings.setQuick(userID, 0.0d);
    }
  }

  static class AverageRatingMapper extends Mapper<IntWritable,VectorWritable,IntWritable,VectorWritable> {

    private final IntWritable firstIndex = new IntWritable(0);
    private final Vector featureVector = new RandomAccessSparseVector(Integer.MAX_VALUE, 1);
    private final VectorWritable featureVectorWritable = new VectorWritable();

    @Override
    protected void map(IntWritable r, VectorWritable v, Context ctx) throws IOException, InterruptedException {
      RunningAverage avg = new FullRunningAverage();
      Iterator<Vector.Element> elements = v.get().iterateNonZero();
      while (elements.hasNext()) {
        avg.addDatum(elements.next().get());
      }

      featureVector.setQuick(r.get(), avg.getAverage());
      featureVectorWritable.set(featureVector);
      ctx.write(firstIndex, featureVectorWritable);

      // prepare instance for reuse
      featureVector.setQuick(r.get(), 0.0d);
    }
  }

  public static void main(String[] args) throws Exception {
    ToolRunner.run(new PrepareALSJob(), args);
  }
}
