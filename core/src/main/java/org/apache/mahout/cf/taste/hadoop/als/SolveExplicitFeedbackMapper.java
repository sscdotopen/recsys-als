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

import com.google.common.base.Preconditions;
import com.google.common.io.Closeables;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.LocalFileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

import org.apache.mahout.math.map.OpenIntObjectHashMap;

import java.io.IOException;

/** Solving mapper that can be safely executed using multiple threads */
public class SolveExplicitFeedbackMapper
    extends SharingMapper<IntWritable,VectorWritable,IntWritable,VectorWritable,OpenIntObjectHashMap<Vector>> {

  private double lambda;
  private int numFeatures;
  private final VectorWritable uiOrmj = new VectorWritable();

  @Override
  OpenIntObjectHashMap<Vector> createSharedInstance(Context ctx) throws IOException {

    Configuration conf = ctx.getConfiguration();

    IntWritable rowIndex = new IntWritable();
    VectorWritable row = new VectorWritable();

    LocalFileSystem localFs = FileSystem.getLocal(conf);
    Path[] cacheFiles = DistributedCache.getLocalCacheFiles(conf);

    int numEntities = conf.getInt(ParallelALSFactorizationJob.NUM_ENTITIES, -1);

    OpenIntObjectHashMap<Vector> featureMatrix = numEntities > 0
        ? new OpenIntObjectHashMap<Vector>(numEntities) : new OpenIntObjectHashMap<Vector>();

    for (int n = 0; n < cacheFiles.length; n++) {
      Path localCacheFile = localFs.makeQualified(cacheFiles[n]);

      // fallback for local execution
      if (!localFs.exists(localCacheFile)) {
        localCacheFile = new Path(DistributedCache.getCacheFiles(conf)[n].getPath());
      }

      SequenceFile.Reader reader = null;
      try {
        reader = new SequenceFile.Reader(localFs, localCacheFile, conf);
        while (reader.next(rowIndex, row)) {
          featureMatrix.put(rowIndex.get(), row.get());
        }
      } finally {
        Closeables.close(reader, true);
      }
    }

    Preconditions.checkState(!featureMatrix.isEmpty(), "Feature matrix is empty");
    return featureMatrix;
  }

  @Override
  protected void setup(Mapper.Context ctx) throws IOException, InterruptedException {
    lambda = Double.parseDouble(ctx.getConfiguration().get(ParallelALSFactorizationJob.LAMBDA));
    numFeatures = ctx.getConfiguration().getInt(ParallelALSFactorizationJob.NUM_FEATURES, -1);
    Preconditions.checkArgument(numFeatures > 0, "numFeatures was not set correctly!");
  }

  @Override
  protected void map(IntWritable userOrItemID, VectorWritable ratingsWritable, Context ctx)
    throws IOException, InterruptedException {
    OpenIntObjectHashMap<Vector> uOrM = getSharedInstance();
    uiOrmj.set(ALS.solveExplicit(ratingsWritable, uOrM, lambda, numFeatures));
    ctx.write(userOrItemID, uiOrmj);
  }

}
