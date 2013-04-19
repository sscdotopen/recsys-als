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

package org.apache.mahout.math.als;

import com.google.common.base.Preconditions;
import com.google.common.collect.Iterables;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.jblas.DoubleMatrix;
import org.jblas.Solve;

import java.util.Iterator;

/**
 * See
 * <a href="http://www.hpl.hp.com/personal/Robert_Schreiber/papers/2008%20AAIM%20Netflix/netflix_aaim08(submitted).pdf">
 * this paper.</a>
 */
public final class JBlasAlternatingLeastSquaresSolver {

  private JBlasAlternatingLeastSquaresSolver() {}

  //TODO make feature vectors a simple array
  public static Vector solve(Iterable<Vector> featureVectors, Vector ratingVector, double lambda, int numFeatures) {

    Preconditions.checkNotNull(featureVectors, "Feature vectors cannot be null");
    Preconditions.checkArgument(!Iterables.isEmpty(featureVectors));
    Preconditions.checkNotNull(ratingVector, "rating vector cannot be null");
    Preconditions.checkArgument(ratingVector.getNumNondefaultElements() > 0, "Rating vector cannot be empty");
    Preconditions.checkArgument(Iterables.size(featureVectors) == ratingVector.getNumNondefaultElements());

    int nui = ratingVector.getNumNondefaultElements();

    DoubleMatrix MiIi = createMiIi(featureVectors, numFeatures);
    DoubleMatrix RiIiMaybeTransposed = createRiIiMaybeTransposed(ratingVector);

    /* compute Ai = MiIi * t(MiIi) + lambda * nui * E */
    DoubleMatrix Ai = miTimesMiTransposePlusLambdaTimesNuiTimesE(MiIi, lambda, nui);

    /* compute Vi = MiIi * t(R(i,Ii)) */
    DoubleMatrix Vi = MiIi.mmul(RiIiMaybeTransposed);

    /* compute Ai * ui = Vi */
    return solve(Ai, Vi);
  }

  private static Vector solve(DoubleMatrix Ai, DoubleMatrix Vi) {
    return new DenseVector(Solve.solveSymmetric(Ai, Vi).data, true);
  }

  private static DoubleMatrix miTimesMiTransposePlusLambdaTimesNuiTimesE(DoubleMatrix MiIi, double lambda, int nui) {

    double lambdaTimesNui = lambda * nui;

    DoubleMatrix Ai = MiIi.mmul(MiIi.transpose());

    for (int n = 0; n < MiIi.rows; n++) {
      Ai.put(n, n, Ai.get(n, n) + lambdaTimesNui);
    }

    return Ai;
  }

  static DoubleMatrix createMiIi(Iterable<Vector> featureVectors, int numFeatures) {
    int numRatings = Iterables.size(featureVectors);
    DoubleMatrix MiIi = new DoubleMatrix(numFeatures, numRatings);
    int n = 0;
    for (Vector featureVector : featureVectors) {
      for (int m = 0; m < numFeatures; m++) {
        MiIi.put(m, n, featureVector.getQuick(m));
      }
      n++;
    }

    return MiIi;
  }

  static DoubleMatrix createRiIiMaybeTransposed(Vector ratingVector) {
    Preconditions.checkArgument(ratingVector.isSequentialAccess());

    int numRatings = ratingVector.getNumNondefaultElements();
    double[] RiIiMaybeTransposed = new double[numRatings];
    Iterator<Vector.Element> ratingsIterator = ratingVector.iterateNonZero();
    int index = 0;
    while (ratingsIterator.hasNext()) {
      Vector.Element elem = ratingsIterator.next();
      RiIiMaybeTransposed[index++] = elem.get();
    }
    return new DoubleMatrix(numRatings, 1, RiIiMaybeTransposed);
  }

}
