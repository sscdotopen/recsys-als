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

import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.MahoutTestCase;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.SequentialAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.jblas.DoubleMatrix;
import org.junit.Test;

import java.util.Arrays;

public class JblasAlternatingLeastSquaresSolverTest extends MahoutTestCase {

  @Test
  public void createMiIi() {
    Vector f1 = new DenseVector(new double[] { 1, 2, 3 });
    Vector f2 = new DenseVector(new double[] { 4, 5, 6 });

    DoubleMatrix miIi = JBlasAlternatingLeastSquaresSolver.createMiIi(Arrays.asList(f1, f2), 3);

    assertEquals(1.0, miIi.get(0, 0), EPSILON);
    assertEquals(2.0, miIi.get(1, 0), EPSILON);
    assertEquals(3.0, miIi.get(2, 0), EPSILON);
    assertEquals(4.0, miIi.get(0, 1), EPSILON);
    assertEquals(5.0, miIi.get(1, 1), EPSILON);
    assertEquals(6.0, miIi.get(2, 1), EPSILON);
  }

  @Test
  public void createRiIiMaybeTransposed() {
    Vector ratings = new SequentialAccessSparseVector(3);
    ratings.setQuick(1, 1.0);
    ratings.setQuick(3, 3.0);
    ratings.setQuick(5, 5.0);

    DoubleMatrix riIiMaybeTransposed = JBlasAlternatingLeastSquaresSolver.createRiIiMaybeTransposed(ratings);
    assertEquals(1, riIiMaybeTransposed.columns, 1);
    assertEquals(3, riIiMaybeTransposed.rows, 3);

    assertEquals(1.0, riIiMaybeTransposed.get(0, 0), EPSILON);
    assertEquals(3.0, riIiMaybeTransposed.get(1, 0), EPSILON);
    assertEquals(5.0, riIiMaybeTransposed.get(2, 0), EPSILON);
  }

  @Test
  public void createRiIiMaybeTransposedExceptionOnNonSequentialVector() {
    Vector ratings = new RandomAccessSparseVector(3);
    ratings.setQuick(1, 1.0);
    ratings.setQuick(3, 3.0);
    ratings.setQuick(5, 5.0);

    try {
      JBlasAlternatingLeastSquaresSolver.createRiIiMaybeTransposed(ratings);
      fail();
    } catch (IllegalArgumentException e) {}
  }

}
