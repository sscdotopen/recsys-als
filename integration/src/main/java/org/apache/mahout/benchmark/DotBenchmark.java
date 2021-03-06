package org.apache.mahout.benchmark;

import static org.apache.mahout.benchmark.VectorBenchmarks.DENSE_FN_RAND;
import static org.apache.mahout.benchmark.VectorBenchmarks.DENSE_FN_SEQ;
import static org.apache.mahout.benchmark.VectorBenchmarks.DENSE_VECTOR;
import static org.apache.mahout.benchmark.VectorBenchmarks.RAND_FN_DENSE;
import static org.apache.mahout.benchmark.VectorBenchmarks.RAND_FN_SEQ;
import static org.apache.mahout.benchmark.VectorBenchmarks.RAND_SPARSE_VECTOR;
import static org.apache.mahout.benchmark.VectorBenchmarks.SEQ_FN_DENSE;
import static org.apache.mahout.benchmark.VectorBenchmarks.SEQ_FN_RAND;
import static org.apache.mahout.benchmark.VectorBenchmarks.SEQ_SPARSE_VECTOR;

import org.apache.mahout.benchmark.BenchmarkRunner.BenchmarkFn;
import org.apache.mahout.benchmark.BenchmarkRunner.BenchmarkFnD;

public class DotBenchmark {
  private static final String DOT_PRODUCT = "DotProduct";
  private static final String NORM1 = "Norm1";
  private static final String LOG_NORMALIZE = "LogNormalize";
  private final VectorBenchmarks mark;

  public DotBenchmark(VectorBenchmarks mark) {
    this.mark = mark;
  }

  public void benchmark() {
    benchmarkDot();
    benchmarkNorm1();
    benchmarkLogNormalize();
  }

  private void benchmarkLogNormalize() {
    mark.printStats(mark.getRunner().benchmark(new BenchmarkFn() {
      @Override
      public Boolean apply(Integer i) {
        return depends(mark.vectors[0][mark.vIndex(i)].logNormalize());
      }
    }), LOG_NORMALIZE, DENSE_VECTOR);

    mark.printStats(mark.getRunner().benchmark(new BenchmarkFn() {
      @Override
      public Boolean apply(Integer i) {
        return depends(mark.vectors[1][mark.vIndex(i)].logNormalize());
      }
    }), LOG_NORMALIZE, RAND_SPARSE_VECTOR);

    mark.printStats(mark.getRunner().benchmark(new BenchmarkFn() {
      @Override
      public Boolean apply(Integer i) {
        return depends(mark.vectors[2][mark.vIndex(i)].logNormalize());
      }
    }), LOG_NORMALIZE, SEQ_SPARSE_VECTOR);
  }

  private void benchmarkNorm1() {
    mark.printStats(mark.getRunner().benchmarkD(new BenchmarkFnD() {
      @Override
      public Double apply(Integer i) {
        return mark.vectors[0][mark.vIndex(i)].norm(1);
      }
    }), NORM1, DENSE_VECTOR);

    mark.printStats(mark.getRunner().benchmarkD(new BenchmarkFnD() {
      @Override
      public Double apply(Integer i) {
        return mark.vectors[1][mark.vIndex(i)].norm(1);
      }
    }), NORM1, RAND_SPARSE_VECTOR);

    mark.printStats(mark.getRunner().benchmarkD(new BenchmarkFnD() {
      @Override
      public Double apply(Integer i) {
        return mark.vectors[2][mark.vIndex(i)].norm(1);
      }
    }), NORM1, SEQ_SPARSE_VECTOR);
  }

  private void benchmarkDot() {
    mark.printStats(mark.getRunner().benchmarkD(new BenchmarkFnD() {
      @Override
      public Double apply(Integer i) {
        return mark.vectors[0][mark.vIndex(i)].dot(mark.vectors[0][mark.vIndex(randIndex())]);
      }
    }), DOT_PRODUCT, DENSE_VECTOR);

    mark.printStats(mark.getRunner().benchmarkD(new BenchmarkFnD() {
      @Override
      public Double apply(Integer i) {
        return mark.vectors[1][mark.vIndex(i)].dot(mark.vectors[1][mark.vIndex(randIndex())]);
      }
    }), DOT_PRODUCT, RAND_SPARSE_VECTOR);

    mark.printStats(mark.getRunner().benchmarkD(new BenchmarkFnD() {
      @Override
      public Double apply(Integer i) {
        return mark.vectors[2][mark.vIndex(i)].dot(mark.vectors[2][mark.vIndex(randIndex())]);
      }
    }), DOT_PRODUCT, SEQ_SPARSE_VECTOR);

    mark.printStats(mark.getRunner().benchmarkD(new BenchmarkFnD() {
      @Override
      public Double apply(Integer i) {
        return mark.vectors[0][mark.vIndex(i)].dot(mark.vectors[1][mark.vIndex(randIndex())]);
      }
    }), DOT_PRODUCT, DENSE_FN_RAND);

    mark.printStats(mark.getRunner().benchmarkD(new BenchmarkFnD() {
      @Override
      public Double apply(Integer i) {
        return mark.vectors[0][mark.vIndex(i)].dot(mark.vectors[2][mark.vIndex(randIndex())]);
      }
    }), DOT_PRODUCT, DENSE_FN_SEQ);

    mark.printStats(mark.getRunner().benchmarkD(new BenchmarkFnD() {
      @Override
      public Double apply(Integer i) {
        return mark.vectors[1][mark.vIndex(i)].dot(mark.vectors[0][mark.vIndex(randIndex())]);
      }
    }), DOT_PRODUCT, RAND_FN_DENSE);

    mark.printStats(mark.getRunner().benchmarkD(new BenchmarkFnD() {
      @Override
      public Double apply(Integer i) {
        return mark.vectors[1][mark.vIndex(i)].dot(mark.vectors[2][mark.vIndex(randIndex())]);
      }
    }), DOT_PRODUCT, RAND_FN_SEQ);

    mark.printStats(mark.getRunner().benchmarkD(new BenchmarkFnD() {
      @Override
      public Double apply(Integer i) {
        return mark.vectors[2][mark.vIndex(i)].dot(mark.vectors[0][mark.vIndex(randIndex())]);
      }
    }), DOT_PRODUCT, SEQ_FN_DENSE);

    mark.printStats(mark.getRunner().benchmarkD(new BenchmarkFnD() {
      @Override
      public Double apply(Integer i) {
        return mark.vectors[2][mark.vIndex(i)].dot(mark.vectors[1][mark.vIndex(randIndex())]);
      }
    }), DOT_PRODUCT, SEQ_FN_RAND);
  }
}
