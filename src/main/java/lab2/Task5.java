package lab2;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URI;
import java.nio.charset.StandardCharsets;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

/**
 * 任务5：基于 MapReduce 实现客户信贷风险预测（高斯朴素贝叶斯）。
 *
 * <p>整体流程：
 * <ol>
 *   <li>训练 Job：扫描 train.csv，统计每个类别（bad_good=0/1）的先验频度，以及
 *       每个类别下每个连续特征的均值与方差，输出紧凑的模型文件；</li>
 *   <li>预测 Job：通过 DistributedCache 把模型文件分发到各 Mapper，在 Map 端对
 *       test.csv 逐客户计算对数后验概率并取最大类别，完成分布式预测。</li>
 * </ol>
 * 使用 10 个连续特征：存款、理财、私人理财、基金、国债、银保通、贵金属、
 * 资产总额、负债总额、贷款账户月余额。缺失值按 0 处理。
 *
 * <p>用法：{@code hadoop jar lab2.jar lab2.Task5 <train_input> <test_input> <model_output> <predict_output>}
 */
public class Task5 {

    private static final int NUM_FEATURES = Csv.TRAIN_FEAT_IDX.length;
    private static final double SMOOTHING = 1e-6;

    /** 训练 Mapper：发射类别先验频度与特征统计量。 */
    public static class TrainMapper extends Mapper<LongWritable, Text, Text, Text> {

        private final Text outKey = new Text();
        private final Text outVal = new Text();

        @Override
        protected void map(LongWritable key, Text value, Context context)
                throws IOException, InterruptedException {
            String[] f = value.toString().split(",", -1);
            if (Csv.isHeader(f) || f.length < 13) {
                return;
            }
            int label = (int) Csv.fieldLong(f, 2); // bad_good

            // 类别先验：key = C<TAB>label
            outKey.set("C\t" + label);
            outVal.set("1");
            context.write(outKey, outVal);

            // 特征统计：key = F<TAB>label<TAB>featureIdx，value = 1<TAB>x<TAB>x^2
            for (int j = 0; j < NUM_FEATURES; j++) {
                double x = Csv.fieldDouble(f, Csv.TRAIN_FEAT_IDX[j]);
                outKey.set("F\t" + label + "\t" + j);
                outVal.set("1\t" + x + "\t" + (x * x));
                context.write(outKey, outVal);
            }
        }
    }

    /** 训练 Reducer：汇总频度与统计量，输出模型文件。 */
    public static class TrainReducer extends Reducer<Text, Text, Text, Text> {

        private final Text outVal = new Text();

        @Override
        protected void reduce(Text key, Iterable<Text> values, Context context)
                throws IOException, InterruptedException {
            String keyStr = key.toString();
            if (keyStr.startsWith("C\t")) {
                long cnt = 0;
                for (Text v : values) {
                    cnt += Long.parseLong(v.toString());
                }
                outVal.set(String.valueOf(cnt));
                context.write(key, outVal);
            } else {
                long cnt = 0;
                double sum = 0.0;
                double sumSq = 0.0;
                for (Text v : values) {
                    String[] p = v.toString().split("\\t", -1);
                    cnt += Long.parseLong(p[0]);
                    sum += Double.parseDouble(p[1]);
                    sumSq += Double.parseDouble(p[2]);
                }
                double mean = sum / cnt;
                double var = sumSq / cnt - mean * mean;
                if (var < 0) {
                    var = 0.0;
                }
                outVal.set(cnt + "\t" + mean + "\t" + var);
                context.write(key, outVal);
            }
        }
    }

    /** 预测 Mapper：加载模型，对测试样本做分布式预测。 */
    public static class PredictMapper extends Mapper<LongWritable, Text, Text, Text> {

        private final double[] prior = new double[2];
        private final double[][] mean = new double[2][NUM_FEATURES];
        private final double[][] variance = new double[2][NUM_FEATURES];
        private final Text outKey = new Text();
        private final Text outVal = new Text();

        @Override
        protected void setup(Context context) throws IOException {
            URI[] cacheFiles = context.getCacheFiles();
            if (cacheFiles == null) {
                return;
            }
            double[] classCount = new double[2];
            double total = 0.0;
            for (URI uri : cacheFiles) {
                String name = new Path(uri.getPath()).getName();
                try (BufferedReader br = new BufferedReader(
                        new InputStreamReader(new FileInputStream(name), StandardCharsets.UTF_8))) {
                    String line;
                    while ((line = br.readLine()) != null) {
                        String[] p = line.split("\\t", -1);
                        if (p.length < 3) {
                            continue;
                        }
                        if ("C".equals(p[0])) {
                            int label = Integer.parseInt(p[1]);
                            classCount[label] = Double.parseDouble(p[2]);
                            total += classCount[label];
                        } else if ("F".equals(p[0])) {
                            int label = Integer.parseInt(p[1]);
                            int j = Integer.parseInt(p[2]);
                            mean[label][j] = Double.parseDouble(p[4]);
                            variance[label][j] = Double.parseDouble(p[5]);
                        }
                    }
                }
            }
            for (int label = 0; label < 2; label++) {
                prior[label] = total > 0 ? classCount[label] / total : 0.0;
            }
            // 方差正则化：防止某类中恒定的特征（方差≈0）在预测时被过度加权。
            // 用该特征的全局方差（组内+组间）的 1% 作为加性平滑项。
            for (int j = 0; j < NUM_FEATURES; j++) {
                double sum0 = classCount[0] * mean[0][j];
                double sum1 = classCount[1] * mean[1][j];
                double sumSq0 = classCount[0] * (variance[0][j] + mean[0][j] * mean[0][j]);
                double sumSq1 = classCount[1] * (variance[1][j] + mean[1][j] * mean[1][j]);
                double meanTotal = total > 0 ? (sum0 + sum1) / total : 0.0;
                double meanSqTotal = total > 0 ? (sumSq0 + sumSq1) / total : 0.0;
                double totalVar = meanSqTotal - meanTotal * meanTotal;
                if (totalVar < 0) {
                    totalVar = 0.0;
                }
                variance[0][j] += 0.01 * totalVar;
                variance[1][j] += 0.01 * totalVar;
            }
        }

        @Override
        protected void map(LongWritable key, Text value, Context context)
                throws IOException, InterruptedException {
            String[] f = value.toString().split(",", -1);
            if (Csv.isHeader(f) || f.length < 14) {
                return;
            }
            String custId = Csv.fieldStr(f, 0);
            double[] x = new double[NUM_FEATURES];
            for (int j = 0; j < NUM_FEATURES; j++) {
                x[j] = Csv.fieldDouble(f, Csv.TEST_FEAT_IDX[j]);
            }

            int predict = predict(x);
            outKey.set(custId);
            outVal.set(String.valueOf(predict));
            context.write(outKey, outVal);
        }

        private int predict(double[] x) {
            double bestScore = Double.NEGATIVE_INFINITY;
            int bestLabel = 0;
            for (int label = 0; label < 2; label++) {
                double score = Math.log(prior[label] + SMOOTHING);
                for (int j = 0; j < NUM_FEATURES; j++) {
                    score += logGaussian(x[j], mean[label][j], variance[label][j]);
                }
                if (score > bestScore) {
                    bestScore = score;
                    bestLabel = label;
                }
            }
            return bestLabel;
        }

        private static double logGaussian(double x, double mean, double var) {
            double v = var + SMOOTHING;
            return -0.5 * Math.log(2 * Math.PI * v) - (x - mean) * (x - mean) / (2.0 * v);
        }
    }

    public static void main(String[] args) throws Exception {
        if (args.length != 4) {
            System.err.println("Usage: Task5 <train_input> <test_input> <model_output> <predict_output>");
            System.exit(2);
        }
        Path trainInput = new Path(args[0]);
        Path testInput = new Path(args[1]);
        Path modelOutput = new Path(args[2]);
        Path predictOutput = new Path(args[3]);

        Configuration conf = new Configuration();

        // ---- 训练 Job ----
        Job trainJob = Job.getInstance(conf, "lab2-task5-train");
        trainJob.setJarByClass(Task5.class);
        trainJob.setMapperClass(TrainMapper.class);
        trainJob.setReducerClass(TrainReducer.class);
        trainJob.setOutputKeyClass(Text.class);
        trainJob.setOutputValueClass(Text.class);
        trainJob.setNumReduceTasks(1); // 模型很小，单 reducer 输出单个文件
        FileInputFormat.addInputPath(trainJob, trainInput);
        FileOutputFormat.setOutputPath(trainJob, modelOutput);
        if (!trainJob.waitForCompletion(true)) {
            System.exit(1);
        }

        // ---- 预测 Job ----
        Job predictJob = Job.getInstance(conf, "lab2-task5-predict");
        predictJob.setJarByClass(Task5.class);
        predictJob.addCacheFile(new Path(modelOutput, "part-r-00000").toUri());
        predictJob.setMapperClass(PredictMapper.class);
        predictJob.setNumReduceTasks(0); // 预测在 Map 端完成
        predictJob.setOutputKeyClass(Text.class);
        predictJob.setOutputValueClass(Text.class);
        FileInputFormat.addInputPath(predictJob, testInput);
        FileOutputFormat.setOutputPath(predictJob, predictOutput);
        System.exit(predictJob.waitForCompletion(true) ? 0 : 1);
    }
}
