package lab2;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Arrays;
import java.util.Comparator;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

/**
 * 任务4：客户资产分层与资产结构统计。
 *
 * <p>基于任务3的四分位数将客户划分为 低资产 / 中等资产 / 较高资产 / 高资产 四个层级，
 * 分别统计各层级下 7 类细分资产的汇总规模，并计算每类资产在对应层级内的占比。
 * 输出先按等级升序，再按占比由大到小排序。
 *
 * <p>用法：{@code hadoop jar lab2.jar lab2.Task4 <train_input> <task3_output> <output>}
 */
public class Task4 {

    /** 层级名称，与分层索引一一对应。 */
    private static final String[] TIER_NAMES = {"低资产", "中等资产", "较高资产", "高资产"};

    public static class StratifyMapper extends Mapper<LongWritable, Text, IntWritable, Text> {

        private double q25;
        private double q50;
        private double q75;
        private final IntWritable outKey = new IntWritable();
        private final Text outVal = new Text();

        @Override
        protected void setup(Context context) {
            Configuration conf = context.getConfiguration();
            q25 = conf.getDouble("lab2.q25", 0.0);
            q50 = conf.getDouble("lab2.q50", 0.0);
            q75 = conf.getDouble("lab2.q75", 0.0);
        }

        @Override
        protected void map(LongWritable key, Text value, Context context)
                throws IOException, InterruptedException {
            String[] f = value.toString().split(",", -1);
            if (Csv.isHeader(f) || f.length < 13) {
                return;
            }
            double asset = Csv.fieldDouble(f, 10);
            int tier;
            if (asset < q25) {
                tier = 0;
            } else if (asset < q50) {
                tier = 1;
            } else if (asset < q75) {
                tier = 2;
            } else {
                tier = 3;
            }
            outKey.set(tier);
            for (int i = 0; i < Csv.ASSET_TYPE_NAMES.length; i++) {
                double amt = Csv.fieldDouble(f, Csv.TRAIN_ASSET_TYPE_IDX[i]);
                outVal.set(i + "\t" + amt);
                context.write(outKey, outVal);
            }
        }
    }

    public static class StratifyReducer extends Reducer<IntWritable, Text, Text, Text> {

        private final Text outKey = new Text();
        private final Text outVal = new Text();

        @Override
        protected void reduce(IntWritable key, Iterable<Text> values, Context context)
                throws IOException, InterruptedException {
            double[] sum = new double[Csv.ASSET_TYPE_NAMES.length];
            for (Text v : values) {
                String[] p = v.toString().split("\\t", -1);
                int idx = Integer.parseInt(p[0]);
                sum[idx] += Double.parseDouble(p[1]);
            }
            double total = 0.0;
            for (double s : sum) {
                total += s;
            }
            final double totalFinal = total;

            // 按占比由大到小排序（对 7 类资产的索引排序）
            Integer[] order = new Integer[Csv.ASSET_TYPE_NAMES.length];
            for (int i = 0; i < order.length; i++) {
                order[i] = i;
            }
            Arrays.sort(order, Comparator.comparingDouble(
                    (Integer i) -> totalFinal > 0 ? -sum[i] / totalFinal : 0.0));

            String tierName = TIER_NAMES[key.get()];
            outKey.set(tierName);
            for (int idx : order) {
                double proportion = total > 0 ? sum[idx] / total : 0.0;
                outVal.set(Csv.ASSET_TYPE_NAMES[idx] + "\t" + String.format("%.4f", proportion));
                context.write(outKey, outVal);
            }
        }
    }

    public static void main(String[] args) throws Exception {
        if (args.length != 3) {
            System.err.println("Usage: Task4 <train_input> <task3_output> <output>");
            System.exit(2);
        }
        Path trainInput = new Path(args[0]);
        Path task3Output = new Path(args[1]);
        Path output = new Path(args[2]);

        Configuration conf = new Configuration();

        // 从任务3输出读取四分位数
        double[] qs = readQuantiles(conf, task3Output);
        conf.setDouble("lab2.q25", qs[0]);
        conf.setDouble("lab2.q50", qs[1]);
        conf.setDouble("lab2.q75", qs[2]);

        Job job = Job.getInstance(conf, "lab2-task4-asset-stratify");
        job.setJarByClass(Task4.class);
        job.setMapperClass(StratifyMapper.class);
        job.setReducerClass(StratifyReducer.class);
        job.setMapOutputKeyClass(IntWritable.class);
        job.setMapOutputValueClass(Text.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);
        // 单 reducer，保证输出按等级顺序集中在同一文件
        job.setNumReduceTasks(1);
        FileInputFormat.addInputPath(job, trainInput);
        FileOutputFormat.setOutputPath(job, output);
        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }

    /** 读取任务3输出文件（part-r-00000），解析 Q25 / Q50 / Q75。 */
    private static double[] readQuantiles(Configuration conf, Path task3Output) throws IOException {
        FileSystem fs = FileSystem.get(conf);
        Path qFile = new Path(task3Output, "part-r-00000");
        try (BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(qFile)))) {
            String line = br.readLine();
            if (line == null) {
                throw new IOException("任务3输出为空：" + qFile);
            }
            // 行格式：Q25:x<TAB>Q50:y<TAB>Q75:z
            String[] parts = line.split("\\t");
            double q25 = parseQuantile(parts[0]);
            double q50 = parseQuantile(parts[1]);
            double q75 = parseQuantile(parts[2]);
            return new double[]{q25, q50, q75};
        }
    }

    private static double parseQuantile(String field) {
        int colon = field.indexOf(':');
        return Double.parseDouble(field.substring(colon + 1));
    }
}
