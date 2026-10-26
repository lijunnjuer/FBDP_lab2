package lab2;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.Map;

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
 * 任务2：平均值分析。
 *
 * <p>基于任务1的结果，统计每个机构中客户资产高于 / 低于该机构客户资产平均值的客户数量。
 * 等于平均值的归入“高于”部分。
 *
 * <p>实现为一个两阶段流水线：
 * <ol>
 *   <li>Job A（map-only）：读取任务1输出，计算每个机构的平均资产，写入中间目录；</li>
 *   <li>Job B：通过 DistributedCache 分发平均值表，遍历 train.csv 逐客户比较，统计高于/低于数量。</li>
 * </ol>
 *
 * <p>用法：{@code hadoop jar lab2.jar lab2.Task2 <task1_output> <train_input> <output>}
 */
public class Task2 {

    /** Job A：由任务1输出计算机构平均资产。 */
    public static class AvgMapper extends Mapper<LongWritable, Text, Text, Text> {

        private final Text outKey = new Text();
        private final Text outVal = new Text();

        @Override
        protected void map(LongWritable key, Text value, Context context)
                throws IOException, InterruptedException {
            // 任务1输出：机构编号 <TAB> 客户数量,总资产,总负债,逾期客户数
            String[] parts = value.toString().split("\\t", -1);
            if (parts.length < 2) {
                return;
            }
            String[] stats = parts[1].split(",", -1);
            if (stats.length < 2) {
                return;
            }
            long count = Long.parseLong(stats[0]);
            if (count == 0) {
                return;
            }
            double avg = Double.parseDouble(stats[1]) / count;
            outKey.set(parts[0]);
            outVal.set(String.valueOf(avg));
            context.write(outKey, outVal);
        }
    }

    /** Job B：比较每个客户的资产与其机构平均值，统计高于/低于数量。 */
    public static class CountMapper extends Mapper<LongWritable, Text, Text, Text> {

        private final Map<String, Double> avgMap = new HashMap<>();
        private final Text outKey = new Text();
        private final Text outVal = new Text();

        @Override
        protected void setup(Context context) throws IOException {
            // 读取 DistributedCache 中的平均值表：机构编号 <TAB> 平均资产
            URI[] cacheFiles = context.getCacheFiles();
            if (cacheFiles == null) {
                return;
            }
            for (URI uri : cacheFiles) {
                String name = new Path(uri.getPath()).getName();
                try (BufferedReader br = new BufferedReader(
                        new InputStreamReader(new FileInputStream(name), StandardCharsets.UTF_8))) {
                    String line;
                    while ((line = br.readLine()) != null) {
                        String[] kv = line.split("\\t", -1);
                        if (kv.length >= 2) {
                            avgMap.put(kv[0], Double.parseDouble(kv[1]));
                        }
                    }
                }
            }
        }

        @Override
        protected void map(LongWritable key, Text value, Context context)
                throws IOException, InterruptedException {
            String[] f = value.toString().split(",", -1);
            if (Csv.isHeader(f) || f.length < 13) {
                return;
            }
            String org = Csv.fieldStr(f, 1);
            Double avg = avgMap.get(org);
            if (avg == null) {
                return;
            }
            double asset = Csv.fieldDouble(f, 10);
            // 资产 >= 平均值 → 高于（等于归入大于部分）
            String above = asset >= avg ? "1\t0" : "0\t1";
            outKey.set(org);
            outVal.set(above);
            context.write(outKey, outVal);
        }
    }

    public static class CountReducer extends Reducer<Text, Text, Text, Text> {

        private final Text outVal = new Text();

        @Override
        protected void reduce(Text key, Iterable<Text> values, Context context)
                throws IOException, InterruptedException {
            long above = 0;
            long below = 0;
            for (Text v : values) {
                String[] p = v.toString().split("\\t", -1);
                above += Long.parseLong(p[0]);
                below += Long.parseLong(p[1]);
            }
            outVal.set(above + "," + below);
            context.write(key, outVal);
        }
    }

    public static void main(String[] args) throws Exception {
        if (args.length != 3) {
            System.err.println("Usage: Task2 <task1_output> <train_input> <output>");
            System.exit(2);
        }
        Path task1Output = new Path(args[0]);
        Path trainInput = new Path(args[1]);
        Path output = new Path(args[2]);

        Configuration conf = new Configuration();

        // ---- Job A：计算机构平均资产（单 reducer，保证输出为单个 part-r-00000 文件） ----
        Path avgDir = new Path(output.getParent(), output.getName() + "_avg");
        Job avgJob = Job.getInstance(conf, "lab2-task2-avg");
        avgJob.setJarByClass(Task2.class);
        avgJob.setMapperClass(AvgMapper.class);
        avgJob.setNumReduceTasks(1);
        avgJob.setOutputKeyClass(Text.class);
        avgJob.setOutputValueClass(Text.class);
        FileInputFormat.addInputPath(avgJob, task1Output);
        FileOutputFormat.setOutputPath(avgJob, avgDir);
        if (!avgJob.waitForCompletion(true)) {
            System.exit(1);
        }

        // ---- Job B：将平均值表加入 DistributedCache，统计高于/低于数量 ----
        Job countJob = Job.getInstance(conf, "lab2-task2-count");
        countJob.setJarByClass(Task2.class);
        countJob.addCacheFile(new Path(avgDir, "part-r-00000").toUri());
        countJob.setMapperClass(CountMapper.class);
        countJob.setReducerClass(CountReducer.class);
        countJob.setOutputKeyClass(Text.class);
        countJob.setOutputValueClass(Text.class);
        FileInputFormat.addInputPath(countJob, trainInput);
        FileOutputFormat.setOutputPath(countJob, output);
        System.exit(countJob.waitForCompletion(true) ? 0 : 1);
    }
}
