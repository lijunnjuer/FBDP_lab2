package lab2;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

/**
 * 任务3：找出客户总资产的四分位数（Q25 / Q50 / Q75）。
 *
 * <p>利用 MapReduce 框架内置的排序机制（Map → Shuffle(分区、排序) → Reduce）：
 * Map 以资产金额为 key 输出，框架保证进入 Reduce 的数据按 key 升序；
 * 使用单 reducer 得到全局有序的资产序列后，结合总样本条数计算四分位数。
 *
 * <p>用法：{@code hadoop jar lab2.jar lab2.Task3 <train_input> <output>}
 */
public class Task3 {

    public static class AssetMapper extends Mapper<LongWritable, Text, DoubleWritable, NullWritable> {

        private final DoubleWritable outKey = new DoubleWritable();

        @Override
        protected void map(LongWritable key, Text value, Context context)
                throws IOException, InterruptedException {
            String[] f = value.toString().split(",", -1);
            if (Csv.isHeader(f) || f.length < 13) {
                return;
            }
            outKey.set(Csv.fieldDouble(f, 10)); // CUST_ASSET_AMT
            context.write(outKey, NullWritable.get());
        }
    }

    public static class QuantileReducer extends Reducer<DoubleWritable, NullWritable, Text, Text> {

        // 数据按 key 升序进入 reduce，因此这里得到的列表天然有序
        private final List<Double> sortedAssets = new ArrayList<>();

        @Override
        protected void reduce(DoubleWritable key, Iterable<NullWritable> values, Context context)
                throws IOException, InterruptedException {
            for (NullWritable ignored : values) {
                sortedAssets.add(key.get());
            }
        }

        @Override
        protected void cleanup(Context context) throws IOException, InterruptedException {
            double q25 = quantile(sortedAssets, 0.25);
            double q50 = quantile(sortedAssets, 0.50);
            double q75 = quantile(sortedAssets, 0.75);
            // 输出单行：Q25:<值> TAB Q50:<值> TAB Q75:<值>
            context.write(new Text(String.format("Q25:%.2f", q25)),
                    new Text(String.format("Q50:%.2f\tQ75:%.2f", q50, q75)));
        }

        /** 线性插值法计算分位数（等价于 numpy 默认的 linear 插值）。 */
        private static double quantile(List<Double> sorted, double p) {
            int n = sorted.size();
            if (n == 0) {
                return 0.0;
            }
            double pos = p * (n - 1);
            int lo = (int) Math.floor(pos);
            int hi = (int) Math.ceil(pos);
            if (lo == hi) {
                return sorted.get(lo);
            }
            double frac = pos - lo;
            return sorted.get(lo) + frac * (sorted.get(hi) - sorted.get(lo));
        }
    }

    public static void main(String[] args) throws Exception {
        if (args.length != 2) {
            System.err.println("Usage: Task3 <train_input> <output>");
            System.exit(2);
        }
        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "lab2-task3-asset-quantile");
        job.setJarByClass(Task3.class);
        job.setMapperClass(AssetMapper.class);
        job.setReducerClass(QuantileReducer.class);
        job.setMapOutputKeyClass(DoubleWritable.class);
        job.setMapOutputValueClass(NullWritable.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);
        // 单 reducer 以获得全局有序序列
        job.setNumReduceTasks(1);
        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));
        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}
