package lab2;

import java.io.IOException;

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
 * 任务1：各开户机构客户数量以及资产统计。
 *
 * <p>统计每个开户机构的：客户数量、总资产、总负债、个贷逾期客户数。
 * 缺失值按 0 处理。输出：开户机构编号 &lt;TAB&gt; 客户数量,总资产,总负债,个贷逾期客户数。
 *
 * <p>用法：{@code hadoop jar lab2.jar lab2.Task1 <train_input> <output>}
 */
public class Task1 {

    public static class OrgStatMapper extends Mapper<LongWritable, Text, Text, Text> {

        private final Text outKey = new Text();
        private final Text outVal = new Text();

        @Override
        protected void map(LongWritable key, Text value, Context context)
                throws IOException, InterruptedException {
            String[] f = value.toString().split(",", -1);
            if (Csv.isHeader(f) || f.length < 13) {
                return;
            }
            String org = Csv.fieldStr(f, 1);
            if (org.isEmpty()) {
                return;
            }
            double asset = Csv.fieldDouble(f, 10);
            double debt = Csv.fieldDouble(f, 11);
            long overdue = Csv.fieldLong(f, 2) == 1 ? 1 : 0;

            outKey.set(org);
            outVal.set("1\t" + asset + "\t" + debt + "\t" + overdue);
            context.write(outKey, outVal);
        }
    }

    public static class OrgStatReducer extends Reducer<Text, Text, Text, Text> {

        private final Text outVal = new Text();

        @Override
        protected void reduce(Text key, Iterable<Text> values, Context context)
                throws IOException, InterruptedException {
            long count = 0;
            double sumAsset = 0.0;
            double sumDebt = 0.0;
            long overdue = 0;
            for (Text v : values) {
                String[] p = v.toString().split("\\t", -1);
                count += Long.parseLong(p[0]);
                sumAsset += Double.parseDouble(p[1]);
                sumDebt += Double.parseDouble(p[2]);
                overdue += Long.parseLong(p[3]);
            }
            outVal.set(String.format("%d,%.2f,%.2f,%d", count, sumAsset, sumDebt, overdue));
            context.write(key, outVal);
        }
    }

    public static void main(String[] args) throws Exception {
        if (args.length != 2) {
            System.err.println("Usage: Task1 <train_input> <output>");
            System.exit(2);
        }
        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "lab2-task1-org-stat");
        job.setJarByClass(Task1.class);
        job.setMapperClass(OrgStatMapper.class);
        job.setReducerClass(OrgStatReducer.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);
        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));
        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}
