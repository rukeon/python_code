import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;

import java.io.IOException;
import java.util.StringTokenizer;

/**
 * Created by Ryu Ki Hyun
 */
public class OutdegreeCounter extends Configured implements Tool {


    public static class OutdegreeMapper extends Mapper<LongWritable, Text, IntWritable, IntWritable>{
        private static final IntWritable ONE = new IntWritable(1);
        private final transient IntWritable word = new IntWritable();

        @Override
        public void map(final LongWritable key, final Text value, final Context context) throws IOException, InterruptedException{
            final String line = value.toString();
            final String splitData[] = line.split("\t");
            word.set(Integer.parseInt(splitData[0]));
            context.write(word, ONE);
        }
    }

    public static class OutdegreeReducer extends Reducer<IntWritable, IntWritable, IntWritable, IntWritable> {

        @Override
        public void reduce(final IntWritable key, final Iterable<IntWritable> values, final Context context) throws IOException, InterruptedException{
            int sum = 0;
            for (final IntWritable val : values) {
                sum += val.get();
            }
            context.write(key, new IntWritable(sum));
        }
    }

    public int run(final String[] args) throws Exception {
        final Configuration conf = this.getConf();
        final Job job = Job.getInstance(conf, "Outdegree Count");
        job.setJarByClass(OutdegreeCounter.class);

        job.setMapperClass(OutdegreeMapper.class);
        job.setReducerClass(OutdegreeReducer.class);

        job.setOutputKeyClass(IntWritable.class);
        job.setOutputValueClass(IntWritable.class);

        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        return job.waitForCompletion(true)? 0 : 1;
    }

    public static void main(final String[] args) throws Exception {
        final int returnCode = ToolRunner.run(new Configuration(), new OutdegreeCounter(), args);
        System.exit(returnCode);
    }
}
