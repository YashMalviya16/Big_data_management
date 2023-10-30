import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;

public class Silhouette {

    public static final String OutputKM = "input/Task3BYOD_centroids.csv";
    public static final String outputPath = "MapReduceTasks/Output_Silhouette";
    public static class MapperSilhouette extends Mapper<Object, Text, IntWritable, Text> {
        private static int Id = 0;

        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            String[] points = value.toString().split("\\|");
            for (int i = 1; i < points.length; i++){
                context.write(new IntWritable(0), new Text(Id+","+points[i]));
            }

            Id++;
        }
    }

    public static class ReducerSilhouette extends Reducer<IntWritable, Text, Text, Text> {

        public void reduce(IntWritable key, Iterable<Text> values, Context context) throws IOException, InterruptedException {


            int maxId = 0;
            HashMap<Integer, ArrayList<Point>> groups = new HashMap<>();
            for (Text value : values){
                String[] parts = value.toString().split(",");
                if (parts.length != 3){
                    System.out.println("Malformed point\n");
                    System.exit(3);
                }
                int groupID = Integer.parseInt(parts[0]);
                int x = Integer.parseInt(parts[1]);
                int y = Integer.parseInt(parts[2]);

                if (!groups.containsKey(groupID)){
                    groups.put(groupID, new ArrayList<>());
                }

                groups.get(groupID).add(new Point(x, y));

                if (groupID > maxId){
                    maxId = groupID;
                }
            }
            for (Integer groupsKey : groups.keySet()){
                ArrayList<Point> h = groups.get(groupsKey);

                for (Point point : h){
                    ArrayList<Double> groupMeanDistance = new ArrayList<>();

                    for (ArrayList<Point> group : groups.values()){
                        int TotalPoints = 0;
                        double TotalDistance = 0;
                        for (Point otherPoint : group){
                            TotalPoints++;
                            TotalDistance += Math.sqrt(Math.pow(point.x-otherPoint.x, 2) + Math.pow(point.y-otherPoint.y, 2));
                        }
                        groupMeanDistance.add(TotalDistance/TotalPoints);
                    }
                    double minDiffGroupMeanDist = Double.MAX_VALUE;
                    for (Double meanDist : groupMeanDistance){
                        if (!meanDist.equals(groupMeanDistance.get(groupsKey)) && meanDist < minDiffGroupMeanDist){
                            minDiffGroupMeanDist = meanDist;
                        }
                    }

                    point.silhouetteVal = silhouetteValue(groupMeanDistance.get(groupsKey), minDiffGroupMeanDist);
                    context.write(new Text(point.x+","+point.y), new Text(groupsKey+","+point.silhouetteVal));
                }
            }
        }

        private static Double silhouetteValue(Double a, Double b) {
            if (a < b) {
                return 1 - (a/b);
            }
            else if (a > b) {
                return (b/a) - 1;
            }

            return 0.0;
        }
    }

    private static class Point {
        public Integer x, y;
        public Double silhouetteVal;

        public Point(int x, int y){
            this.x = x;
            this.y = y;
            this.silhouetteVal = null;
        }

    }

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "Sillhouette");
        job.setJarByClass(MapperSilhouette.class);
        job.setMapperClass(MapperSilhouette.class);
        job.setReducerClass(ReducerSilhouette.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);
        FileInputFormat.addInputPath(job, new Path(OutputKM));
        FileOutputFormat.setOutputPath(job, new Path(outputPath));
        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}

