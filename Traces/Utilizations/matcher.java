import java.util.*;
import java.io.*;
import java.lang.Math;

public class matcher {
    private static void print_util(List<vssd> vssds, String filename){
        HashMap<String, Double> util_per_vm = new HashMap<>();
        HashMap<String, Double> vssd_per_vm = new HashMap<>();
        HashMap<String, Double> max_util_vm = new HashMap<>();
        for(vssd v : vssds){
            String id = v.id;
            if(!util_per_vm.containsKey(id)){
                util_per_vm.put(id, v.total_util);
                vssd_per_vm.put(id, v.length);
                max_util_vm.put(id, v.max_util);
            }
            else{
                util_per_vm.put(id, util_per_vm.get(id)+v.total_util);
                vssd_per_vm.put(id, vssd_per_vm.get(id)+v.length);
                max_util_vm.put(id, Math.min(max_util_vm.get(id), v.max_util));
            }
        }

        // for(String k : util_per_vm.keySet()){
        //     System.out.println(k+" "+(1-util_per_vm.get(k)/vssd_per_vm.get(k)/allocs.get(k)));
        // }

        try (BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(new File(filename))))) {
            for(String k : util_per_vm.keySet()){
                writer.write(k+" "+(1-util_per_vm.get(k)/vssd_per_vm.get(k))+" "+(1-max_util_vm.get(k)));
                writer.newLine();  // method provided by BufferedWriter
            }
        } catch (IOException e) {}

        // try (BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(new File(filename+".alloc"))))) {
            
        // } catch (IOException e) {}
    }
    public static void main(String[] args) throws FileNotFoundException ,IOException{
        if (args.length != 2) {
            System.out.println("Add the files");
            System.exit(0);
        }
        PriorityQueue<Event> q = new PriorityQueue<>();
        TreeMap<vssd, Integer> set = new TreeMap<>();
        List<vssd> vssds = new ArrayList<>();
        HashMap<String, Double> allocs = new HashMap<>();
        double bufup = 1.05;
        double bufdown = 1/1.05;
        BufferedReader br1 = new BufferedReader(new FileReader(args[0]));
        String line = br1.readLine();
        while(line!=null) {
            String[] ss = line.split(",");
            double amt = Double.parseDouble(ss[3]) * bufup;
            double start = Double.parseDouble(ss[1]);
            double end = Double.parseDouble(ss[2]);
            double len = end - start;
            end = start + (bufup * len);
            vssd vtemp = new vssd(amt, start, end);
            Event ev = new Event(vtemp, 2); 
            q.offer(ev);
            line = br1.readLine();
        }
        System.out.println("Done parse file 1");
        BufferedReader br2 = new BufferedReader(new FileReader(args[1]));
        line = br2.readLine();
        while(line!=null) {
            String[] ss = line.split(",");
            double amt = Double.parseDouble(ss[3]);
            double start = Double.parseDouble(ss[1]);
            double end = Double.parseDouble(ss[2]);
            String id = ss[0];
            double len = end - start;
            end = start + (bufdown * len);

            vssd vtemp = new vssd(amt, start, end, id);
            vssd vtemp2 = new vssd(amt, end, end);
            vssds.add(vtemp);

            Event ev = new Event(vtemp, 0); 
            Event ev2 = new Event(vtemp2, 1);
            q.offer(ev);
            q.offer(ev2);
            line = br2.readLine();
        }
        System.out.println("Done parse file 2");
        print_util(vssds, "original_util.txt");
        int tot_reqs = 0;
        int suc = 0;
        while(!q.isEmpty()) {
            if(tot_reqs % 100000 == 0 && tot_reqs != 0)
                System.out.println("Total harvest requests: "+ tot_reqs + " Succ Rate: " + ((double)suc/tot_reqs));
            // if(tot_reqs == 20000) break;
            Event cur_ev = q.poll();
            if (cur_ev.type == 0) {
                if(!set.containsKey(cur_ev.v))
                    set.put(cur_ev.v, 1);
                else
                    set.replace(cur_ev.v, set.get(cur_ev.v)+1);
                // System.out.println("Adding: " + cur_ev.v.toString());
            } else if (cur_ev.type == 1) {
                if(set.containsKey(cur_ev.v)){
                    if(set.get(cur_ev.v) == 1) set.remove(cur_ev.v);
                    else set.replace(cur_ev.v, set.get(cur_ev.v)-1);
                }
                // System.out.println("Removing: " + cur_ev.v.toString());
            } else {
                tot_reqs++;
                // System.out.println("Querying: " + cur_ev.v.toString());
                vssd vv = set.higherKey(cur_ev.v);
                int reason =0;
                while(vv!=null && cur_ev.v.rss > vv.rss * bufdown) {
                    vv = set.higherKey(vv);
                    reason = 1;
                }
                if (vv != null) {
                    // if(cur_ev.v.rss > 0.002 || vv.rss > 0.002)
                    // System.out.println(set.toString());
                    vv.allocate(cur_ev.v.start, cur_ev.v.end, cur_ev.v.rss, cur_ev.v);
                    suc++;
                    // System.out.println(set.toString()+cur_ev.v.toString());
                    if(set.get(vv) == 1) set.remove(vv);
                    else set.replace(vv, set.get(vv)-1);
                    vv.start = cur_ev.v.end;
                    q.offer(new Event(vv, 0));
                }
                // else if (cur_ev.v.rss > 0.002){
                //     if(reason == 0) System.out.println("Unmatch reason: no vssds in this period");
                //     if(reason == 1){
                //         System.out.println("Unmatch reason: vssds has lower harvestable resource for " + cur_ev.v.toString());
                //         for(vssd v : set){
                //             if (cur_ev.v.rss <= v.rss)
                //                 System.out.println(v.toString());
                //         }
                //     }
                // }
            }
        }

        print_util(vssds, "improved_util.txt");
    }
    static class Event implements Comparable<Event> {
        int type;
        vssd v;
        public Event(vssd vv, int tt) {
            v = vv; type = tt;
        }
        public int compareTo(Event o) {
            if (Double.compare(v.start, o.v.start) == 0) {
                return type - o.type;
            }
            return Double.compare(v.start, o.v.start);
        }
    }
    static class vssd implements Comparable<vssd>{
        double rss;
        double start;
        int channels;
        double end;
        String id;
        double total_util;
        double length;
        double max_util;
        double harvest_length;
        public vssd(double rr, double ss, double ee, String uid) {
            rss = rr; start = ss; end = ee; id = uid; 
            length = end - start; total_util = length * rss;
            max_util = rss;
            harvest_length = 0;
        }
        public vssd(double rr, double ss, double ee) {
            rss = rr; start = ss; end = ee; id = ""; 
            length = end - start; total_util = length * rss;
            max_util = rss;
            harvest_length = 0;
        }
        public void allocate(double ev_start, double ev_end, double ev_rss, vssd env_v){
            total_util -= (ev_end - ev_start) * ev_rss;
            max_util = Math.min(rss, rss - ev_rss);
            harvest_length += ev_end - ev_start;
        }
        public int compareTo(vssd o) {
            if (Double.compare(end, o.end) == 0) {
                if (Double.compare(rss, o.rss) == 0) {
                    return id.compareTo(o.id);
                }
                return Double.compare(rss, o.rss);
            }
            return Double.compare(end, o.end);
        }
        public String toString() {
            return "(" + id + "," + rss + "," + start + "," + end + ")";
        }
    }
}

