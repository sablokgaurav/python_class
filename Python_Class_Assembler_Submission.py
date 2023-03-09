import os
class AssemblySubmission:
    """_summary_
    A template for the submission of
    the jobs to the computing cluster
    You can append your command to the
    written file or else follow on git
    where i am posting a complete
    snakemake file
    """
    def __init__(self, name, queue, threads, core, memory, user, change, mail,filename):
        self.name = name
        self.queue = queue
        self.threads = threads
        self.core = core
        self.memory = memory
        self.dir = os.getcwd()
        self.change = os.chroot(change)
        self.mail = mail
        self.user = user
        self.filename = filename

    def read_file(self):
        self.filecontents = open(self.filename, 'rb')
        for i in self.filecontents.readlines():
            if i.startswith('#SBATCH'):
                raise FileExistsError('Already a configuration file')
            else:
                print(f'the_name_of_the_filename')

    def getCluster(self):
        return print(f'#SBATCH -J self.name
                     \n#SBATCH -p constraint="snb|hsw
                     \n#SBATCH -p self.queue
                     \n#SBATCH -n self.threads
                     \n#SBATCH -c self.core
                     \n#SBATCH --mem=self.memory
                     \n#SBATCH --workdir = self.change
                     \n#SBATCH --mail = self.mail
                    \n#SBATCH --mail-type=END')
    def writeCluster(self):
            self.filecontent = open(self.filename, 'rb')
            self.filecontents.write(self.getCluster())
            self.filecontents.write(self.getcwd())
            self.filecontents.close()
            print(f'the_cluster_file_for_the_configuration_run:{self.filecontents}')

# A draft template for the same using the Java and you can
#put the method of your cluster
public class ClusterScheduleService{
    // variable declaration for the constructor
    public String name;
    public String queue
    public int threads;
    public int core;
    public int memory;
    public String path;
    public String change;
    public String mail;
    public String user;
    public String filename;
}
    // initalize the constructor
    public ClusterScheduleService(){}
    public ClusterScheduleService(String n,
                                  String q,
                                  String t,
                                  String c,
                                  String m,
                                  String d,
                                  String c,
                                  String m,
                                  String u,
                                  string f){
        this.n = name;
        this.q = queue;
        this.t = threads;
        this.c = core;
        this.m = memory;
        this.d = path;
        this.c = change;
        this.m = mail;
        this.u = user;
        this.f = filename;
    }
    // getters if you want to define
    public String getName(){
        return this.n;
    }
    public String getQueue(){
        return this.q;
    }
    public int getThreads(){
        return this.t;
    }
    public int getCore(){
        return this.c;
    }
    public int getMemory(){
        return this.m;
    }
    public String getDirectory(){
        return this.d;
    }
    public String getChange(){
        return this.c;
    }
    public String getMail(){
       return this.m;
   }
    public String getUser(){
       return this.u;
   }
    public String getFilename(){
       return this.f;
   }
    // setters if you want to define
    public void setName(String name){
        this.n = name;
    }
    public void setQueue(String queue){
        this.q = queue;
    }
    public void setThreads(int threads){
        this.t = threads;
    }
    public void setCore(int core){
        this.n = name;
    }
    public void setMemory(int memory){
        this.m = memory;
    }
    public void setChange(String change){
        this.c = change;
    }
    public void setDirectory(String path){
        this.d = path;
    }
    public void setMail(String mail){
        this.m = mail;
    }
    public void setUser(String user){
        this.u = user;
    }
    public void setFilename(String filename){
        this.f = filename;
    }

    public static void main (String[] args) throws Exception {
        ClusterScheduleService b = new ClusterScheduleService();
        System.out.println("")
        // you can print according to your choice
    }