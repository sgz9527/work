

public class IoFile {
    public static void main(String[] args) {
    IoFile ios=new IoFile();
    ios.test1();
    }
    public void readFile(){
       // String realPath=this.getClass().getClassLoader().getResource("/").getPath();
        //System.out.println("path:"+realPath);
    }
    public void test1(){
        for(int i=0;i<9;i++)
        {
            int num=(int)(Math.random()*9);
            System.out.println("num:"+num);
        }
    }
}
