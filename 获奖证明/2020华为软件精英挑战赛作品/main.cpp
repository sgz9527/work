/*
边节点里面的 adjvex为顶点数组中的下标
不存在重复转账记录，即重边 
*/
#include<stdio.h>
#include<unordered_map>
#include<iostream>
#include<vector>
#include<string.h>
#include<algorithm>
#include<pthread.h>
using namespace std;
typedef unsigned int elemtype;
#define NANS -1
#define LEN 7
#define maxsize 230000
#define CN 5
#define THREAD_NUM 32
#define MAX_ID 300000
#define BITSIZE 37501
/*enum sorts{
	before=0,after=1
};
*/
/*
	54,3738,38252,58284,77409,1004812,3512444,2896262
./testData/HW2020-testdata/test/test_data.txt
./testData/HW2020-testdata/test/out.txt
./test_data.txt
./out.txt
*/
typedef struct{//栈
	int arr[LEN];
	int length; 
}stack;
struct ARG_THREAD{
    int left;
    int right;
    int num;
}arg_threads[THREAD_NUM];
int adjvex[maxsize][30];
int adjIn[maxsize][55];
int outdegree[maxsize]={0};
int indegree[maxsize]={0};
char data[maxsize][20];
int dataLen[maxsize];
vector<stack > path_map[5<<CN];//第k长度第N份的路径 对应下标为k*size+n,eg:n=7,k=4,res=4*8+7=39 
char buf[300000000];
unsigned char bitmap[BITSIZE];//位图 
unordered_map<elemtype,int> ma;//ID-->index(newID) ,这里将对应的索引号作为新的ID存入顶点的data中，在存入文件时候再映射回旧ID 
elemtype inputArr[560000];
elemtype extra[560000];//存放超出阈值的ID值 
 string submitInputFile= "/data/test_data.txt";
 string submitOutputFile= "/projects/student/result.txt";
 string inputfile= "./testData/hwdata/1004812/test_data.txt";
 string outputfile="./testData/hwdata/1004812/out1.txt";
int ans[THREAD_NUM]={0};
int cntvex=0;//统计ID个数 
int cntedge=0;//边的个数 
int thread_n=THREAD_NUM;//线程个数 
int ans_sum = 0;
inline void init_stack(stack *s){
	s->length=-1;	
}
bool isEmpty(stack *s){
	if(s->length==-1){
		return true;
	}
	return false;
}
inline void push_stack(stack *s,int x){
	s->arr[++s->length]=x;
}
inline void createBitmap( int arraySize,int &len){//extra是ID值超过表示范围的数 
        int i;
	//将array中的每个数所对应的bit下标设置为1
        for(i=0; i<arraySize; ++i){
        	if(inputArr[i]>MAX_ID){//如果超出这个最大值，则位图放不下，将其添加到extra数组，不作处理 
        	//	cout<<len<<endl;
        		//cout<<">max:"<<inputArr[i]<<endl;
        		extra[len++]=inputArr[i];
			}else{
				bitmap[inputArr[i]/8] = bitmap[inputArr[i]/8]|(0x1<<(7-inputArr[i]%8));
			}
        }
}
inline void creategraph(){
		for( int i=0,index_from,index_to,count=cntedge<<1;i<count;i+=2)
		{
			 
			 index_from = ma[inputArr[i]];
			 index_to = ma[inputArr[i+1]];
			adjvex[index_from][outdegree[index_from]++]=index_to;
			adjIn[index_to][indegree[index_to]++]=index_from;
		}
}

inline void inputFile(string &fileName){
	cout<<"inputfile"<<endl;
	elemtype from,to,w;
	FILE* fp=fopen(fileName.c_str(),"r");
	if(fp==NULL){
		cout<<"open inputfile error"<<endl;
		exit(0);
	}
	int i=0;
	while(fscanf(fp,"%u,%u,%u",&from,&to,&w)!=EOF){
			inputArr[i++]=from;
			inputArr[i++]=to;
	}
	cout<<"inputfile"<<endl;
	fclose(fp);
	//将ID按一行一行 
	cntedge = i>>1;//边的个数，也即inputArr的长度的一半 
	int extraLen=0;//记录extra长度 
	createBitmap(i,extraLen);
	//printf("inputArr size is:%d",inputArr.size());
	unsigned char mask = 0x1;
    int j;
    int x;
	//遍历每个unsigned char
	if(extraLen<maxsize){
		for(i=0; i<BITSIZE; ++i){
		//遍历每个unsigned char中的bit
        for(j=7; j>=0;j--){
			//如果指定的unsigned char bitmap[i]的第j位（从低位往高位数）不为0，输出该bit的下标
            if((bitmap[i] & (0x1<<j)) != 0){
                x = (unsigned int)(i*8+(7-j));//找到一个新的ID 
                ma[x]=cntvex;
				dataLen[cntvex]=sprintf(data[cntvex],"%u,",x);
				++cntvex;
            	}
        	}
    	}
	}
    if(extraLen){
    	//extraLen不为0说明还有大于MAX_ID的ID，对其进行排序去重
    //	cout<<extraLen<<endl;
		sort(extra,extra+extraLen);
		int len = unique(extra,extra+extraLen)-extra;
		for(int i=0;i<len;++i){
			x=extra[i];
			ma[x]=cntvex;
			dataLen[cntvex]=sprintf(data[cntvex],"%u,",x);
			++cntvex;
		} 
	}
}
//打印图信息
void print()
{
    printf("graph struct：\n");
    printf("already exist edge num is:%d\n",cntedge);
    printf("already exist point num is:%d\n",cntvex);
    //for(int i=0;i<cntvex;++i){
    //	for(int j=0;j<outdegree[i];++j){
   // 		cout<<"-->"<<adjvex[i][j];
	//	}
	//	cout<<endl;
//	}
}

void findNearIn(int vex_index,int begin,int *isNear,bool *isVisit,int depth){
	/*
	   本函数找到以begin为起点的邻域小于等于3的入边点,结果保存在isNear中
	   当后续进行深度搜索找以begin为起点的环时，可根据isNear是否等于begin判断遍历到的点是否在该邻域中，如果不在，
	   则通过该点不可能找到以begin为起点的长度<=7的环,可以直接跳过，以此达到剪枝的目的 
	*/
	*(isVisit+vex_index) = true;
	register auto tmp = adjIn[vex_index];
	int indeg = indegree[vex_index];
	for(int i=0;i<indeg;++i){
			int adj = tmp[i];
			if(!*(isVisit+adj)&&begin<adj){
				if(depth<=3){//当到达第三层时
					*(isNear+adj)=begin;
				}
				if(depth<3){
					findNearIn(adj,begin,isNear,isVisit,depth+1);
				}
			}					
	}
	*(isVisit+vex_index)= false;
}

void findCircle(int vex_index,int begin,int *isNear,bool *isVisit,stack *s,int depth,int n){
	push_stack(s,vex_index);
	*(isVisit+vex_index)=true;
	register auto tmp = adjvex[vex_index];
	int outgre = outdegree[vex_index];
	for(int i=0;i<outgre;++i){
		 int adj = tmp[i];
		if(adj==begin && depth>=3){
			++ans[n];
            path_map[((depth-3)<<CN)+n].emplace_back(*s);
        }
        if(depth<7 && !*(isVisit+adj) && adj>begin){//isNear是保证该邻接点在起点的邻域内，如果在邻域外则不可能通过这个邻点找到环 
        	if(depth<=3){
        		findCircle(adj,begin,isNear,isVisit,s,depth+1,n);
			}
			else{ 
				if(*(isNear+adj)==begin){
					findCircle(adj,begin,isNear,isVisit,s,depth+1,n);
				}
			}
        }
	}
	--s->length;	
	*(isVisit+vex_index)=false;
}
void *findCircleByThread(void *arg){
	/*
	一个线程负责指定区间内的点为起点的找环任务，各个线程互不干扰
	left为左区间索         引，right为右区间索引，左闭右开,即不包括right索引对应的点 
	 n表示这个区间属于8大块里的第几块，由上一层调用函数分配好
	 本函数线程执行不会与其他线程执行产生临界区 
	*/
    
    int left=((ARG_THREAD*)arg)->left;
    int right=((ARG_THREAD*)arg)->right;
    int n=((ARG_THREAD*)arg)->num; 
	bool isVisit[maxsize];//访问状态数组
	int *isNear = new int[cntvex];//记录遍历遇到的点是否在起点的3邻域内，不在邻域内的点直接跳过，不可能组成长度<=7的环 
	 for(int i=0;i<cntvex;++i){
	 	isNear[i]=-1;
	 }
	 stack s;
	 init_stack(&s);
	 for(int i=left;i<right;++i){
		if(!outdegree[i]||!indegree){//出入度是顶点固有属性，不会变 
			continue;
		}
		findNearIn(i,i,isNear,isVisit,1);
		//1为depth，固定起始值为1，第一个i表示进入findcircle的当前点,而第二个i表示当前找环任务是找的以谁为起点 
		findCircle(i,i,isNear,isVisit,&s,1,n);
	}
	 
}
void findAllCircles(){
		//cntvex为点个数，将其按顺序分成八份，设置好标号num为0-7，
    int arg_thread[THREAD_NUM][3];   //arg_thread[0][]  arg_thread是线程传参用
    int num_every[THREAD_NUM];  
    int num_all=0;  
    //求和
	num_all = (1+THREAD_NUM)<<(CN-1);
	//cout<<"sum:"<<num_all<<endl;
    int round=cntvex/num_all;
    if(cntvex%num_all)
        ++round;
    for(int i=0;i<THREAD_NUM;++i){
        if(i==0)
            arg_thread[i][0]=0;
        else 
            arg_thread[i][0]=arg_thread[i-1][1];
        
        if(i!=THREAD_NUM-1)
            arg_thread[i][1]=arg_thread[i][0]+(i+1)*round;
        else
            arg_thread[i][1]=cntvex;
        arg_thread[i][2]=i;    
    }
    for(int i=0;i<THREAD_NUM;i++){
    //	cout<<arg_thread[i][0]<<" "<<arg_thread[i][1]<<" "<<arg_thread[i][2]<<endl;
        arg_threads[i].left=arg_thread[i][0];
        arg_threads[i].right=arg_thread[i][1];
        arg_threads[i].num=arg_thread[i][2];  
    }

    pthread_t thread_id[THREAD_NUM];
    for(int i=0;i<THREAD_NUM;i++)  
        pthread_create(&thread_id[i], NULL, findCircleByThread, (void *)&arg_threads[i]);
    
    for(int i=0;i<THREAD_NUM;i++)
        pthread_join(thread_id[i], NULL);
    
	//并将每一份的left下标,right下标,num传入findCircleByThread函数执行
	//共8个线程执行findCircleByThread() 
}

void save1(string &outputFile){
        //printf("begin to write file,total Loops %d\n",ans);
        FILE *fp;
        if((fp=fopen(outputFile.c_str(),"w"))==NULL){
		printf("The file cannot open.\n");
		exit(0);
	}
	//cout<<"write to buf:"<<endl;
        int bytes = sprintf(buf,"%u\n",ans_sum);
        int offset=bytes;
        //cout<<ans<<endl;
        int num = 5<<CN;
        	for(int k=0;k<num;++k){
        		for( auto temp:path_map[k]){
        			char tmp[128];
        			int off=0;
					int j,len;
				//cout<<data[temp[0]];
    			for(j=0;j<temp.length;++j){
    				len = dataLen[temp.arr[j]];
    			//	bytes=sprintf(tmp+off,",%u",data[temp.arr[j]]);
    				memcpy(tmp+off,data[temp.arr[j]],len);
    				off+=len;
    			//	cout<<","<<data[temp.arr[j]];
	    		}
	    		len = dataLen[temp.arr[j]];
	    		memcpy(tmp+off,data[temp.arr[j]],len-1);
	    		off+=len-1;
	    		memcpy(tmp+off,"\n",1);
	    		++off;
	    		//cout<<off<<":"<<strlen(tmp)<<endl;
	    		memcpy(buf+offset,tmp,off);
	    		offset+=off;
	    		//cout<<endl;
			} 
		}
	//	cout<<"end "<<endl;
		//fseek(fp,0,SEEK_SET);
		memcpy(buf+offset,"\0",1);
		fwrite(buf,offset,1,fp);
		fclose(fp);
		//exit(0);
    }

void sortOutEdge(){
	for(int i=0,degree;i<cntvex;++i){
		degree= outdegree[i];	
		if(degree){
			sort(adjvex[i],adjvex[i]+degree);
		}
	}
}
int main()
{
    //inputFile(submitInputFile);
    //cout<<sizeof(stack)<<endl;
   // long startTime,endTime,endTime2;
    //startTime  = clock();
    inputFile(inputfile);
    //printf("begin to create graph\n");
    creategraph();
   // endTime = clock();
    //print();
    sortOutEdge();
    findAllCircles();
    //save(submitOutputFile);
    int sums=0;
    for(int i=0;i<THREAD_NUM;++i){
    	sums+=ans[i];
	}
	ans_sum=sums;
    //printf("begin to write file,total Loops %d\n",ans_sum);
    //save1(submitOutputFile);
    save1(inputfile);
    //endTime2 = clock();
    //cout<<"find circle time:"<<(double)(endTime-startTime)/CLOCKS_PER_SEC<<"s"<<endl;
    //cout<<"save file time:"<<(double)(endTime2-endTime)/CLOCKS_PER_SEC<<"s"<<endl;
    return 0;
}
