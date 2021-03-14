/*
�߽ڵ������ adjvexΪ���������е��±�
�������ظ�ת�˼�¼�����ر� 
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
typedef struct{//ջ
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
vector<stack > path_map[5<<CN];//��k���ȵ�N�ݵ�·�� ��Ӧ�±�Ϊk*size+n,eg:n=7,k=4,res=4*8+7=39 
char buf[300000000];
unsigned char bitmap[BITSIZE];//λͼ 
unordered_map<elemtype,int> ma;//ID-->index(newID) ,���ｫ��Ӧ����������Ϊ�µ�ID���붥���data�У��ڴ����ļ�ʱ����ӳ��ؾ�ID 
elemtype inputArr[560000];
elemtype extra[560000];//��ų�����ֵ��IDֵ 
 string submitInputFile= "/data/test_data.txt";
 string submitOutputFile= "/projects/student/result.txt";
 string inputfile= "./testData/hwdata/1004812/test_data.txt";
 string outputfile="./testData/hwdata/1004812/out1.txt";
int ans[THREAD_NUM]={0};
int cntvex=0;//ͳ��ID���� 
int cntedge=0;//�ߵĸ��� 
int thread_n=THREAD_NUM;//�̸߳��� 
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
inline void createBitmap( int arraySize,int &len){//extra��IDֵ������ʾ��Χ���� 
        int i;
	//��array�е�ÿ��������Ӧ��bit�±�����Ϊ1
        for(i=0; i<arraySize; ++i){
        	if(inputArr[i]>MAX_ID){//�������������ֵ����λͼ�Ų��£�������ӵ�extra���飬�������� 
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
	//��ID��һ��һ�� 
	cntedge = i>>1;//�ߵĸ�����Ҳ��inputArr�ĳ��ȵ�һ�� 
	int extraLen=0;//��¼extra���� 
	createBitmap(i,extraLen);
	//printf("inputArr size is:%d",inputArr.size());
	unsigned char mask = 0x1;
    int j;
    int x;
	//����ÿ��unsigned char
	if(extraLen<maxsize){
		for(i=0; i<BITSIZE; ++i){
		//����ÿ��unsigned char�е�bit
        for(j=7; j>=0;j--){
			//���ָ����unsigned char bitmap[i]�ĵ�jλ���ӵ�λ����λ������Ϊ0�������bit���±�
            if((bitmap[i] & (0x1<<j)) != 0){
                x = (unsigned int)(i*8+(7-j));//�ҵ�һ���µ�ID 
                ma[x]=cntvex;
				dataLen[cntvex]=sprintf(data[cntvex],"%u,",x);
				++cntvex;
            	}
        	}
    	}
	}
    if(extraLen){
    	//extraLen��Ϊ0˵�����д���MAX_ID��ID�������������ȥ��
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
//��ӡͼ��Ϣ
void print()
{
    printf("graph struct��\n");
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
	   �������ҵ���beginΪ��������С�ڵ���3����ߵ�,���������isNear��
	   ���������������������beginΪ���Ļ�ʱ���ɸ���isNear�Ƿ����begin�жϱ������ĵ��Ƿ��ڸ������У�������ڣ�
	   ��ͨ���õ㲻�����ҵ���beginΪ���ĳ���<=7�Ļ�,����ֱ���������Դ˴ﵽ��֦��Ŀ�� 
	*/
	*(isVisit+vex_index) = true;
	register auto tmp = adjIn[vex_index];
	int indeg = indegree[vex_index];
	for(int i=0;i<indeg;++i){
			int adj = tmp[i];
			if(!*(isVisit+adj)&&begin<adj){
				if(depth<=3){//�����������ʱ
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
        if(depth<7 && !*(isVisit+adj) && adj>begin){//isNear�Ǳ�֤���ڽӵ������������ڣ�������������򲻿���ͨ������ڵ��ҵ��� 
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
	һ���̸߳���ָ�������ڵĵ�Ϊ�����һ����񣬸����̻߳�������
	leftΪ��������         ����rightΪ����������������ҿ�,��������right������Ӧ�ĵ� 
	 n��ʾ�����������8�����ĵڼ��飬����һ����ú��������
	 �������߳�ִ�в����������߳�ִ�в����ٽ��� 
	*/
    
    int left=((ARG_THREAD*)arg)->left;
    int right=((ARG_THREAD*)arg)->right;
    int n=((ARG_THREAD*)arg)->num; 
	bool isVisit[maxsize];//����״̬����
	int *isNear = new int[cntvex];//��¼���������ĵ��Ƿ�������3�����ڣ����������ڵĵ�ֱ����������������ɳ���<=7�Ļ� 
	 for(int i=0;i<cntvex;++i){
	 	isNear[i]=-1;
	 }
	 stack s;
	 init_stack(&s);
	 for(int i=left;i<right;++i){
		if(!outdegree[i]||!indegree){//������Ƕ���������ԣ������ 
			continue;
		}
		findNearIn(i,i,isNear,isVisit,1);
		//1Ϊdepth���̶���ʼֵΪ1����һ��i��ʾ����findcircle�ĵ�ǰ��,���ڶ���i��ʾ��ǰ�һ��������ҵ���˭Ϊ��� 
		findCircle(i,i,isNear,isVisit,&s,1,n);
	}
	 
}
void findAllCircles(){
		//cntvexΪ����������䰴˳��ֳɰ˷ݣ����úñ��numΪ0-7��
    int arg_thread[THREAD_NUM][3];   //arg_thread[0][]  arg_thread���̴߳�����
    int num_every[THREAD_NUM];  
    int num_all=0;  
    //���
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
    
	//����ÿһ�ݵ�left�±�,right�±�,num����findCircleByThread����ִ��
	//��8���߳�ִ��findCircleByThread() 
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
