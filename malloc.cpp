#include <iostream>
#include <fstream>
#include <cstring>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <set>
#include <iterator>
#include <algorithm>
#include <CL/cl.h>
#include <ctime>

#define HA 1
#define WA 784
#define WB 512
#define WD 10

#define HB WA
#define HC HA
#define WC WB
#define HD WC
#define HE HC
#define WE WD

using namespace std;

clock_t start_t, end_t, total_t;
const string training_image_fn = "mnist/train-images.idx3-ubyte";
const string training_label_fn = "mnist/train-labels.idx1-ubyte";
const string model_fn = "model-neural-network.dat";
const string report_fn = "training-report.dat";
const int nTraining = 60000;
const int width = 28;
const int height = 28;

const int n1 = width * height;
const int n2 = 512; 
const int n3 = 10;
const int epochs = 15;
const float learning_rate = 1e-2;
const float momentum = 0.9;
const float epsilon = 1e-5;
int img = 0;

// From layer 1 to layer 2. Or: Input layer - Hidden layer
float *w1[n1 + 1], *delta1[n1 + 1], *out1;

// From layer 2 to layer 3. Or; Hidden layer - Output layer
float *w2[n2 + 1], *delta2[n2 + 1], *in2, *out2, *theta2;

// Layer 3 - Output layer
float *in3, *out3, *theta3;
float expected[n3 + 1];

// Image. In MNIST: 28x28 gray scale images.
float d[nTraining + 1][width + 1][height + 1];
float outA[nTraining + 1][n1 + 1];

// File stream to read data (image, label) and write down a report
ifstream image;
ifstream label;
ofstream report;


// Memory allocation for the network

void init_array() {
	// Layer 1 - Layer 2 = Input layer - Hidden layer
    for (int i = 1; i <= n1; ++i) {
        w1[i] = new float [n2 + 1];
        delta1[i] = new float [n2 + 1];
    }
    
    out1 = new float [n1 + 1];

	// Layer 2 - Layer 3 = Hidden layer - Output layer
    for (int i = 1; i <= n2; ++i) {
        w2[i] = new float [n3 + 1];
        delta2[i] = new float [n3 + 1];
    }
    
    in2 = new float [n2 + 1];
    out2 = new float [n2 + 1];
    theta2 = new float [n2 + 1];

	// Layer 3 - Output layer
    in3 = new float [n3 + 1];
    out3 = new float [n3 + 1];
    theta3 = new float [n3 + 1];
    
    // Initialization for weights from Input layer to Hidden layer
    for (int i = 1; i <= n1; ++i) {
        for (int j = 1; j <= n2; ++j) {
            int sign = rand() % 2;

            // Another strategy to randomize the weights - quite good 
            // w1[i][j] = (float)(rand() % 10 + 1) / (10 * n2);
            
            w1[i][j] = (float)(rand() % 6) / 10.0;
            if (sign == 1) {
				w1[i][j] = - w1[i][j];
			}
        }
	}
	
	// Initialization for weights from Hidden layer to Output layer
    for (int i = 1; i <= n2; ++i) {
        for (int j = 1; j <= n3; ++j) {
            int sign = rand() % 2;
			
			// Another strategy to randomize the weights - quite good 
            // w2[i][j] = (float)(rand() % 6) / 10.0;

            w2[i][j] = (float)(rand() % 10 + 1) / (10.0 * n3);
            if (sign == 1) {
				w2[i][j] = - w2[i][j];
			}
        }
	}
}

// Reading input - gray scale image and the corresponding label

//void input() {}

// Load OpenCL Kernel

long LoadOpenCLKernel(char const* path, char **buf)
{
    FILE  *fp;
    size_t fsz;
    long   off_end;
    int    rc;

    /* Open the file */
    fp = fopen(path, "r");
    if( NULL == fp ) {
        return -1L;
    }

    /* Seek to the end of the file */
    rc = fseek(fp, 0L, SEEK_END);
    if( 0 != rc ) {
        return -1L;
    }

    /* Byte offset to the end of the file (size) */
    if( 0 > (off_end = ftell(fp)) ) {
        return -1L;
    }
    fsz = (size_t)off_end;

    /* Allocate a buffer to hold the whole file */
    *buf = (char *) malloc( fsz+1);
    if( NULL == *buf ) {
        return -1L;
    }

    /* Rewind file pointer to start of file */
    rewind(fp);

    /* Slurp file into buffer */
    if( fsz != fread(*buf, 1, fsz, fp) ) {
        free(*buf);
        return -1L;
    }

    /* Close the file */
    if( EOF == fclose(fp) ) {
        free(*buf);
        return -1L;
    }


    /* Make sure the buffer is NUL-terminated, just in case */
    (*buf)[fsz] = '\0';

    /* Return the file size */
    return (long)fsz;
}

// Allocates a matrix with random float entries.
void randomMemInit(float* data, int size)
{
   int i;

   for (i = 0; i < size; ++i)
   	data[i] = (float)(rand() % 10 + 1) / (1000);
}

// Main Program

int main(int argc, char *argv[]) {
   start_t = clock();
   int err;
   cl_device_id device_id;             // compute device id 
   cl_context context;                 // compute context
   cl_command_queue commands;          // compute command queue
   cl_program program;                 // compute program
   cl_kernel kernel;                   // compute kernel

   cl_mem d_A;
   cl_mem d_B;
   cl_mem d_C;
   cl_mem d_D;
   cl_mem d_E;

    report.open(report_fn.c_str(), ios::out);
    image.open(training_image_fn.c_str(), ios::in | ios::binary); // Binary image file
    label.open(training_label_fn.c_str(), ios::in | ios::binary ); // Binary label file

	// Reading file headers
    char number;
    for (int i = 1; i <= 16; ++i) {
        image.read(&number, sizeof(char));
	}
    for (int i = 1; i <= 8; ++i) {
        label.read(&number, sizeof(char));
	}
		
	// Neural Network Initialization
    init_array();
   
    for (int sample = 1; sample <= 10/*nTraining*/; ++sample) {
        cout << "Sample " << sample << endl;
        
   // Getting (image, label)
   //input();		
    
	// Reading image
    char number;
    for (int j = 1; j <= height; ++j) {
        for (int i = 1; i <= width; ++i) {
            image.read(&number, sizeof(char));
            if (number == 0) {
				d[sample][i][j] = 0;
			} else {
				d[sample][i][j] = 1;
			}
        }
	}

/*~	cout << "Image:" << endl;
	for (int j = 1; j <= height; ++j) {
		for (int i = 1; i <= width; ++i) {
			cout << d[sample][i][j];
		}
		cout << endl;
	}
	cout<<endl;
*/
    for (int j = 1; j <= height; ++j) {
        for (int i = 1; i <= width; ++i) {
            int pos = i + (j - 1) * width;
            outA[sample][pos] = d[sample][i][j];
/*~	    cout << outA[sample][pos];*/
        }
	}

	// Reading label
    label.read(&number, sizeof(char));
    for (int i = 1; i <= n3; ++i) {
		expected[i] = 0.0;
	}
    expected[number + 1] = 1.0;
    cout << endl;
    cout << "Label: " << (int)(number) << endl;



   //Allocate host memory for matrices A and B
   unsigned int size_A = WA * HA;
   unsigned int mem_size_A = sizeof(float) * size_A;
   float* h_A = (float*) malloc(mem_size_A);
 
   unsigned int size_B = WB * HB;
   unsigned int mem_size_B = sizeof(float) * size_B;
   float* h_B = (float*) malloc(mem_size_B);

   //Allocate host memory for the result C
   unsigned int size_C = WC * HC;
   unsigned int mem_size_C = sizeof(float) * size_C;
   float* h_C = (float*) malloc(mem_size_C);

   unsigned int size_D = WD * HD;
   unsigned int mem_size_D = sizeof(float) * size_D;
   float* h_D = (float*) malloc(mem_size_D);

   unsigned int size_E = WE * HE;
   unsigned int mem_size_E = sizeof(float) * size_E;
   float* h_E = (float*) malloc(mem_size_E);

   //Initialize host memory
   h_A = outA[sample];
   randomMemInit(h_B, size_B);
   randomMemInit(h_D, size_D);

   printf("Initializing OpenCL device...\n"); 

   cl_uint dev_cnt = 0;
   clGetPlatformIDs(0, 0, &dev_cnt);
	
   cl_platform_id platform_ids[100];
   clGetPlatformIDs(dev_cnt, platform_ids, NULL);
	
   // Connect to a compute device
   int gpu = 1;
   err = clGetDeviceIDs(platform_ids[0], gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);
   if (err != CL_SUCCESS)   { printf("Error: Failed to create a device group!\n");    return EXIT_FAILURE;   }
  
   // Create a compute context 
   context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
   if (!context)    {       printf("Error: Failed to create a compute context!\n");        return EXIT_FAILURE;   }

   // Create a command commands
   commands = clCreateCommandQueue(context, device_id, 0, &err);
   if (!commands)   {   printf("Error: Failed to create a command commands!\n");     return EXIT_FAILURE;   }

   // Create the compute program from the source file
   char *KernelSource;
   long lFileSize;

   lFileSize = LoadOpenCLKernel("malloc_kernel.cl", &KernelSource);
   if( lFileSize < 0L ) {       perror("File read failed");       return 1;   }

   program = clCreateProgramWithSource(context, 1, (const char **) & KernelSource, NULL, &err);
   if (!program)   {       printf("Error: Failed to create compute program!\n");       return EXIT_FAILURE;   }

   // Build the program executable
   err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
   if (err != CL_SUCCESS)   {   size_t len;       char buffer[2048];       printf("Error: Failed to build program executable!\n");
       clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len); printf("%s\n", buffer);  exit(1);   }

   // Create the compute kernel in the program we wish to run
   kernel = clCreateKernel(program, "matrixMul", &err);
   if (!kernel || err != CL_SUCCESS)   {       printf("Error: Failed to create compute kernel!\n");       exit(1);   }

   // Create the input and output arrays in device memory for our calculation
   d_C = clCreateBuffer(context, CL_MEM_READ_WRITE, mem_size_A, NULL, &err);
   d_A = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mem_size_A, h_A, &err);
   d_B = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mem_size_B, h_B, &err);
   d_D = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mem_size_D, h_D, &err);
   d_E = clCreateBuffer(context, CL_MEM_READ_WRITE, mem_size_C, NULL, &err);
   if (!d_A || !d_B || !d_C || !d_D || !d_E)   {   printf("Error: Failed to allocate device memory!\n");       exit(1);   }    

   printf("Running matrix multiplication for matrices A (%dx%d) and B (%dx%d) ...\n", HA,WA,HB,WB); 

   //Launch OpenCL kernel
   size_t localWorkSize[2], globalWorkSize[2];
 
   int wA = WA;
   int wC = WC;
   int wB = WB;
   int wD = WD;
   int wE = WE;
   err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&d_C);
   err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&d_A);
   err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&d_B);
   err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&d_D);
   err |= clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&d_E);
   err |= clSetKernelArg(kernel, 5, sizeof(int), (void *)&wA);
   err |= clSetKernelArg(kernel, 6, sizeof(int), (void *)&wB);
   err |= clSetKernelArg(kernel, 7, sizeof(int), (void *)&wC);
   err |= clSetKernelArg(kernel, 8, sizeof(int), (void *)&wD);
   err |= clSetKernelArg(kernel, 9, sizeof(int), (void *)&wE);
   if (err != CL_SUCCESS)   {       printf("Error: Failed to set kernel arguments! %d\n", err);       exit(1);   }
 
   localWorkSize[0] = 1; //16
   localWorkSize[1] = 1; 
   globalWorkSize[0] = 512; //1024
   globalWorkSize[1] = 512;

   err = clEnqueueNDRangeKernel(commands, kernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
   if (err != CL_SUCCESS)   {   printf("Error: Failed to execute kernel! %d\n", err);    exit(1);   }
 
   //Retrieve result from device
   err = clEnqueueReadBuffer(commands, d_C, CL_TRUE, 0, mem_size_C, h_C, 0, NULL, NULL);
   if (err != CL_SUCCESS)   {   printf("Error: Failed to read output array! %d\n", err);     exit(1);   }
 
   //print out the results
///*~
   printf("\n\nMatrix E (n)\Results ");
   int i;
   for(i = 0; i < size_C; i++)
   {
      printf("%f ", h_C[i]);
      if(((i + 1) % WC) == 0)
      printf("\n");
   }
   printf("\n");
//*/  
   printf("Matrix multiplication completed...\n"); 

   end_t = clock();
   double total_t = (end_t - start_t)/(double) CLOCKS_PER_SEC;
   cout << "Time: " << total_t << "\n";
   //Shutdown and cleanup
//   free(h_A);
//   free(h_B);
//   free(h_C);
 
   clReleaseMemObject(d_A);
   clReleaseMemObject(d_C);
   clReleaseMemObject(d_B);

   clReleaseProgram(program);
   clReleaseKernel(kernel);
   clReleaseCommandQueue(commands);
   clReleaseContext(context);
}

    return 0;
}
