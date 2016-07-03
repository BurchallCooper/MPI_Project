// Distributed two-dimensional Discrete FFT transform
// W B Cooper
// Project 1


#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <math.h>
#include <mpi.h>
#include <cstdlib>
#include "Complex.h"
#include "InputImage.h"

using namespace std;


bool DEBUG = true; // to print out the data after each operation if true
bool idft = true;  // perform idft after dft if true

string finalOutput_dft = "WBCooperResult.txt";
string finalOutput_idft= "WBCooperResult_idft.txt";

void Transform1D(Complex* h, int w, Complex* H, bool idft);
void Transpose( Complex* a, Complex* b, int width, int height);
void WriteImageData(const char* newFileName, Complex* d, int w, int h);
void Broadcast_data_to_all_cpus(Complex* after1D, Complex* before2D, int width, int height, bool DEBUG);
void Send_all_data_to_CPU_zero(Complex* after2D, Complex* result2D, int width, int height);
void debug(string filename, Complex* array1, Complex* array2, int width, int height);
void Transform_test(Complex* h, int w, Complex* H); //just a copy function

int main(int argc, char** argv)
{
	int rc = MPI_Init(&argc,&argv);
	if (rc != MPI_SUCCESS) {
		printf ("Error starting MPI program. Terminating.\n");
		MPI_Abort(MPI_COMM_WORLD, rc);
	}


	int numTasks,rank;
	MPI_Comm_size(MPI_COMM_WORLD, &numTasks);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);

// read the  file
	string inputFN("Tower.txt"); // default file name
	if (argc > 1) inputFN = string(argv[1]);  // if name specified on cmd line
	InputImage image(inputFN.c_str()); // Create the helper object for reading the image
	int height = image.GetHeight();
	int width = image.GetWidth();
	Complex* data1 = image.GetImageData();
	Complex  data2[width*height]; // create a working array

	if (DEBUG) debug("000_original_array_rank",data1, data2, width, height);

// perform the dft
//	Transform_test(data1, width, data2);
	Transform1D(data1, width, data2, false);								//transform rows
	if (DEBUG) debug("001_after1D_rank",data2, data1, width, height);
	Broadcast_data_to_all_cpus (data2, data1, width, height, DEBUG);			//distribute data to all cpus
	if (DEBUG) debug("002_after_broadcast_before2D_rank",data1, data2, width, height);
	Transpose(data1, data2, width, height);								//transpose matrix
	if (DEBUG) debug("003_after_transpose_before2D_rank",data2, data1, width, height);
	Transform1D(data2, width, data1, false);
//	Transform_test(data2, width, data1);								//transform columns
	if (DEBUG) debug("004_after2D_rank", data1, data2, width, height);

// if idft is false have all the cpus send data to cpu zero and
// do not perform the inverse DFT

	if (idft == false)
	{
		Send_all_data_to_CPU_zero(data1, data2, width, height);			//send data to cpu zero
		if (rank == 0)
		{
			if (DEBUG) debug("005_after_sendtozero_after2D_rank",data2, data1, width, height);
			Transpose(data2, data1, width, height);						//transpose matrix
			if (DEBUG) debug("006_after_transpose_after2D_rank", data1, data2, width, height);
			WriteImageData(finalOutput_dft.c_str(), data1, width, height);	//save dft
		}
	}

// if idft is true all the cpus will need the data
// perform the inverse DFT after writing the dft

	else
	{
		Broadcast_data_to_all_cpus (data1, data2, width, height, DEBUG);		//send data to all cpus
		if (DEBUG) debug("005_after_broadcast_after2D_rank", data2, data1, width, height);
		Transpose(data2, data1, width, height);							//transpose matrix
		if (DEBUG) debug("006_after_transpose_after2D_rank", data1, data2, width, height);
		if (rank == 0) WriteImageData(finalOutput_dft.c_str(), data1, width, height);		//save dft transform


// now perform the inverse dft

//		Transform_test(data1, width, data2);
		Transform1D(data1, width, data2, true);							//transform matrix
		if (DEBUG) debug("inv_001_after1D_rank",data2, data1, width, height);
		Broadcast_data_to_all_cpus (data2, data1, width, height, DEBUG);		//distribute data to all cpus
		if (DEBUG) debug("inv_002_after_broadcast_before2D_rank",data1, data2, width, height);
		Transpose(data1, data2, width, height);							//transpose matrix
		if (DEBUG) debug("inv_003_after_transpose_before2D_rank",data2, data1, width, height);
//		Transform_test(data2, width, data1);
		Transform1D(data2, width, data1, true);							//transform columns
		if (DEBUG) debug("inv_004_after2D", data1, data2, width, height);
		Send_all_data_to_CPU_zero(data1, data2, width, height);			//send data to cpu zero
		if (rank == 0)
		{
			if (DEBUG) debug("inv_005__after_sendtozero_after2D_rank",data2, data1, width, height);
			Transpose(data2, data1, width, height);						//transpose matrix
			if (DEBUG) debug("inv_006_after_transpose_after2D_rank", data1, data2, width, height);
			WriteImageData(finalOutput_idft.c_str(), data1, width, height);	//save dft
		}
	}

	MPI_Finalize();
	return(0);
}

Complex getwRaisetoNK(Complex W,double N,double K){
	Complex result = W;
	for (int var = 0; var < N*K; ++var) {
		result=result*W;
	}
	return result;
}

// Implement a 1-d DFT using the double summation equation
// given in the assignment handout.  h is the time-domain input
// data, w is the width and H is the output array

void Transform1D(Complex* h, int w, Complex* H, bool idft)
{
	int numTasks,rank;
	MPI_Comm_size(MPI_COMM_WORLD, &numTasks);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);

	//Determining the start and the end rows for which the current rank has to calculate the DFT
	const int rowsPerRank = w/numTasks;
	const int startRow = rowsPerRank*rank;
	const int endRow = (rowsPerRank*(rank+1)-1);


	//Creating the loops to calculate the 1D FFT of the designated rows
	for(int i = startRow; i <= endRow; i++)
	{

		for(int n = 0; n < w; n++)
		{

			H[i*w+n] = Complex(0,0);

			for(int k = 0;  k < w; k++)
			{
				double term = -(2*M_PI*n*k)/w;
				if (idft == true) term = -term;
				double cosTerm = cos(term);
				double sinTerm = sin(term);

				Complex W(cosTerm,sinTerm);
				H[i*w+n] = H[i*w+n] + (W*h[i*w+k]);

			}
			if (idft == true) H[i*w+n] = H[i*w+n]/(Complex(w,0));
		}


	}

}

void Transform_test(Complex* h, int w, Complex* H)
{
	int numTasks,rank;
	MPI_Comm_size(MPI_COMM_WORLD, &numTasks);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);

	//Determining the start and the end rows for which the current rank has to calculate the DFT
	const int rowsPerRank = w/numTasks;
	const int startRow = rowsPerRank*rank;
	const int endRow = (rowsPerRank*(rank+1)-1);

//	cout << "start row " << startRow << " end row " << endRow << " w " << w <<endl;

	//Creating the loops to calculate the 1D FFT of the designated rows
	for(int i = startRow; i <= endRow; i++)
	{

		for(int n = 0; n < w; n++)
		{
			H[i*w+n] = h[i*w+n];
		}
	}
}

void Transpose( Complex* a, Complex* b, int width, int height)
{
	for (int i = 0; i < width; i++)
	{
		for (int j = 0; j < height; j++)
		{
			b[j*width+i] = a[i*width+j];
		}
	}
}

void WriteImageData(const char* newFileName, Complex* d,
                               int w, int h)
{
  ofstream ofs(newFileName);
  if (!ofs)
    {
      cout << "Can't create output image " << newFileName << endl;
      return;
    }
  ofs << w << " " << h << endl;
  for (int r = 0; r < h; ++r)
    { // for each row
      for (int c = 0; c < w; ++c)
        { // for each column
          ofs << d[r * w + c].Mag() << " ";
        }
      ofs << endl;
    }
}


void Broadcast_data_to_all_cpus(Complex* after1D, Complex* before2D, int width, int height, bool DEBUG)
{
	// send all data to all the n-1 Cpu's

	//Determining the start and the end rows for which the current rank has to calculate the DFT

	int numTasks,rank;
	MPI_Comm_size(MPI_COMM_WORLD, &numTasks);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	int numRowsPerRank = width/numTasks;
//	int startRow = numRowsPerRank*rank;
//	int endRow = (numRowsPerRank*(rank+1)-1);

	//	cout <<numRowsPerRank <<" "<<startRow<<" "<<endRow<<endl;

	// Non Blocking Send

	// Sending to all the partner CPU everything
	//	MPI_Status status_transmit;

	MPI_Request request[numTasks];
	MPI_Status status[numTasks];

	for (int cpuNum = 0; cpuNum < numTasks; ++cpuNum) {
		int rc = MPI_Isend(&after1D[rank*width*numRowsPerRank], sizeof(Complex)*width*numRowsPerRank , MPI_CHAR, cpuNum,
				0, MPI_COMM_WORLD, &request[cpuNum]);
		if (rc != MPI_SUCCESS)
		{
			cout << "Rank " << rank
					<< " send failed, rc " << rc << endl;
			MPI_Finalize();
			exit(1);
		}


	}
	// Now Blocking Receive


	for (int cpuNum = 0; cpuNum < numTasks; ++cpuNum)
	{
		MPI_Status status_rx;
		int rc = MPI_Recv(&before2D[cpuNum*width*numRowsPerRank], sizeof(Complex)*width*numRowsPerRank , MPI_CHAR, cpuNum,
				0, MPI_COMM_WORLD, &status_rx);
		if (rc != MPI_SUCCESS)
		{
			cout << "Rank " << rank
					<< " send failed, rc " << rc << endl;
			MPI_Finalize();
			exit(1);
		}

		 int count = 0;
         // The receive is now completed and data available.

		if (DEBUG)
		{
			MPI_Get_count(&status_rx, MPI_CHAR, &count);
			cout << "**** Rank " << rank
				<< " received " << count << " bytes from "
				<< status_rx.MPI_SOURCE << endl;
		}
	}

		MPI_Waitall (numTasks, request, status);
}

void Send_all_data_to_CPU_zero(Complex* after2D, Complex* result2D, int width, int height)
{
	// send all data to CPU zero

	//Determining the start and the end rows for which the current rank has to calculate the DFT

	int numTasks,rank;
	MPI_Comm_size(MPI_COMM_WORLD, &numTasks);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	int numRowsPerRank = width/numTasks;
//	int startRow = numRowsPerRank*rank;
//	int endRow = (numRowsPerRank*(rank+1)-1);

// Sending the data from all CPU to CPU 0

	MPI_Request request;
	MPI_Status status;
	int rc = MPI_Isend(&after2D[rank*numRowsPerRank*width], sizeof(Complex)*numRowsPerRank*width , MPI_CHAR, 0,
			0, MPI_COMM_WORLD, &request);
	if (rc != MPI_SUCCESS)
	{
		cout << "Rank " << rank
				<< " send failed, rc " << rc << endl;
		MPI_Finalize();
		exit(1);
	}

	if (rank == 0) {
		// then collect all the data and save it to a file
		for (int cpuNum = 0; cpuNum < numTasks; ++cpuNum) {
			MPI_Status status_rx;
			int rc = MPI_Recv(&result2D[cpuNum*numRowsPerRank*width],sizeof(Complex)*numRowsPerRank*width, MPI_CHAR, cpuNum ,
					0, MPI_COMM_WORLD, &status_rx);
			if (rc != MPI_SUCCESS)
			{
				cout << "Rank " << rank
						<< " send failed, rc " << rc << endl;
				MPI_Finalize();
				exit(1);
			}
		}
	}
	MPI_Waitall (1, &request, &status);
}

void debug(string filename,Complex* array1, Complex* array2, int width,int height)
{
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	ostringstream rankStr;
	rankStr << rank;
	WriteImageData(filename.append(rankStr.str()).append(".wbctxt").c_str(),array1,width,height);
	cout << "Done with 1d fft in rank" << rank <<
			"  Printing the result in File: "<<filename<< endl;

//  clear array for readability

	for (int i = 0; i < width*height; i++)
	{
		array2[i] = Complex(0,0);
	}
}



//**********************************************************************

// Constructors


Complex::Complex()
    : real(0), imag(0), NaN(false)
{
}

Complex::Complex(bool n)
    : real(0), imag(0), NaN(n)
{
}

Complex::Complex(double r)
    : real(r), imag(0), NaN(false)
{
}

Complex::Complex(double r, double i)
    : real(r), imag(i), NaN(false)
{
}

// Operators
Complex Complex::operator+(const Complex& b) const
{
  if (NaN || b.NaN) return Complex(true); // Not a number
  return Complex(real + b.real, imag + b.imag);
}

Complex Complex::operator-(const Complex& b) const
{
  if (NaN || b.NaN) return Complex(true); // Not a number
  return Complex(real - b.real, imag - b.imag);
}

Complex Complex::operator*(const Complex& b) const
{
  if (NaN || b.NaN)  return Complex(true); // NaN
  return Complex(real*b.real - imag*b.imag,
                 real*b.imag + imag*b.real);
}

Complex Complex::operator/(const Complex& b) const
{
  if (NaN || b.NaN)  return Complex(true); // NaN
  Complex tmp = (*this) * b.Conj();
  Complex magSquared = b.Mag() * b.Mag();
  if (magSquared.Mag().real == 0.0)
    {
      return Complex(true); // Nan
    }
  return Complex(tmp.real/magSquared.real, tmp.imag/magSquared.real);
}

// Member functions
Complex Complex::Mag() const
{
  if (NaN) return Complex(true); // Not a number
  return Complex(sqrt(real*real + imag*imag));
}

Complex Complex::Angle() const
{
  if (NaN) return Complex(true); // Not a number
  if (Mag().real == 0) return Complex(true);
  return Complex(atan2(imag, real) * 360 / (2 * M_PI));
}

Complex Complex::Conj() const
{ // Return to complex conjugate
  if (NaN) return Complex(true); // Not a number
  return Complex(real, -imag);
}

void Complex::Print() const
{
  if (NaN)
    {
      cout << "NaN";
    }
  else
    {
      if (imag == 0)
        { // just real part with no parens
          cout << real;
        }
      else
        {
          cout << '(' << real << "," << imag << ')';
        }
    }
}

// Global function to output a Complex value
std::ostream& operator << (std::ostream &os, const Complex& c)
{
  if (c.NaN)
    {
      os << "NaN";
    }
  else
    {
      if (c.imag == 0)
        { // just real part with no parens
          os << c.real;
        }
      else
        {
          os << '(' << c.real << "," << c.imag << ')';
        }
    }
  return os;
}
InputImage::InputImage(const char* fileName)
{
  ifstream ifs(fileName);
  if (!ifs)
    {
      cout << "Can't open image file " << fileName << endl;
      exit(1);
    }
  ifs >> w >> h;
  data = new Complex[w * h]; // Allocate the data array
  for (int r = 0; r < h; ++r)
    { // For each row
      for (int c = 0; c < w; ++c)
        {
          double real;
          ifs >> real;
          data[r * w + c] = Complex((double)real);
        }
    }
}

int InputImage::GetWidth() const
{
  return w;
}

int InputImage::GetHeight() const
{
  return h;
}

Complex* InputImage::GetImageData() const
{
  return data;
}

void InputImage::SaveImageData(const char* newFileName, Complex* d,
                               int w, int h)
{
  ofstream ofs(newFileName);
  if (!ofs)
    {
      cout << "Can't create output image " << newFileName << endl;
      return;
    }
  ofs << w << " " << h << endl;
  for (int r = 0; r < h; ++r)
    { // for each row
      for (int c = 0; c < w; ++c)
        { // for each column
          ofs << d[r * w + c].Mag() << " ";
        }
      ofs << endl;
    }
}
