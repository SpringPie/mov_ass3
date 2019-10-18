#include "nmmintrin.h" // for SSE4.2 
#include "immintrin.h" // for AVX  
#include <iostream>

int main()
{
	std::cout << "\n_______________________ TESTING QUADFLOADS _____________________________\n";

	//__m128 x4 = _mm_set_ps(1.0f, 1.0f, 1.0f, 1.0f);
	//union { __m128 mask1; float mask1f[4]; };
	//mask1 = _mm_cmpge_ps(x4, _mm_setzero_ps());
	//std::cout << "Print mask" << mask1[0] << ", " << mask1[1] << ", " << mask1[2] << ", " << mask1[3] << std::endl;

	__m128 x4 = _mm_set_ps(1.0f, 1.0f, 1.0f, 1.0f);
	union { __m128 mask1; unsigned int mask1f[4]; };
	mask1 = _mm_cmpge_ps(x4, _mm_setzero_ps());
	std::cout << "Print mask " << mask1f[0] << ", " << mask1f[1] << ", " << mask1f[2] << ", " << mask1f[3] << std::endl;

	union { int first; float second; };
	first = 2;
	std::cout << "first: " << first <<
		" , second: " << second << " pointer val " << *(int*)& second << std::endl;

	union { __m128 d4; float d[4]; };
	d4 = _mm_set_ps(1.0f, 2.0f, 3.0f, 4.0f);
	std::cout << "quadfloat d4 " << d[0] << ", " << d[1] << ", " << d[2] << ", " << d[3] << std::endl;

	__m128 e4 = _mm_set_ps(1.0f, 2.0f, 3.0f, 4.0f);
	d4 = _mm_mul_ps(d4, e4);
	std::cout << "d4 times e4 " << d[0] << ", " << d[1] << ", " << d[2] << ", " << d[3] << std::endl;

	//Compare floats
	union { __m128 a4; unsigned int a4Array[4]; };
	a4 = _mm_set_ps(22.0f, 0.0f, 22.0f, 22.0f);
	union { __m128 mask2; unsigned int mask2Array[4]; };
	mask2 = _mm_cmpeq_ps(a4, _mm_setzero_ps());
	int movemask1 = _mm_movemask_ps(mask2);
	std::cout << "\n_______________________ MOVEMASK FLOATS _____________________________\n";
	std::cout << "a4 values 0: " << a4Array[0] << ", "<< a4Array[1] << ", " << a4Array[2] << ", " << a4Array[3] << std::endl;
	std::cout << "mask with a4 and zeroes gives " << mask2Array[0] << ", " << mask2Array[1] << ", " << mask2Array[2] << ", " << mask2Array[3] << std::endl;
	std::cout << "moveing that mask with movemask gives " << movemask1 << std::endl;
	std::cout << "4 in bits is 0100" << std::endl;

	//Compare ints
	union { __m128i b4; unsigned int b4Array[4]; };
	b4 = _mm_set_epi32(22, 0, 22, 22);
	union { __m128i mask3; unsigned int mask3Array[4]; };
	mask3 = _mm_cmpeq_epi32(b4, _mm_setzero_si128());
	int movemask2 = _mm_movemask_epi8(mask3);
	std::cout << "\n_______________________ MOVEMASK INTS _____________________________\n";
	std::cout << "b4 values 0: " << b4Array[0] << ", " << b4Array[1] << ", " << b4Array[2] << ", " << b4Array[3] << std::endl;
	std::cout << "mask with b4 and zeroes gives " << mask3Array[0] << ", " << mask3Array[1] << ", " << mask3Array[2] << ", " << mask3Array[3] << std::endl;
	std::cout << "moveing that mask with movemask gives " << movemask2 << std::endl;
	std::cout << "3840 in bits is 0000 1111 0000 0000" << std::endl;
	
}