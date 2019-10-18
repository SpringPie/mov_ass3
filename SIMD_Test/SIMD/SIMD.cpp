#include "nmmintrin.h" // for SSE4.2 
#include "immintrin.h" // for AVX  
#include <iostream>

int main()
{
	printf("hoi");

	//__m128 x4 = _mm_set_ps(1.0f, 1.0f, 1.0f, 1.0f);
	//union { __m128 mask1; float mask1f[4]; };
	//mask1 = _mm_cmpge_ps(x4, _mm_setzero_ps());
	//std::cout << "Print mask" << mask1[0] << ", " << mask1[1] << ", " << mask1[2] << ", " << mask1[3] << "\n";

	__m128 x4 = _mm_set_ps(1.0f, 1.0f, 1.0f, 1.0f);
	union { __m128 mask1; float mask1f[4]; };
	mask1 = _mm_cmpge_ps(x4, _mm_setzero_ps());
	std::cout << "Print mask" << mask1f[0] << ", " << mask1f[1] << ", " << mask1f[2] << ", " << mask1f[3] << "\n";

	union { int first; float second; };
	first = 2;
	std::cout << "first: " << first <<
		" , second: " << second << " pointer val " << *(int*)& second << "\n";

	union { __m128 d4; float d[4]; };
	d4 = _mm_set_ps(1.0f, 2.0f, 3.0f, 4.0f);
	std::cout << "quadfloat d4 " << d[0] << ", " << d[1] << ", " << d[2] << ", " << d[3] << "\n";

	__m128 e4 = _mm_set_ps(1.0f, 2.0f, 3.0f, 4.0f);
	d4 = _mm_mul_ps(d4, e4);
	std::cout << "d4 times e4 " << d[0] << ", " << d[1] << ", " << d[2] << ", " << d[3] << "\n";

	__m128 c4 = _mm_min_ps(e4, d4);
	std::cout << "min value e4 and d4 " << c4[0] << ", " << c4[1] << ", " << c4[2] << ", " << c4[3] << "\n";
}