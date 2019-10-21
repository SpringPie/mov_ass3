#include "nmmintrin.h" // for SSE4.2 
#include "immintrin.h" // for AVX  
#include <iostream>

int main()
{
    std::cout << "Hello World!\n";

	// testing: a union without a name. 
	union { int a; float b; };
	a = 4; 
	std::cout << "a: " << a << 
		" , b: " << b << 
		", interpreted float b as an int, by dereferencing a transformed int pointer: *(int*)&b: " << *(int*)& b << "\n";

	// example 2, union with int and char
	union { unsigned int c4; unsigned char c[4]; };

	c[0] = 'a'; // somehow correspond to lowest byte in integer
	c[1] = 0;
	c[2] = 0;
	c[3] = 0;
	int aChar_id = *(int*)c;
	std::cout << "\nanother union, putting aChar_id in an int first (" << c[0] << ")" << "as int: ";
	std::cout << c4 << "\n";
	c4 = aChar_id + (aChar_id + 1 << 8) + (aChar_id + 2 << 16) + (aChar_id + 3  << 24);
	

	std::cout << "\nc4: "<< c4 <<
		" , c[0]: " << c[0] << ", \tc[1]: " << c[1] << ", \tc[2]: " << c[2] << ", \tc[3] : " << c[3] << "\n";


	// A __m128 variable contains four floats, so we can use the union trick again :
	union { __m128 d4; float d[4]; };
	d4 = _mm_set_ps(0.2f, 0.1f, 0.38f, 0.32f); 
	std::cout << "\n_______________________ SIMD _____________________________\n";
	std::cout << "__m128 float union, filled __m128 with: (4.0f, 4.1f, 4.2f, 4.3f), " <<
		"we are not allowed to print the content of __M128 directly\n" <<
		"printing the float values (reading from array d[4]): " << d[0] << ", " << d[1] << ", " << d[2] << ", " << d[3] << "\n";
	
	
	// we can create a quadfloat directly
	__m128 e4 = _mm_set_ps(255.9, 255.9f, 255.9f, 255.9f);
	d4 = _mm_mul_ps(d4, e4);
	std::cout << "After creating a __m128 filled with 4 times 255.9 values, for. e.g. scaling to a float to a color, " << 
		"we can use it to multiply with d4, which results in: " << d[0] << ", " << d[1] << ", " << d[2] << ", " << d[3] << "\n";

	/* 
	
	 _mm_add_ps( a4, b4 ); 
	 _mm_sub_ps( a4, b4 ); 
	 _mm_mul_ps( a4, b4 ); 
	 _mm_div_ps( a4, b4 ); 
	 _mm_sqrt_ps( a4 ); 
	 _mm_rcp_ps( a4 ); // reciprocal 
	 A full overview of SSE and AVX instructions can be found here:
	https://software.intel.com/sites/landingpage/IntrinsicsGuide/
	*/


	/* from tutorial from Jacco: 
	
	// min and max functions
	__m128 c4 = _mm_min_ps( a4, b4 ); 
	__m128 c4 = _mm_max_ps( a4, b4 ); 
	
	// float to int conversion 
	union { __m128i tmp1; int oxi[4]; };
	tmp1 = _mm_cvtps_epi32( ox4 ); 

	// SSE and AVX do not have if statements, but they do in fact have comparison instructions. 
	// These do not yield ‘quadbools’, but they do return something useful: bitmasks. 
	// e.g. to compare to zero, >= 0
	__m128 mask = _mm_cmpge_ps( x4, _mm_setzero_ps() ); // if (x4 >= 0)
	
	// Similar comparison instructions exist: 
	// - greater (_mm_cmpgt_ps), 
	// - less (_mm_cmplt_ps), 
	// - less or equal (_mm_cmple_ps), 
	// - equal (_mm_cmpeq_ps) 
	// - and not equal (_mm_cmpne_ps).
	// The mask value is a 128-bit value. 
	//After the comparison, its contents reflect the result: 32 zeroes for a ‘false’, and 32 ones for a ‘true’.
	
	// you can then combine comparisons, e.g. like: 
	__m128 mask1 = _mm_cmpge_ps( x4, _mm_setzero_ps() ); // if (x4 >= 0) 
	__m128 mask2 = _mm_cmpge_ps( y4, _mm_setzero_ps() ); // if (y4 >= 0) 
	__m128 mask = _mm_and_ps( mask1, mask2 ); // if (x4 >= 0 && y4 >= 0)
	// None of this is actually conditional: we unconditionally calculate bitmasks.

	// This instruction takes a mask, and returns a 4-bit value, where each bit is set to 1 if the 32 bits for a lane are 1, and 0 otherwise. Now we can test the bits individually:
	int  result = _mm_movemask_ps( mask ); 
	if (result & 1) { … } // result for first lane is true 
	if (result & 2) { … } // result for second lane is true  
	if (result & 4) { … } // result for third lane is true 
	if (result & 8) { … } // result for fourth lane is true
	But ...... didn’t solve the actual problem though: the conditional code still breaks our vector flow. 

	// INSTEAD: 
	// need to use the masks differently: to disable functionality for lanes.
	// pragmatic solution: if a pixel happens to be off-screen, we write it to location (0,0).
	__m128 mask1 = _mm_cmpge_ps( x4, _mm_setzero_ps() );  // if (x4 >= 0) 
	__m128 mask2 = _mm_cmpge_ps( y4, _mm_setzero_ps() ); 
	__m128 mask3 = _mm_cmplt_ps( x4, _mm_set_ps1( m_Width ) );   // _mm_set_ps1??? 
	__m128 mask4 = _mm_cmplt_ps( y4, _mm_set_ps1( m_Height ) );  
	__m128 mask = _mm_and_ps( _mm_and_ps( _mm_and_ps( mask1, mask2 ), mask3 ), mask4 );

	__m128i address4 = _mm_add_epi32( _mm_mullo_epi32( y4, m_Pitch4 ), x4 ); 
	address4 = _mm_and_si128( address, *(__m128i*)&mask ) );
	// NOTES!!! 
	// - Multiplying two 32-bit integers yields a 64-bit integer, which doesn’t fit in a 32-bit lane. 
	// - The _mm_mullo_epi32 instruction discards the top 32-bit, which is fine in this case. 
	// - There is no _mm_and_epi32 instruction; instead doing a bitwise and to integers operates directly on the 128 bits using _mm_and_si128. 
	// - Our mask is a quadfloat, while _mm_and_si128 expects a quadint mask. We thus convert it on-the-fly to the correct type. 
	// - The second line uses the calculated mask to reset all off-screen pixel addresses to 0, as we planned to do

	
	// Another example, condition to map something else: 
	float a = a == 0 ? b : c
	__m128 mask = _mm_cmpeq_ps( a4, _mm_setzero_ps() ); 
	__m128 part1 = _mm_and_ps( mask, b4 ); 
	__m128 part2 = _mm_andnot_ps( mask, c4 ); 
	a4 = _mm_or_ps( part1, part2 );
	// A more direct way to obtain this result is to use the _mm_blendv_ps instruction:
	__m128 mask = _mm_cmpeq_ps( a4, _mm_setzero_ps() ); 
	a4 = _mm_blendv_ps( b4, c4, mask );



	// HINTS (see tutorial document) 
	// - Instruction count.  In principle every intrinsic compiles to a single compiler instruction. 
	// - Floating point versus integer.   Floating point support in SSE and AVX is much better than integer support.
	// - Reduce the use of _mm_set_ps   You will frequently need constants in your vectorized code, as we have seen in the Mandelbrot example.
			It may be tempting to create quadfloats on the spot for these. However, _mm_set_ps is an expensive function, as it takes four operands. 
			Consider caching the result: calculate the quadfloat outside loops, so you can use it many times without penalty inside the loop. 
			Similarly, if you need to expand scalars to quadfloats (like m_Pitch in the Plot method), consider caching the expanded version in the class. 
	// - Avoid gather operations   An additional pitfall related to _mm_set_ps is that the data you feed it comes from locations scattered though memory. 
			The fastest way to get data from memory to a quadfloat is when it is already stored in memory as a quadfloat, i.e. in 16 consecutive bytes.  
	// - Data alignment   One thing to keep in mind is that a quadfloat in memory must always be stored at an address that is a multiple of 16. Failure to do so will result in a crash. 
	This is why C# used a slow unaligned read for SSE/AVX data: C# cannot guarantee data alignment. In C++, variables created on the stack will automatically obey this rule. 
	Variables allocated using new however may be unaligned, causing unexpected crashes. 
	If you do experience a crash, check if the data that is being processed is properly aligned: the (hexadecimal) address should always end with a zero. 
	- Vectorize bottlenecks only 
	*/
	

}


#if 0


// example code from a tutorial from Jacco 
'original'
float scale = 1 + cosf(t); 
t += 0.01f; 
for (int y = 0; y < SCRHEIGHT; y++)
{
	float yoffs = ((float)y / SCRHEIGHT - 0.5f) * scale;
	float xoffs = -0.5f * scale, dx = scale / SCRWIDTH;
	for (int x = 0; x < SCRWIDTH; x++, xoffs += dx)
	{
		float ox = 0, oy = 0, py;
		for (int i = 0; i < 99; i++)
		{
			px = ox; 
			py = oy;
			oy = -(py * py - px * px - 0.55f + xoffs)
			ox = -(px * py + py * px - 0.55f + yoffs);
		}
		int r = min(255, max(0, (int)(ox * 255)));
		int g = min(255, max(0, (int)(oy * 255)));
		screen->Plot(x, y, (r << 16) + (g << 8));
	}
}


void Surface::Plot4(__m128 x4, __m128 y4, __m128i c4) { 
	__m128 mask1 = _mm_cmpge_ps(x4, _mm_setzero_ps());     
	__m128 mask2 = _mm_cmpge_ps(y4, _mm_setzero_ps());     
	__m128 mask3 = _mm_cmplt_ps(x4, _mm_set_ps1((float)m_Width));     
	__m128 mask4 = _mm_cmplt_ps(y4, _mm_set_ps1((float)m_Height));    
	__m128 mask = _mm_and_ps(_mm_and_ps(_mm_and_ps(mask1, mask2), mask3), mask4);     
	union { __m128i address4; int address[4]; };     
	__m128i m_Pitch4 = _mm_set1_epi32(m_Pitch);     
	__m128i x4i = _mm_cvtps_epi32(x4);     
	__m128i y4i = _mm_cvtps_epi32(y4);     
	address4 = _mm_add_epi32(_mm_mullo_epi32(y4i, m_Pitch4), x4i);     
	for (int i = 0; i < 4; i++) m_Buffer[address[i]] = c4.m128i_i32[i]; 
}

#endif

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
