/*
Copyright (C) 2014-2025, Youssef Touil <youssef@airspy.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to use,
copy, modify, merge, and distribute the Software exclusively as part of the 
Airspy ecosystem, which includes Airspy-branded hardware, official tools, and
associated software directly approved or maintained by the original authors.

Any redistribution, publication, sublicensing, or commercial use outside the
Airspy ecosystem is strictly prohibited without prior written consent from the 
copyright holders.

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software used within the Airspy ecosystem.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE, PROVIDED SUCH USE REMAINS WITHIN THE AIRSPY ECOSYSTEM.
*/

#include "iqconverter_float.h"
#include <stdlib.h>
#include <string.h>

#include <stdio.h>

#if defined(__MINGW32__) && !defined(__MINGW64_VERSION_MAJOR)
  #include <malloc.h>
  #define _aligned_malloc __mingw_aligned_malloc
  #define _aligned_free  __mingw_aligned_free
  #define _inline inline
  #define FIR_STANDARD
#elif defined(__APPLE__)
  #include <malloc/malloc.h>
  #define _aligned_malloc(size, alignment) malloc(size)
  #define _aligned_free(mem) free(mem)
  #define _inline inline
  #define FIR_STANDARD
#elif defined(__FreeBSD__)
  #if defined(__x86_64__) || defined(__i386__)
    #define USE_SSE2
    #include <immintrin.h>
    #if defined(__AVX2__)
      #define USE_AVX2
    #endif
    #if defined(__FMA__)
      #define USE_FMA3
    #endif
  #elif defined(__ARM_NEON)
    #define USE_NEON
    #include <arm_neon.h>
  #endif
  #define _inline inline
  #define _aligned_free(mem) free(mem)
void *_aligned_malloc(size_t size, size_t alignment)
{
    void *result;
    if (posix_memalign(&result, alignment, size) == 0)
        return result;
    return 0;
}
#elif defined(__GNUC__) && !defined(__MINGW64_VERSION_MAJOR)
  #include <malloc.h>
  #define _aligned_malloc(size, alignment) memalign(alignment, size)
  #define _aligned_free(mem) free(mem)
  #define _inline inline
  #if defined(__x86_64__) || defined(__i386__)
    #define USE_SSE2
    #include <immintrin.h>
    #if defined(__AVX2__)
      #define USE_AVX2
    #endif
    #if defined(__FMA__)
      #define USE_FMA3
    #endif
  #elif defined(__ARM_NEON)
    #define USE_NEON
    #include <arm_neon.h>
  #endif
#else
	#if (_MSC_VER >= 1800)
		//#define USE_SSE2
		//#include <immintrin.h>
	#endif
#endif

#define SIZE_FACTOR 32
#define DEFAULT_ALIGNMENT 32
#define HPF_COEFF 0.01f

#if defined(_MSC_VER)
	#define ALIGNED __declspec(align(DEFAULT_ALIGNMENT))
#else
	#define ALIGNED
#endif

iqconverter_float_t *iqconverter_float_create(const float *hb_kernel, int len)
{
	int i, j;
	size_t buffer_size;
	iqconverter_float_t *cnv = (iqconverter_float_t *) _aligned_malloc(sizeof(iqconverter_float_t), DEFAULT_ALIGNMENT);

	cnv->len = len / 2 + 1;
	cnv->hbc = hb_kernel[len / 2];

	buffer_size = cnv->len * sizeof(float);

	cnv->fir_kernel = (float *) _aligned_malloc(buffer_size, DEFAULT_ALIGNMENT);
	cnv->fir_queue = (float *) _aligned_malloc(buffer_size * SIZE_FACTOR, DEFAULT_ALIGNMENT);
	cnv->delay_line = (float *) _aligned_malloc(buffer_size / 2, DEFAULT_ALIGNMENT);

	iqconverter_float_reset(cnv);

	for (i = 0, j = 0; i < cnv->len; i++, j += 2)
	{
		cnv->fir_kernel[i] = hb_kernel[j];
	} 

	return cnv;
}

void iqconverter_float_free(iqconverter_float_t *cnv)
{
	_aligned_free(cnv->fir_kernel);
	_aligned_free(cnv->fir_queue);
	_aligned_free(cnv->delay_line);
	_aligned_free(cnv);
}

void iqconverter_float_reset(iqconverter_float_t *cnv)
{
	cnv->avg = 0.0f;
	cnv->fir_index = 0;
	cnv->delay_index = 0;
	memset(cnv->delay_line, 0, cnv->len * sizeof(float) / 2);
	memset(cnv->fir_queue, 0, cnv->len * sizeof(float) * SIZE_FACTOR);
}

static _inline float process_fir_taps(const float *kernel, const float *queue, int len)
{
	int i;

#ifdef USE_AVX2

	/* --- AVX2 path: 16 floats/iter with optional FMA3 + prefetch --- */
	__m256 acc256 = _mm256_setzero_ps();
	_mm_prefetch((const char *)kernel, _MM_HINT_T0);
	_mm_prefetch((const char *)queue,  _MM_HINT_T0);

	if (len >= 16)
	{
		int it = len >> 4;
		for (i = 0; i < it; i++)
		{
			_mm_prefetch((const char *)(kernel + 16), _MM_HINT_T0);
			_mm_prefetch((const char *)(queue  + 16), _MM_HINT_T0);
#ifdef USE_FMA3
			acc256 = _mm256_fmadd_ps(_mm256_loadu_ps(kernel),     _mm256_loadu_ps(queue),     acc256);
			acc256 = _mm256_fmadd_ps(_mm256_loadu_ps(kernel + 8), _mm256_loadu_ps(queue + 8), acc256);
#else
			acc256 = _mm256_add_ps(acc256,
				_mm256_add_ps(
					_mm256_mul_ps(_mm256_loadu_ps(kernel),     _mm256_loadu_ps(queue)),
					_mm256_mul_ps(_mm256_loadu_ps(kernel + 8), _mm256_loadu_ps(queue + 8))));
#endif
			kernel += 16;
			queue  += 16;
		}
		len &= 15;
	}

	/* Horizontal reduce __m256 → __m128 */
	__m128 acc = _mm_add_ps(
		_mm256_extractf128_ps(acc256, 0),
		_mm256_extractf128_ps(acc256, 1));
	_mm256_zeroupper();

	/* SSE2 cleanup for remaining <16 samples */
	if (len >= 8)
	{
		__m128 h1 = _mm_loadu_ps(queue);
		__m128 k1 = _mm_load_ps(kernel);
		__m128 h2 = _mm_loadu_ps(queue + 4);
		__m128 k2 = _mm_load_ps(kernel + 4);
		acc = _mm_add_ps(acc, _mm_add_ps(_mm_mul_ps(k1, h1), _mm_mul_ps(k2, h2)));
		queue += 8; kernel += 8; len -= 8;
	}
	if (len >= 4)
	{
		__m128 head = _mm_loadu_ps(queue);
		__m128 kern = _mm_load_ps(kernel);
		acc = _mm_add_ps(acc, _mm_mul_ps(kern, head));
		kernel += 4; queue += 4; len &= 3;
	}

	__m128 t = _mm_add_ps(acc, _mm_movehl_ps(acc, acc));
	acc = _mm_add_ss(t, _mm_shuffle_ps(t, t, 1));
	float sum = _mm_cvtss_f32(acc);

#elif defined(USE_NEON)

	/* --- ARM NEON path: 8 floats/iter with vmlaq_f32 --- */
	float32x4_t acc_n = vdupq_n_f32(0.0f);

	if (len >= 8)
	{
		int it = len >> 3;
		for (i = 0; i < it; i++)
		{
			acc_n = vmlaq_f32(acc_n, vld1q_f32(kernel),     vld1q_f32(queue));
			acc_n = vmlaq_f32(acc_n, vld1q_f32(kernel + 4), vld1q_f32(queue + 4));
			kernel += 8;
			queue  += 8;
		}
		len &= 7;
	}
	if (len >= 4)
	{
		acc_n = vmlaq_f32(acc_n, vld1q_f32(kernel), vld1q_f32(queue));
		kernel += 4; queue += 4; len &= 3;
	}

	float32x2_t s2 = vadd_f32(vget_low_f32(acc_n), vget_high_f32(acc_n));
	s2 = vpadd_f32(s2, s2);
	float sum = vget_lane_f32(s2, 0);

#elif defined(USE_SSE2)

	/* --- SSE2 path: 8 floats/iter --- */
	__m128 acc = _mm_set_ps(0, 0, 0, 0);

	if (len >= 8)
	{
		int it = len >> 3;
		for (i = 0; i < it; i++)
		{
			__m128 head1 = _mm_loadu_ps(queue);
			__m128 kern1 = _mm_load_ps(kernel);
			__m128 head2 = _mm_loadu_ps(queue + 4);
			__m128 kern2 = _mm_load_ps(kernel + 4);
			__m128 mul1  = _mm_mul_ps(kern1, head1);
			__m128 mul2  = _mm_mul_ps(kern2, head2);
			acc = _mm_add_ps(acc, _mm_add_ps(mul1, mul2));
			queue  += 8;
			kernel += 8;
		}
		len &= 7;
	}
	if (len >= 4)
	{
		__m128 head = _mm_loadu_ps(queue);
		__m128 kern = _mm_load_ps(kernel);
		acc = _mm_add_ps(acc, _mm_mul_ps(kern, head));
		kernel += 4; queue += 4; len &= 3;
	}

	__m128 t = _mm_add_ps(acc, _mm_movehl_ps(acc, acc));
	acc = _mm_add_ss(t, _mm_shuffle_ps(t, t, 1));
	float sum = _mm_cvtss_f32(acc);

#else

	/* --- Scalar fallback --- */
	float sum = 0.0f;

	if (len >= 8)
	{
		int it = len >> 3;
		for (i = 0; i < it; i++)
		{
			sum += kernel[0] * queue[0]
				+ kernel[1] * queue[1]
				+ kernel[2] * queue[2]
				+ kernel[3] * queue[3]
				+ kernel[4] * queue[4]
				+ kernel[5] * queue[5]
				+ kernel[6] * queue[6]
				+ kernel[7] * queue[7];
			queue  += 8;
			kernel += 8;
		}
		len &= 7;
	}
	if (len >= 4)
	{
		sum += kernel[0] * queue[0]
			+ kernel[1] * queue[1]
			+ kernel[2] * queue[2]
			+ kernel[3] * queue[3];
		kernel += 4; queue += 4; len &= 3;
	}

#endif

	if (len >= 2)
	{
		sum += kernel[0] * queue[0]
			+ kernel[1] * queue[1];
	}

	return sum;
}

static void fir_interleaved_4(iqconverter_float_t *cnv, float *samples, int len)
{
	int i;
	int fir_index = cnv->fir_index;
	int fir_len = cnv->len;
	float *fir_kernel = cnv->fir_kernel;
	float *fir_queue = cnv->fir_queue;
	float *queue;
	float acc;

	for (i = 0; i < len; i += 2)
	{
		queue = fir_queue + fir_index;

		queue[0] = samples[i];

		acc = fir_kernel[0] * (queue[0] + queue[4 - 1])
			+ fir_kernel[1] * (queue[1] + queue[4 - 2]);

		samples[i] = acc;

		if (--fir_index < 0)
		{
			fir_index = fir_len * (SIZE_FACTOR - 1);
			memcpy(fir_queue + fir_index + 1, fir_queue, (fir_len - 1) * sizeof(float));
		}
	}

	cnv->fir_index = fir_index;
}

static void fir_interleaved_8(iqconverter_float_t *cnv, float *samples, int len)
{
	int i;
	int fir_index = cnv->fir_index;
	int fir_len = cnv->len;
	float *fir_kernel = cnv->fir_kernel;
	float *fir_queue = cnv->fir_queue;
	float *queue;
	float acc;

	for (i = 0; i < len; i += 2)
	{
		queue = fir_queue + fir_index;

		queue[0] = samples[i];

		acc = fir_kernel[0] * (queue[0] + queue[8 - 1])
			+ fir_kernel[1] * (queue[1] + queue[8 - 2])
			+ fir_kernel[2] * (queue[2] + queue[8 - 3])
			+ fir_kernel[3] * (queue[3] + queue[8 - 4]);

		samples[i] = acc;

		if (--fir_index < 0)
		{
			fir_index = fir_len * (SIZE_FACTOR - 1);
			memcpy(fir_queue + fir_index + 1, fir_queue, (fir_len - 1) * sizeof(float));
		}
	}

	cnv->fir_index = fir_index;
}

static void fir_interleaved_12(iqconverter_float_t *cnv, float *samples, int len)
{
	int i;
	int fir_index = cnv->fir_index;
	int fir_len = cnv->len;
	float *fir_kernel = cnv->fir_kernel;
	float *fir_queue = cnv->fir_queue;
	float *queue;
	float acc = 0;

	for (i = 0; i < len; i += 2)
	{
		queue = fir_queue + fir_index;

		queue[0] = samples[i];

		acc = fir_kernel[0]  * (queue[0]  + queue[12 - 1])
			+ fir_kernel[1]  * (queue[1]  + queue[12 - 2])
			+ fir_kernel[2]  * (queue[2]  + queue[12 - 3])
			+ fir_kernel[3]  * (queue[3]  + queue[12 - 4])
			+ fir_kernel[4]  * (queue[4]  + queue[12 - 5])
			+ fir_kernel[5]  * (queue[5]  + queue[12 - 6]);

		samples[i] = acc;

		if (--fir_index < 0)
		{
			fir_index = fir_len * (SIZE_FACTOR - 1);
			memcpy(fir_queue + fir_index + 1, fir_queue, (fir_len - 1) * sizeof(float));
		}
	}

	cnv->fir_index = fir_index;
}

static void fir_interleaved_24(iqconverter_float_t *cnv, float *samples, int len)
{
	int i;
	int fir_index = cnv->fir_index;
	int fir_len = cnv->len;
	float *fir_kernel = cnv->fir_kernel;
	float *fir_queue = cnv->fir_queue;
	float *queue;
	float acc = 0;

	for (i = 0; i < len; i += 2)
	{
		queue = fir_queue + fir_index;

		queue[0] = samples[i];

		acc = fir_kernel[0]  * (queue[0]  + queue[24 - 1])
			+ fir_kernel[1]  * (queue[1]  + queue[24 - 2])
			+ fir_kernel[2]  * (queue[2]  + queue[24 - 3])
			+ fir_kernel[3]  * (queue[3]  + queue[24 - 4])
			+ fir_kernel[4]  * (queue[4]  + queue[24 - 5])
			+ fir_kernel[5]  * (queue[5]  + queue[24 - 6])
			+ fir_kernel[6]  * (queue[6]  + queue[24 - 7])
			+ fir_kernel[7]  * (queue[7]  + queue[24 - 8])
			+ fir_kernel[8]  * (queue[8]  + queue[24 - 9])
			+ fir_kernel[9]  * (queue[9]  + queue[24 - 10])
			+ fir_kernel[10] * (queue[10] + queue[24 - 11])
			+ fir_kernel[11] * (queue[11] + queue[24 - 12]);

		samples[i] = acc;

		if (--fir_index < 0)
		{
			fir_index = fir_len * (SIZE_FACTOR - 1);
			memcpy(fir_queue + fir_index + 1, fir_queue, (fir_len - 1) * sizeof(float));
		}
	}

	cnv->fir_index = fir_index;
}

static void fir_interleaved_generic(iqconverter_float_t *cnv, float *samples, int len)
{
	int i;
	int fir_index = cnv->fir_index;
	int fir_len = cnv->len;
	float *fir_kernel = cnv->fir_kernel;
	float *fir_queue = cnv->fir_queue;
	float *queue;

	for (i = 0; i < len; i += 2)
	{
		queue = fir_queue + fir_index;

		queue[0] = samples[i];

		samples[i] = process_fir_taps(fir_kernel, queue, fir_len);

		if (--fir_index < 0)
		{
			fir_index = fir_len * (SIZE_FACTOR - 1);
			memcpy(fir_queue + fir_index + 1, fir_queue, (fir_len - 1) * sizeof(float));
		}
	}

	cnv->fir_index = fir_index;
}

static void fir_interleaved_32(iqconverter_float_t *cnv, float *samples, int len)
{
	int i;
	int fir_index = cnv->fir_index;
	int fir_len = cnv->len;
	float *fir_kernel = cnv->fir_kernel;
	float *fir_queue = cnv->fir_queue;
	float *queue;

	for (i = 0; i < len; i += 2)
	{
		queue = fir_queue + fir_index;

		queue[0] = samples[i];

		samples[i] = process_fir_taps(fir_kernel, queue, 32);

		if (--fir_index < 0)
		{
			fir_index = fir_len * (SIZE_FACTOR - 1);
			memcpy(fir_queue + fir_index + 1, fir_queue, (fir_len - 1) * sizeof(float));
		}
	}

	cnv->fir_index = fir_index;
}

static void fir_interleaved(iqconverter_float_t *cnv, float *samples, int len)
{
	switch (cnv->len)
	{
	case 4:
		fir_interleaved_4(cnv, samples, len);
		break;
	case 8:
		fir_interleaved_8(cnv, samples, len);
		break;
	case 12:
		fir_interleaved_12(cnv, samples, len);
		break;
	case 24:
		fir_interleaved_24(cnv, samples, len);
		break;
	case 32:
		fir_interleaved_32(cnv, samples, len);
		break;
	default:
		fir_interleaved_generic(cnv, samples, len);
		break;
	}
}

static void delay_interleaved(iqconverter_float_t *cnv, float *samples, int len)
{
	int i;
	ALIGNED int index;
	ALIGNED int half_len;
	ALIGNED float res;
	
	half_len = cnv->len >> 1;
	index = cnv->delay_index;

	for (i = 0; i < len; i += 2)
	{
		res = cnv->delay_line[index];
		cnv->delay_line[index] = samples[i];
		samples[i] = res;

		if (++index >= half_len)
		{
			index = 0;
		}
	}
	
	cnv->delay_index = index;
}

#define SCALE (0.005f)

static void remove_dc(iqconverter_float_t *cnv, float *samples, int len)
{
	int i;
	ALIGNED float avg = cnv->avg;

	for (i = 0; i < len; i++)
	{
		samples[i] -= avg;
		avg += SCALE * samples[i];
	}

	cnv->avg = avg;
}

static void translate_fs_4(iqconverter_float_t *cnv, float *samples, int len)
{
	int i;
	ALIGNED float hbc = cnv->hbc;

#ifdef USE_AVX2

	/* --- AVX2 path: 8 samples/iter (256-bit) --- */
	float *buf = samples;
	__m256 rot8 = _mm256_set_ps(hbc, 1.0f, -hbc, -1.0f, hbc, 1.0f, -hbc, -1.0f);

	for (i = 0; i < len / 8; i++, buf += 8)
	{
		__m256 vec = _mm256_loadu_ps(buf);
		_mm256_storeu_ps(buf, _mm256_mul_ps(vec, rot8));
	}

	/* Handle remaining 4 samples if len is not a multiple of 8 */
	if ((len % 8) >= 4)
	{
		__m128 rot4 = _mm_set_ps(hbc, 1.0f, -hbc, -1.0f);
		_mm_storeu_ps(buf, _mm_mul_ps(_mm_loadu_ps(buf), rot4));
	}
	_mm256_zeroupper();

#elif defined(USE_NEON)

	/* --- ARM NEON path: 4 samples/iter --- */
	float *buf = samples;
	const float32x4_t rot4 = { -1.0f, -hbc, 1.0f, hbc };

	for (i = 0; i < len / 4; i++, buf += 4)
	{
		vst1q_f32(buf, vmulq_f32(vld1q_f32(buf), rot4));
	}

#elif defined(USE_SSE2)

	/* --- SSE2 path: 4 samples/iter (128-bit) --- */
	float *buf = samples;
	ALIGNED __m128 vec;
	ALIGNED __m128 rot = _mm_set_ps(hbc, 1.0f, -hbc, -1.0f);

	for (i = 0; i < len / 4; i++, buf += 4)
	{
		vec = _mm_loadu_ps(buf);
		vec = _mm_mul_ps(vec, rot);
		_mm_storeu_ps(buf, vec);
	}

#else

	/* --- Scalar fallback --- */
	int j;

	for (i = 0; i < len / 4; i++)
	{
		j = i << 2;
		samples[j + 0] = -samples[j + 0];
		samples[j + 1] = -samples[j + 1] * hbc;
		//samples[j + 2] = samples[j + 2];
		samples[j + 3] = samples[j + 3] * hbc;
	}

#endif

	fir_interleaved(cnv, samples, len);
	delay_interleaved(cnv, samples + 1, len);
}

void iqconverter_float_process(iqconverter_float_t *cnv, float *samples, int len)
{
	remove_dc(cnv, samples, len);
	translate_fs_4(cnv, samples, len);
}
