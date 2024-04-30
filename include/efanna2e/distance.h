//
// Created by 付聪 on 2017/6/21.
// Modified  by 陈萌 on 2024/4/30.
//

#ifndef EFANNA2E_DISTANCE_H
#define EFANNA2E_DISTANCE_H

#include <immintrin.h>
#include <x86intrin.h>

#include <iostream>

namespace efanna2e {
enum Metric { L2 = 0, INNER_PRODUCT = 1, FAST_L2 = 2, PQ = 3, COSINE = 4 };
class Distance {
   public:
    virtual float compare(const float *a, const float *b, unsigned length) const = 0;
    virtual ~Distance() {}
};

class DistanceL2 : public Distance {
    static inline __m128 masked_read(int d, const float *x) {
        // assert(0 <= d && d < 4);
        __attribute__((__aligned__(16))) float buf[4] = {0, 0, 0, 0};
        switch (d) {
            case 3:
                buf[2] = x[2];
            case 2:
                buf[1] = x[1];
            case 1:
                buf[0] = x[0];
        }
        return _mm_load_ps(buf);
        // cannot use AVX2 _mm_mask_set1_epi32
    }

   public:
    float compare(const float *x, const float *y, unsigned d) const {
        __m512 msum0 = _mm512_setzero_ps();

        while (d >= 16) {
            __m512 mx = _mm512_loadu_ps(x);
            x += 16;
            __m512 my = _mm512_loadu_ps(y);
            y += 16;
            const __m512 a_m_b1 = mx - my;
            msum0 += a_m_b1 * a_m_b1;
            d -= 16;
        }

        __m256 msum1 = _mm512_extractf32x8_ps(msum0, 1);
        msum1 += _mm512_extractf32x8_ps(msum0, 0);

        if (d >= 8) {
            __m256 mx = _mm256_loadu_ps(x);
            x += 8;
            __m256 my = _mm256_loadu_ps(y);
            y += 8;
            const __m256 a_m_b1 = mx - my;
            msum1 += a_m_b1 * a_m_b1;
            d -= 8;
        }

        __m128 msum2 = _mm256_extractf128_ps(msum1, 1);
        msum2 += _mm256_extractf128_ps(msum1, 0);

        if (d >= 4) {
            __m128 mx = _mm_loadu_ps(x);
            x += 4;
            __m128 my = _mm_loadu_ps(y);
            y += 4;
            const __m128 a_m_b1 = mx - my;
            msum2 += a_m_b1 * a_m_b1;
            d -= 4;
        }

        if (d > 0) {
            __m128 mx = masked_read(d, x);
            __m128 my = masked_read(d, y);
            __m128 a_m_b1 = mx - my;
            msum2 += a_m_b1 * a_m_b1;
        }

        msum2 = _mm_hadd_ps(msum2, msum2);
        msum2 = _mm_hadd_ps(msum2, msum2);
        return _mm_cvtss_f32(msum2);
        // return result;
    }
};

class DistanceInnerProduct : public Distance {
   public:
    static inline __m128 masked_read(int d, const float *x) {
        // assert(0 <= d && d < 4);
        __attribute__((__aligned__(16))) float buf[4] = {0, 0, 0, 0};
        switch (d) {
            case 3:
                buf[2] = x[2];
            case 2:
                buf[1] = x[1];
            case 1:
                buf[0] = x[0];
        }
        return _mm_load_ps(buf);
        // cannot use AVX2 _mm_mask_set1_epi32
    }
    float compare(const float *a, const float *b, unsigned size) const {

#ifdef __GNUC__
#ifdef __AVX__
#define AVX_DOT(addr1, addr2, dest, tmp1, tmp2) \
    tmp1 = _mm256_load_ps(addr1);               \
    tmp2 = _mm256_load_ps(addr2);               \
    tmp1 = _mm256_mul_ps(tmp1, tmp2);           \
    dest = _mm256_add_ps(dest, tmp1);

#else
#ifdef __SSE2__
#define SSE_DOT(addr1, addr2, dest, tmp1, tmp2) \
    tmp1 = _mm128_loadu_ps(addr1);              \
    tmp2 = _mm128_loadu_ps(addr2);              \
    tmp1 = _mm128_mul_ps(tmp1, tmp2);           \
    dest = _mm128_add_ps(dest, tmp1);
        __m128 sum;
        __m128 l0, l1, l2, l3;
        __m128 r0, r1, r2, r3;
        unsigned D = (size + 3) & ~3U;
        unsigned DR = D % 16;
        unsigned DD = D - DR;
        const float *l = a;
        const float *r = b;
        const float *e_l = l + DD;
        const float *e_r = r + DD;
        float unpack[4] __attribute__((aligned(16))) = {0, 0, 0, 0};

        sum = _mm_load_ps(unpack);
        switch (DR) {
            case 12:
                SSE_DOT(e_l + 8, e_r + 8, sum, l2, r2);
            case 8:
                SSE_DOT(e_l + 4, e_r + 4, sum, l1, r1);
            case 4:
                SSE_DOT(e_l, e_r, sum, l0, r0);
            default:
                break;
        }
        for (unsigned i = 0; i < DD; i += 16, l += 16, r += 16) {
            SSE_DOT(l, r, sum, l0, r0);
            SSE_DOT(l + 4, r + 4, sum, l1, r1);
            SSE_DOT(l + 8, r + 8, sum, l2, r2);
            SSE_DOT(l + 12, r + 12, sum, l3, r3);
        }
        _mm_storeu_ps(unpack, sum);
        result += unpack[0] + unpack[1] + unpack[2] + unpack[3];
#else

        float dot0, dot1, dot2, dot3;
        const float *last = a + size;
        const float *unroll_group = last - 3;

        /* Process 4 items with each loop for efficiency. */
        while (a < unroll_group) {
            dot0 = a[0] * b[0];
            dot1 = a[1] * b[1];
            dot2 = a[2] * b[2];
            dot3 = a[3] * b[3];
            result += dot0 + dot1 + dot2 + dot3;
            a += 4;
            b += 4;
        }
        /* Process last 0-3 pixels.  Not needed for standard vector lengths. */
        while (a < last) {
            result += *a++ * *b++;
        }
#endif
#endif
#endif
        // using avx-512
        __m512 msum0 = _mm512_setzero_ps();

        while (size >= 16) {
            __m512 mx = _mm512_loadu_ps(a);
            a += 16;
            __m512 my = _mm512_loadu_ps(b);
            b += 16;
            msum0 = _mm512_add_ps(msum0, _mm512_mul_ps(mx, my));
            size -= 16;
        }

        __m256 msum1 = _mm512_extractf32x8_ps(msum0, 1);
        msum1 += _mm512_extractf32x8_ps(msum0, 0);

        if (size >= 8) {
            __m256 mx = _mm256_loadu_ps(a);
            a += 8;
            __m256 my = _mm256_loadu_ps(b);
            b += 8;
            msum1 = _mm256_add_ps(msum1, _mm256_mul_ps(mx, my));
            size -= 8;
        }

        __m128 msum2 = _mm256_extractf128_ps(msum1, 1);
        msum2 += _mm256_extractf128_ps(msum1, 0);

        if (size >= 4) {
            __m128 mx = _mm_loadu_ps(a);
            a += 4;
            __m128 my = _mm_loadu_ps(b);
            b += 4;
            msum2 = _mm_add_ps(msum2, _mm_mul_ps(mx, my));
            size -= 4;
        }

        if (size > 0) {
            __m128 mx = masked_read(size, a);
            __m128 my = masked_read(size, b);
            msum2 = _mm_add_ps(msum2, _mm_mul_ps(mx, my));
        }

        msum2 = _mm_hadd_ps(msum2, msum2);
        msum2 = _mm_hadd_ps(msum2, msum2);
        return -1.0 * _mm_cvtss_f32(msum2);
        // return result;
    }
};
class DistanceFastL2 : public DistanceInnerProduct {
   public:
    float norm(const float *a, unsigned size) const {
        float result = 0;
#ifdef __GNUC__
#ifdef __AVX__
#define AVX_L2NORM(addr, dest, tmp) \
    tmp = _mm256_loadu_ps(addr);    \
    tmp = _mm256_mul_ps(tmp, tmp);  \
    dest = _mm256_add_ps(dest, tmp);

        __m256 sum;
        __m256 l0, l1;
        unsigned D = (size + 7) & ~7U;
        unsigned DR = D % 16;
        unsigned DD = D - DR;
        const float *l = a;
        const float *e_l = l + DD;
        float unpack[8] __attribute__((aligned(32))) = {0, 0, 0, 0, 0, 0, 0, 0};

        sum = _mm256_loadu_ps(unpack);
        if (DR) {
            AVX_L2NORM(e_l, sum, l0);
        }
        for (unsigned i = 0; i < DD; i += 16, l += 16) {
            AVX_L2NORM(l, sum, l0);
            AVX_L2NORM(l + 8, sum, l1);
        }
        _mm256_storeu_ps(unpack, sum);
        result = unpack[0] + unpack[1] + unpack[2] + unpack[3] + unpack[4] + unpack[5] + unpack[6] + unpack[7];
#else
#ifdef __SSE2__
#define SSE_L2NORM(addr, dest, tmp) \
    tmp = _mm128_loadu_ps(addr);    \
    tmp = _mm128_mul_ps(tmp, tmp);  \
    dest = _mm128_add_ps(dest, tmp);

        __m128 sum;
        __m128 l0, l1, l2, l3;
        unsigned D = (size + 3) & ~3U;
        unsigned DR = D % 16;
        unsigned DD = D - DR;
        const float *l = a;
        const float *e_l = l + DD;
        float unpack[4] __attribute__((aligned(16))) = {0, 0, 0, 0};

        sum = _mm_load_ps(unpack);
        switch (DR) {
            case 12:
                SSE_L2NORM(e_l + 8, sum, l2);
            case 8:
                SSE_L2NORM(e_l + 4, sum, l1);
            case 4:
                SSE_L2NORM(e_l, sum, l0);
            default:
                break;
        }
        for (unsigned i = 0; i < DD; i += 16, l += 16) {
            SSE_L2NORM(l, sum, l0);
            SSE_L2NORM(l + 4, sum, l1);
            SSE_L2NORM(l + 8, sum, l2);
            SSE_L2NORM(l + 12, sum, l3);
        }
        _mm_storeu_ps(unpack, sum);
        result += unpack[0] + unpack[1] + unpack[2] + unpack[3];
#else
        float dot0, dot1, dot2, dot3;
        const float *last = a + size;
        const float *unroll_group = last - 3;

        /* Process 4 items with each loop for efficiency. */
        while (a < unroll_group) {
            dot0 = a[0] * a[0];
            dot1 = a[1] * a[1];
            dot2 = a[2] * a[2];
            dot3 = a[3] * a[3];
            result += dot0 + dot1 + dot2 + dot3;
            a += 4;
        }
        /* Process last 0-3 pixels.  Not needed for standard vector lengths. */
        while (a < last) {
            result += (*a) * (*a);
            a++;
        }
#endif
#endif
#endif
        return result;
    }
    using DistanceInnerProduct::compare;
    float compare(const float *a, const float *b, float norm, unsigned size) const {  // not implement
        float result = -2 * DistanceInnerProduct::compare(a, b, size);
        result += norm;
        return result;
    }
};
}  // namespace efanna2e

#endif  // EFANNA2E_DISTANCE_H
