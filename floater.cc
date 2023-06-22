#include <algorithm>
#include <atomic>
#include <chrono>
#include <iostream>
#include <thread>

#include <immintrin.h>
#include <pthread.h>

using namespace std::literals::chrono_literals;
using hires_clock = std::chrono::high_resolution_clock;

#define LIKELY(expr) __builtin_expect(expr, 1)
#define UNLIKELY(expr) __builtin_expect(expr, 0)
#define WAIT(var, cnt) do {} while (var < cnt)

namespace {
    // Number of floats
    constexpr size_t kBufSize = 64 * 1024 * 1024;
    // Number of test samples
    constexpr size_t kNumSamples = 100;
    // Threshhold for read-modify-write
    constexpr float kThres = 0.5f;
    // Number of threads to create for multi-threaded versions of tests
    constexpr size_t kDefaultNumThreads = 32;

    void set_max_prio(const std::thread::native_handle_type& handle) {
        sched_param sch_params;
        sch_params.sched_priority = sched_get_priority_max(SCHED_FIFO);
        if (pthread_setschedparam(handle, SCHED_FIFO, &sch_params)) {
            std::cerr << "Failed to set Thread scheduling : " << std::strerror(errno) << std::endl;
        }
    }
} // namespace 


// Statistics aggregator
struct Stat {
    uint64_t dtmin = ~uint64_t{};
    uint64_t dtmax = 0;
    double dtavg = 0.0;
    uint64_t dtsum = 0;
    uint64_t samples = 0;
    double gops_per_sec = 0;
    double throughput = 0;

    void update(uint64_t dt_usec) {
        dtmin = std::min(dtmin, dt_usec);
        dtmax = std::max(dtmax, dt_usec);
        dtsum += dt_usec;
        ++samples;
        dtavg = dtsum / static_cast<double>(samples);
        gops_per_sec = kBufSize / (dtavg * 1000.0);
        throughput = (sizeof(float) * kBufSize) / (dtavg * 1000.0);
    }
};
////////////////////////////////////////////////////////////////////////////////////////////////////


// Measures time between few points in code
class WatchPoint {
    public:
        inline WatchPoint() {
            pre_ = hires_clock::now();
        }
        inline uint64_t time() const {
            auto post = hires_clock::now();
            return std::chrono::duration_cast<std::chrono::microseconds>(post - pre_).count();
        }
    private:
        hires_clock::time_point pre_;
};
////////////////////////////////////////////////////////////////////////////////////////////////////


// T is a function that we need to measure
template<class T> 
inline void measure() {
    std::cout << "Measurement: " << T::name << std::endl;

    Stat initStat;
    Stat stat;
    for (size_t i = 0; i < kNumSamples; ++i) {
        WatchPoint initWatcher;
        T obj;
        initStat.update(initWatcher.time());
        {
            WatchPoint performWatcher;
            obj.perform();
            stat.update(performWatcher.time());
        }

        // This is to avoid tests code to be optimized out due to unused result
        if (!obj.check() && (rand() % 1000000) == 0) { 
            std::cout << "\r" << std::flush;
        } 
        std::cout << "Running: " << (i * 100 / kNumSamples) << "%\r" << std::flush;
    }
    std::cout << "Complete: " << "init=" << initStat.dtavg / 1'000'000 << "s"
        << " avg=" << stat.dtavg
        << " min=" << stat.dtmin
        << " max=" << stat.dtmax
        << " us"
        << " | gflops=" << stat.gops_per_sec
        << " | throughput=" << stat.throughput << " gb/s"
        << std::endl;
    std::cout << "============================================================" << std::endl;
}
////////////////////////////////////////////////////////////////////////////////////////////////////


// Basic multi-threading template, we don't want to mearure overhead on creating threads
// So it pre-creates threads before perform() call and waits for perform to be called.
template<class T, size_t NumThreads = kDefaultNumThreads>
struct ThreadingTraits {
    std::atomic<unsigned> ready_{0};
    std::atomic<unsigned> start_{0};
    std::atomic<unsigned> done_{0};

    std::thread threads_[NumThreads];

    template<class Func>
    inline void setupThreads(Func&& func) {
        static const size_t stride = kBufSize / NumThreads;

        for (size_t i = 0; i < NumThreads; ++i) {
            threads_[i] = std::thread([=] {
                float* __restrict__ cur_src = static_cast<T*>(this)->src_;
                float* __restrict__ end_src = static_cast<T*>(this)->src_ + stride;
                float* __restrict__ cur_dst = static_cast<T*>(this)->dst_;
                ++ready_;
                WAIT(start_, 1);

                func(cur_src, end_src, cur_dst);
                ++done_;
            });
            set_max_prio(threads_[i].native_handle());
        }
        WAIT(ready_, NumThreads);
    }

    ~ThreadingTraits() {
        for (auto& thr : threads_) {
            thr.join();
        }
    }

    void perform() {
        start_ = 1;
        WAIT(done_, NumThreads);
    }
};
////////////////////////////////////////////////////////////////////////////////////////////////////


// base test, creates 2 arrays of floats, concrete implementation will copy data between the two
struct FloaterBase
{
    FloaterBase() {
        src_ = (float*)aligned_alloc(alignof(__m256), sizeof(float) * kBufSize);
        assert(src_ != nullptr);
        dst_ = (float*)aligned_alloc(alignof(__m256), sizeof(float) * kBufSize);
        assert(dst_ != nullptr);
        for (float* x = src_; x != src_ + kBufSize; ++x) {
            *x = rand() / float(RAND_MAX);
        }

        // warmup TLB:
        memset(dst_, 0, sizeof(float) * kBufSize);
    }

    ~FloaterBase() {
        free(dst_);
        free(src_);
    }

    bool check() const {
        float* cur_src = src_;
        float* cur_dst = dst_;
        float *end_src = src_ + kBufSize;
        while (cur_src != end_src) {
            if (*cur_src++ != *cur_dst++) {
                return false;
            }
        }
        return true;
    }

    float* src_ = nullptr;
    float* dst_ = nullptr;
};


// Simple memcpy from src to dst array
struct Memcpy : FloaterBase {
    static constexpr auto name = "Memcpy";

    void perform() {
        memcpy(dst_, src_, sizeof(float) * kBufSize);
    }
};
////////////////////////////////////////////////////////////////////////////////////////////////////


// Memcpy using AVX streaming instructions, 
struct MemcpyStreamAVX : FloaterBase {
    static constexpr auto name = "MemcpyStreamAVX";

    void perform() {
        const __m256i *mm_cur_src = reinterpret_cast<const __m256i*>(src_);
        const __m256i *mm_end_src = reinterpret_cast<const __m256i*>(src_ + kBufSize);
        __m256i *mm_cur_dst = reinterpret_cast<__m256i*>(dst_);
        #pragma unroll(8)
        while (mm_cur_src != mm_end_src) {
            const __m256i val = _mm256_stream_load_si256(mm_cur_src);
            _mm256_stream_si256(mm_cur_dst, val);
            ++mm_cur_src;
            ++mm_cur_dst;
        }
    }
};
////////////////////////////////////////////////////////////////////////////////////////////////////


// Multithreaded memcpy
struct MemcpyMT : public FloaterBase, public ThreadingTraits<MemcpyMT> {
    static constexpr auto name = "MemcpyMT";

    MemcpyMT() {
        setupThreads([&](float* __restrict__ cur_src, float* __restrict__ end_src, float* __restrict__ cur_dst) {
            memcpy(cur_dst, cur_src, sizeof(float) * (end_src - cur_src));
        });
    }
};
////////////////////////////////////////////////////////////////////////////////////////////////////


// Multithreaded memcpy on AVX instructions
struct MemcpyMT_StreamAVX : public FloaterBase, public ThreadingTraits<MemcpyMT_StreamAVX> {
    static constexpr auto name = "MemcpyMT_StreamAVX";

    MemcpyMT_StreamAVX() {
        setupThreads([&](float* __restrict__ cur_src, float* __restrict__ end_src, float* __restrict__ cur_dst) {
              const __m256i *mm_cur_src = reinterpret_cast<const __m256i*>(cur_src);
              const __m256i *mm_end_src = reinterpret_cast<const __m256i*>(end_src);
              __m256i *mm_cur_dst = reinterpret_cast<__m256i*>(cur_dst);
              size_t cnt = mm_end_src - mm_cur_src;
              #pragma unroll(8)
              while (cnt--) {
                _mm256_stream_si256(mm_cur_dst, _mm256_stream_load_si256(mm_cur_src));
                ++mm_cur_src;
                ++mm_cur_dst;
              }
        });
    }
};
////////////////////////////////////////////////////////////////////////////////////////////////////

// Multithreaded memcpy on AVX instructions
struct MemcpyMT_NonStreamAVX : public FloaterBase, public ThreadingTraits<MemcpyMT_NonStreamAVX> {
    static constexpr auto name = "MemcpyMT_NonStreamAVX";

    MemcpyMT_NonStreamAVX() {
        setupThreads([&](float* __restrict__ cur_src, float* __restrict__ end_src, float* __restrict__ cur_dst) {
              const __m256i *mm_cur_src = reinterpret_cast<const __m256i*>(cur_src);
              const __m256i *mm_end_src = reinterpret_cast<const __m256i*>(end_src);
              __m256i *mm_cur_dst = reinterpret_cast<__m256i*>(cur_dst);
              #pragma unroll(8)
              while (mm_cur_src != mm_end_src) {
                _mm256_store_si256(mm_cur_dst, _mm256_load_si256(mm_cur_src));
                ++mm_cur_src;
                ++mm_cur_dst;
              }
        });
    }
};
////////////////////////////////////////////////////////////////////////////////////////////////////

// Copy floats one by one and clamp if greater than threshold
struct DumbReadModifyWrite : FloaterBase {
    static constexpr auto name = "DumbReadModifyWrite";

    void perform() {
        float* cur_src = src_;
        float* cur_dst = dst_;

        float *end_src = src_ + kBufSize;
        while (UNLIKELY(cur_src != end_src)) {
            if (LIKELY(*cur_src > kThres)) {
                *cur_dst++ = kThres;
                ++cur_src;
            } else {
                *cur_dst++ = *cur_src++;
            }
        }
    }
};
////////////////////////////////////////////////////////////////////////////////////////////////////


// Same as DumbReadModifyWrite but multithreaded
struct ReadModifyWriteMT : public FloaterBase, public ThreadingTraits<ReadModifyWriteMT> {
    static constexpr auto name = "ReadModifyWriteMT";

    ReadModifyWriteMT() {
        setupThreads([&](float* __restrict__ cur_src, float* __restrict__ end_src, float* __restrict__ cur_dst) {
            while (UNLIKELY(cur_src != end_src)) {
                if (LIKELY(*cur_src > kThres)) {
                    *cur_dst++ = kThres;
                    ++cur_src;
                } else {
                    *cur_dst++ = *cur_src++;
                }
            }
        });
    }
};
////////////////////////////////////////////////////////////////////////////////////////////////////


struct ReadModifyWriteMT_AVX : public FloaterBase, ThreadingTraits<ReadModifyWriteMT_AVX> {
    static constexpr auto name = "ReadModifyWriteMT_AVX";
    static constexpr __m256 thresholds{kThres, kThres, kThres, kThres, kThres, kThres, kThres, kThres};
    static constexpr __m256i ones{~0ll,~0ll,~0ll,~0ll};

    ReadModifyWriteMT_AVX() {
        setupThreads([&](float* __restrict__ cur_src, float* __restrict__ end_src, float* __restrict__ cur_dst) {
            const __m256 ones_vec = _mm256_castsi256_ps(ones);
            #pragma unroll(8)
            while (UNLIKELY(cur_src != end_src)) {
                __m256 cur = _mm256_load_ps(cur_src); 
                __m256 mask = _mm256_cmp_ps(cur, thresholds, _CMP_LE_OQ);
    
                // masked store seems a few ticks slower on skylake than manually applying mask
                __m256 result = _mm256_and_ps(cur, mask);
                mask = _mm256_xor_ps(mask, ones_vec);
                result = _mm256_or_ps(_mm256_and_ps(thresholds, mask), result);
                _mm256_store_ps(cur_dst, result); 

                cur_src += 8;
                cur_dst += 8;
            }
        });
    }
};
////////////////////////////////////////////////////////////////////////////////////////////////////


int main(int, char**) {
    set_max_prio(pthread_self());

    measure<Memcpy>();
    measure<MemcpyStreamAVX>();
    measure<MemcpyMT>();
    measure<MemcpyMT_StreamAVX>();
    measure<MemcpyMT_NonStreamAVX>();

    measure<DumbReadModifyWrite>();
    measure<ReadModifyWriteMT>();
    measure<ReadModifyWriteMT_AVX>();
    return 0;
}
