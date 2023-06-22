# Floater
Very simple and dumb benchmark for macbook pro basic memory operations e.g. copying large amount of floats, or copying floats while also modifying it. Explores performance of multiple different ways of doing those operations.
## Results
CPU: 2.4 GHz 8-Core Intel Core i9
Mem: 2 x 32GB DDR4 2667MHz (MT40A4G8BAF-062E:B)
As of Dec 2019:
```Measurement: Memcpy
Complete: init=0.348004s avg=25831.5 min=24192 max=52493 us | gflops=2.59794 | throughput=10.3918 gb/s
============================================================
Measurement: MemcpyStreamAVX
Complete: init=0.34675s avg=20625.2 min=19441 max=36067 us | gflops=3.25373 | throughput=13.0149 gb/s
============================================================
Measurement: MemcpyMT
Complete: init=0.377183s avg=2358.37 min=2018 max=3999 us | gflops=28.4556 | throughput=113.822 gb/s
============================================================
Measurement: MemcpyMT_StreamAVX
Complete: init=0.361673s avg=7623.76 min=7332 max=8446 us | gflops=8.80259 | throughput=35.2104 gb/s
============================================================
Measurement: MemcpyMT_NonStreamAVX
Complete: init=0.359606s avg=2305.5 min=1945 max=4336 us | gflops=29.1082 | throughput=116.433 gb/s
============================================================
Measurement: DumbReadModifyWrite
Complete: init=0.346688s avg=28522.8 min=27716 max=32688 us | gflops=2.35281 | throughput=9.41126 gb/s
============================================================
Measurement: ReadModifyWriteMT
Complete: init=0.355529s avg=3005.32 min=2625 max=3500 us | gflops=22.33 | throughput=89.3201 gb/s
============================================================
Measurement: ReadModifyWriteMT_AVX
Complete: init=0.356066s avg=2964 min=2619 max=3415 us | gflops=22.6413 | throughput=90.5653 gb/s
============================================================
```
## Links
1. [Memory chip data scheet](https://media-www.micron.com/-/media/client/global/documents/products/data-sheet/dram/ddr4/32gb_ddr4_x4x8_2cs_twindie.pdf?rev=60a1f441656240408a5d191e4d05a447)

