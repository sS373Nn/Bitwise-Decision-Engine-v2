# Research References for Bitwise Decision Network

A curated collection of papers, articles, and resources relevant to building efficient neural networks using bitwise operations.

---

## Table of Contents
1. [Binary Neural Networks](#binary-neural-networks)
2. [Quantization & Data Representation](#quantization--data-representation)
3. [Hardware Acceleration](#hardware-acceleration)
4. [Associative Memory & Hopfield Networks](#associative-memory--hopfield-networks)
5. [Bit Manipulation & Optimization](#bit-manipulation--optimization)
6. [Walsh-Hadamard Transform & Fourier Analysis](#walsh-hadamard-transform--fourier-analysis)
7. [Lookup Tables & Efficient Inference](#lookup-tables--efficient-inference)
8. [Books](#books)
9. [Tools & Frameworks](#tools--frameworks)

---

## Binary Neural Networks

### Foundational Papers

**Binarized Neural Networks: Training Deep Neural Networks with Weights and Activations Constrained to +1 or -1**
- Authors: Courbariaux et al.
- Year: 2016
- Link: https://arxiv.org/abs/1602.02830
- Why: The foundational paper that introduced training methods for networks with binary weights and activations
- Key Concept: Replaces floating-point arithmetic with bitwise operations during inference

**XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks**
- Authors: Rastegari, Ordonez, Redmon, Farhadi
- Year: 2016
- Link: https://arxiv.org/abs/1603.05279
- Implementation: https://github.com/allenai/XNOR-Net
- Why: Demonstrates 58× faster convolutions and 32× memory savings using XNOR + popcount
- Key Concept: Binary weights AND binary activations enable pure bitwise operations

**XNOR-Net++: Improved Binary Neural Networks**
- Year: 2019
- Link: https://arxiv.org/abs/1909.13863
- Why: Improved training algorithm for better accuracy in binary networks

**Bitwise Neural Networks**
- Year: 2016
- Link: https://arxiv.org/abs/1601.06071
- Why: Focuses on using binary-valued parameters with basic bit logic for feedforward pass
- Key Concept: Weight compression and noisy backpropagation techniques

### Comprehensive Reviews

**Binary Neural Networks: A Survey**
- Year: 2020
- Link: https://www.sciencedirect.com/science/article/abs/pii/S0031320320300856
- Why: Comprehensive overview of BNN research landscape through 2020

**A comprehensive review of Binary Neural Network**
- Authors: Yuan & Agaian
- Year: 2021
- Link: https://arxiv.org/abs/2110.06804
- Why: Complete investigation from BNN predecessors to latest algorithms (2021)
- Coverage: Focuses exclusively on 1-bit activations and weights

**A Systematic Literature Review on Binary Neural Networks**
- Year: 2023
- Link: https://ieeexplore.ieee.org/document/10072399
- Why: Analysis of 239 research studies, identifies three main research directions:
  - Accuracy optimization
  - Compression optimization
  - Acceleration optimization

**Binarized Neural Network (BNN) and Its Implementation in Machine Learning**
- Link: https://neptune.ai/blog/binarized-neural-network-bnn-and-its-implementation-in-ml
- Why: Practical guide to implementing BNNs

---

## Quantization & Data Representation

### Quantization Fundamentals

**A White Paper on Neural Network Quantization**
- Authors: Markus Nagel et al.
- Year: 2021
- Link: https://arxiv.org/abs/2106.08295
- Why: Comprehensive practical handbook on quantization techniques
- Key Concept: Detailed guide to converting FP32 to INT8 and lower

**Integer Quantization for Deep Learning Inference: Principles and Empirical Evaluation**
- Year: 2020
- Link: https://arxiv.org/abs/2004.09602
- Why: Definitive guide on converting floating-point to integer representation
- Key Concept: Dynamic, static, and quantization-aware training (QAT) methods

**Advances in the Neural Network Quantization: A Comprehensive Review**
- Year: 2024
- Link: https://www.mdpi.com/2076-3417/14/17/7445
- Why: Most recent comprehensive review (August 2024)
- Coverage: From extreme low-bit (1-2 bit) to practical 8-bit quantization

**Neural Network Quantization & Number Formats From First Principles**
- Year: 2024
- Link: https://semianalysis.com/2024/01/11/neural-network-quantization-and-number/
- Why: Excellent primer on number representation and quantization mathematics

### Ternary Networks

**Trained Ternary Quantization**
- Year: 2016
- Link: https://arxiv.org/abs/1612.01064
- OpenReview: https://openreview.net/forum?id=S1_pAu9xl
- Why: Demonstrates {-1, 0, +1} weights achieve better accuracy than pure binary
- Key Concept: 2-bit representation with 16× model size reduction

**Ternary Neural Networks with Fine-Grained Quantization**
- Year: 2017
- Link: https://arxiv.org/abs/1705.01462
- Why: Methods for improving ternary network accuracy

**TAB: Unified and Optimized Ternary, Binary, and Mixed-precision Neural Network Inference on the Edge**
- Link: https://dl.acm.org/doi/10.1145/3508390
- Why: Framework for implementing ternary/binary networks on edge devices
- Key Concept: Bitwidth-last data format eliminates bit extraction overhead

**TRQ: Ternary Neural Networks With Residual Quantization**
- Link: https://ojs.aaai.org/index.php/AAAI/article/view/17036
- Why: Advanced ternary quantization technique

### Shift Networks (Power-of-Two Quantization)

**DenseShift: Towards Accurate and Efficient Low-Bit Power-of-Two Quantization**
- Year: 2022
- Link: https://arxiv.org/abs/2208.09708
- Why: Replace multiplications with bit-shift operations
- Key Concept: 1.6× speedup with competitive accuracy to full-precision

### Fixed-Point Arithmetic

**Fixed Point Quantization of Deep Convolutional Networks**
- Authors: Darryl D. Lin
- Link: http://proceedings.mlr.press/v48/linb16.pdf
- Why: Mathematical foundation for fixed-point neural networks

**Trainable Fixed-Point Quantization for Deep Learning Acceleration on FPGAs**
- Year: 2024
- Link: https://arxiv.org/abs/2401.17544
- Why: Hardware-aware fixed-point quantization methods

**F8Net: Fixed-Point 8-bit Only Multiplication for Network Quantization**
- Link: https://openreview.net/forum?id=_CfpJazzXT2
- Why: Framework using only INT8 operations

**A Post-training Quantization Method for the Design of Fixed-Point-Based FPGA/ASIC Hardware Accelerators for LSTM/GRU Algorithms**
- Link: https://pmc.ncbi.nlm.nih.gov/articles/PMC9117057/
- Why: Practical guide for implementing fixed-point on custom hardware

### Bit Packing & Compression

**Tiled Bit Networks: Sub-Bit Neural Network Compression Through Reuse of Learnable Binary Vectors**
- Year: 2024
- Link: https://arxiv.org/abs/2407.12075
- Why: Achieves 8× compression beyond standard binary networks
- Key Concept: Reusable binary vectors across the network

**Lossless Compression of Neural Network Components: Weights, Checkpoints, and K/V Caches in Low-Precision Formats**
- Year: 2025
- Link: https://arxiv.org/abs/2508.19263
- Why: State-of-the-art compression techniques
- Key Concept: Entropy coding achieves <1 bit per weight

---

## Hardware Acceleration

### Platform Comparisons

**Accelerating Binarized Neural Networks: Comparison of FPGA, CPU, GPU, and ASIC**
- Year: 2017
- Link: https://ieeexplore.ieee.org/document/7929192
- Why: Direct performance comparison across hardware platforms
- Key Finding: BNNs show dramatic advantages on FPGA/ASIC

**Binary Neural Networks in FPGAs: Architectures, Tool Flows and Hardware Comparisons**
- Link: https://pmc.ncbi.nlm.nih.gov/articles/PMC10675041/
- Why: Comprehensive review of FPGA implementations for BNNs

**FPGA-based acceleration for binary neural networks in edge computing**
- Year: 2023
- Link: https://www.sciencedirect.com/science/article/pii/S1674862X23000228
- Why: Recent developments in edge deployment

### Optimization Techniques

**Improving Efficiency in Neural Network Accelerator Using Operands Hamming Distance optimization**
- Year: 2020
- Link: https://arxiv.org/abs/2002.05293
- Why: Shows how Hamming distance correlates with energy consumption
- Key Concept: Minimize bit flips during computation

**Bitwise Neural Network Acceleration: Opportunities and Challenges**
- Link: https://ieeexplore.ieee.org/document/8760178
- Why: Overview of hardware acceleration opportunities for bitwise operations

---

## Associative Memory & Hopfield Networks

### Foundational Work

**Neural networks and physical systems with emergent collective computational abilities**
- Authors: John Hopfield
- Year: 1982
- Why: The original Hopfield network paper - surprisingly readable
- Key Concept: Content-addressable associative memory using binary states

### Modern Developments

**Hopfield Networks is All You Need**
- Year: 2020
- Why: Modern take connecting Hopfield networks to transformer attention mechanisms
- Key Concept: Shows exponentially larger storage capacity with energy-based formulation

**Modern Hopfield Networks**
- Wikipedia: https://en.wikipedia.org/wiki/Modern_Hopfield_network
- Why: Good overview of recent theoretical advances

### Implementation Resources

**Hopfield Network model of associative memory**
- Link: https://neuronaldynamics-exercises.readthedocs.io/en/latest/exercises/hopfield-network.html
- Why: Python tutorial with visualization for building intuition

**Hopfield Networks as Associative Memory**
- Link: https://borgelt.net/doc/hfnd/hfnd.html
- Why: Practical demonstration tool with examples

**Associative memory realized by a reconfigurable memristive Hopfield neural network**
- Link: https://www.nature.com/articles/ncomms8522
- Why: Hardware implementation using memristors

**A quantum Hopfield associative memory implemented on an actual quantum processor**
- Link: https://www.nature.com/articles/s41598-021-02866-z
- Why: Cutting-edge quantum implementation

---

## Bit Manipulation & Optimization

### Essential Resources

**Hacker's Delight**
- Author: Henry S. Warren Jr.
- Why: The bible of bit manipulation tricks
- Key Chapters:
  - Chapter 2: Basics
  - Chapter 5: Counting Bits (popcount algorithms)
  - Chapter 11: Some Elementary Functions
- Essential for building bitmath library

**Bit Twiddling Hacks**
- Link: https://graphics.stanford.edu/~seander/bithacks.html
- Why: Stanford CS resource with practical bit manipulation examples
- Coverage: Popcount, parity, bit reversal, and more

**Beating the popcount**
- Link: https://www.researchgate.net/publication/266289620_Beating_the_popcount
- Why: Advanced popcount optimization techniques

**You Won't Believe This One Weird CPU Instruction! (Popcount)**
- Link: https://vaibhavsagar.com/blog/2019/09/08/popcount/
- Why: Excellent explanation of popcount and its applications in modern computing

**Mastering Bitwise Operations: 4 Essential Algorithms**
- Link: https://arminnorouzi.github.io/posts/2023/05/blog-post-15/
- Why: Practical tutorial on fundamental bitwise algorithms

### Hamming Distance

**Hamming Distance Metric Learning**
- Link: https://norouzi.github.io/research/papers/hdml.pdf
- Why: Using Hamming distance for neural network learning
- Key Concept: Binary codes for efficient similarity search

**Global Robustness Evaluation of Deep Neural Networks with Provable Guarantees for the Hamming Distance**
- Link: https://www.ijcai.org/proceedings/2019/824
- Why: Connection between Hamming distance and network robustness

---

## Walsh-Hadamard Transform & Fourier Analysis

### Fourier Transform Fundamentals (For Understanding)

**But what is a Fourier transform? An visual introduction**
- Author: 3Blue1Brown (Grant Sanderson)
- YouTube: https://www.youtube.com/watch?v=spUNpyF58BY
- Why: Best visual explanation of Fourier transforms ever created
- Key Concept: Shows how complex patterns decompose into simple sine waves
- Essential for building intuition before diving into Walsh transforms

**The Fourier Transform (Interactive Tutorial)**
- Link: https://www.jezzamon.com/fourier/
- Why: Interactive web-based tool to play with Fourier transforms
- Helps build concrete understanding through experimentation

**Understanding the FFT Algorithm**
- Link: https://jakevdp.github.io/blog/2013/08/28/understanding-the-fft/
- Author: Jake VanderPlas
- Why: Explains Fast Fourier Transform algorithm clearly
- Relevant for understanding Fast Hadamard Transform by analogy

### Walsh-Hadamard Transform (Binary Fourier)

**Hadamard Transform**
- Wikipedia: https://en.wikipedia.org/wiki/Hadamard_transform
- Why: Comprehensive overview of the transform and its properties
- Key Concept: Binary equivalent of Fourier transform using only +1/-1 values

**Fast Walsh-Hadamard Transform**
- Wikipedia: https://en.wikipedia.org/wiki/Fast_Walsh%E2%80%93Hadamard_transform
- Why: Details the O(N log N) algorithm for efficient computation
- Key Concept: Uses butterfly operations with only additions/subtractions

**Walsh Functions and Their Applications**
- Paper: "Walsh Functions" by B. Golubov, A. Efimov, V. Skvortsov
- Why: Comprehensive mathematical treatment of Walsh functions
- Key Concept: Square wave basis functions (binary alternatives to sine/cosine)

**The Fast Hadamard Transform**
- Tutorial: http://www.cs.cmu.edu/afs/cs/academic/class/15859-f11/www/notes/lec05.pdf
- Why: CMU lecture notes with clear algorithm explanation and examples
- Key Concept: Shows butterfly diagram and computation steps

### Applications to Neural Networks

**Hadamard Transform for Neural Network Compression**
- Why: Walsh domain can enable better compression of network weights
- Key Concept: Transform weights to frequency domain, store only significant coefficients

**Fast Walsh-Hadamard Transform for Deep Learning**
- Search: "Hadamard transform neural networks" on Google Scholar
- Why: Emerging research area combining Walsh transforms with deep learning
- Key Concept: Frequency-domain pattern analysis for networks

**Orthogonal Binary Codes and Their Applications**
- Link: https://en.wikipedia.org/wiki/Hadamard_code
- Why: Walsh-Hadamard used for error-correcting codes
- Key Concept: Orthogonal patterns maximize separation (relevant for Hopfield)

### Pattern Analysis & Correlation

**Fast Correlation using Walsh-Hadamard Transform**
- Why: WHT enables fast pattern matching in O(N log N) time
- Key Concept: Convolution theorem applies to WHT just like FFT
- Application: Quick similarity detection between binary patterns

**Walsh Spectrum Analysis**
- Why: Analyze which "frequencies" (Walsh functions) patterns contain
- Key Concept: Patterns with non-overlapping Walsh spectra won't interfere
- Application: Choose Hopfield patterns with orthogonal Walsh representations

### Implementation Resources

**Fast Hadamard Transform in C**
- GitHub search: "fast hadamard transform C implementation"
- Why: Reference implementations for studying algorithms
- Look for: Bit-manipulation optimized versions

**FWHT: A Fast Hadamard Transform Library**
- Search: "FWHT library" on GitHub
- Why: Optimized implementations with various sizes
- Key feature: Often includes SIMD optimizations

**Butterfly Network Visualization**
- Link: https://en.wikipedia.org/wiki/Butterfly_network
- Why: Visual representation of Fast Hadamard Transform structure
- Key Concept: Parallel computation structure, ideal for hardware

### Bitwise Optimization for WHT

**Bitwise Butterfly Operations**
- Concept: Implement WHT butterfly using XOR and bit-packing
- Why: Could enable 64× parallelism (64 bits processed simultaneously)
- Status: Novel research direction - not extensively published

**Integer vs Floating-Point WHT**
- Why: Binary patterns use integer arithmetic naturally
- Key Concept: No floating-point needed for +1/-1 transforms
- Application: Perfect match for bitwise neural networks

### Connection to Hopfield Networks

**Walsh Domain Pattern Storage**
- Concept: Store Hopfield patterns as Walsh coefficients
- Why: Patterns with orthogonal Walsh spectra maximize capacity
- Status: Unexplored research direction

**Frequency-Based Pattern Retrieval**
- Concept: Match patterns in Walsh domain instead of state space
- Why: Could be faster than repeated XNOR operations
- Potential: O(N log N) pattern matching instead of O(N²)

### Visual Learning Tools

**Fourier Transform Visualizer**
- Search: "Fourier transform visualizer" online
- Why: Interactive tools to see frequency decomposition
- Helps: Build intuition that transfers to Walsh domain

**Walsh Function Plotter**
- Search: "Walsh functions visualization"
- Why: See the square wave basis functions
- Helps: Understand how binary patterns decompose

**Spectrum Analyzer**
- Any audio spectrum analyzer (browser-based)
- Why: See real-time frequency decomposition
- Analogy: Walsh spectrum works similarly for binary patterns

---

## Lookup Tables & Efficient Inference

**LUT-DLA: Lookup Table as Efficient Extreme Low-Bit Deep Learning Accelerator**
- Year: 2025
- Link: https://arxiv.org/abs/2501.10658
- IEEE: https://ieeexplore.ieee.org/document/10946705
- Why: Most recent work on LUT-based acceleration
- Key Concept: Convert neural networks to lookup tables

**Learnable Lookup Table for Neural Network Quantization**
- Year: 2022
- Link: https://openaccess.thecvf.com/content/CVPR2022/papers/Wang_Learnable_Lookup_Table_for_Neural_Network_Quantization_CVPR_2022_paper.pdf
- Why: 2-bit quantization using learnable LUTs

**LUT-NN: Empower Efficient Neural Network Inference with Centroid Learning and Table Lookup**
- Link: https://dl.acm.org/doi/10.1145/3570361.3613285
- Why: Practical framework achieving:
  - ≤16× FLOPs reduction
  - ≤7× model size reduction
  - ≤41.7× power reduction

**Look-Up Table based Neural Network Hardware**
- Year: 2024
- Link: https://arxiv.org/abs/2406.05282
- Why: Hardware design for LUT-based neural networks

**Efficient Neural Network Encoding for 3D Color Lookup Tables**
- Year: 2024
- Link: https://arxiv.org/abs/2412.15438
- Why: Encode hundreds of LUTs in <0.25 MB with 8-bit encoding

---

## Books

**Hacker's Delight (2nd Edition)**
- Author: Henry S. Warren Jr.
- Publisher: Addison-Wesley, 2012
- ISBN: 978-0321842688
- Why: Essential for mastering bit manipulation
- Focus: Chapters 2, 5, and 11 for neural network applications

**Reinforcement Learning: An Introduction (2nd Edition)**
- Authors: Richard S. Sutton & Andrew G. Barto
- Free online: http://incompleteideas.net/book/the-book.html
- Why: Foundation for Q-Learning and decision-making systems
- Key Chapters:
  - Chapter 4: Dynamic Programming
  - Chapter 6: Temporal-Difference Learning (Q-Learning)

**Neural Networks and Deep Learning**
- Author: Michael Nielsen
- Free online: http://neuralnetworksanddeeplearning.com/
- Why: Excellent introduction with great visualizations
- Chapter 7: Hopfield networks

**Optimized C++**
- Author: Kurt Guntheroth
- Publisher: O'Reilly, 2016
- Why: Performance optimization techniques for C++ implementations

**Professional CMake: A Practical Guide**
- Author: Craig Scott
- Why: Essential for cross-platform C++ project management

---

## Tools & Frameworks

### Quantization Tools

**TensorFlow Lite**
- Why: Industry-standard framework for quantized model deployment
- Link: https://www.tensorflow.org/lite

**PyTorch Quantization**
- Why: Built-in quantization support in PyTorch
- Link: https://pytorch.org/docs/stable/quantization.html

**Neural Network Distiller (Intel)**
- Link: https://intellabs.github.io/distiller/quantization.html
- Why: Compression and quantization toolkit

**QFX**
- Why: PyTorch library for fixed-point arithmetic emulation
- FPGA HLS compatible

### Development Tools

**GoogleTest**
- Why: Current testing framework for the project
- Link: https://github.com/google/googletest

**Intel Intrinsics Guide**
- Link: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/
- Why: SIMD operations for batch bitwise processing

**Microsoft WSL Documentation**
- Why: Cross-platform development (Windows/Linux)

---

## Additional Resources

### Online Courses & Tutorials

**3Blue1Brown: Essence of Linear Algebra**
- YouTube series
- Why: Visual intuition for neural network mathematics

**DeepLizard Q-Learning Playlist**
- Why: Practical reinforcement learning tutorials

**OpenAI Gym**
- Why: Hands-on environment for testing game AI

### Benchmark Datasets

**MNIST**
- Why: Standard benchmark for testing binary/ternary networks

**CIFAR-10**
- Why: Image classification benchmark for BNN research

**ImageNet**
- Why: Used in XNOR-Net and other major BNN papers

### Community Resources

**LeetCode Bit Manipulation Problems**
- Why: Practice implementing bit manipulation algorithms

**GitHub: Binary Neural Networks**
- Search query: "binary neural networks C++"
- Why: Real-world implementations to study

**GitHub: Bitwise Game AI**
- Why: Game-specific bitwise implementations

---

## Research Gaps & Opportunities

Based on literature review, these areas are under-explored:

1. **Pure bitwise Q-Learning**: Most Q-Learning uses floating-point; bitwise implementation largely unexplored
2. **Game-specific bit patterns**: Most BNN research targets vision; game state encoding could be more efficient
3. **Bitwise associative memory**: Combining Hopfield-style memory with pure bitwise operations
4. **Temporal decision sequences**: Handling game states over time with bitwise operations
5. **Interpretable bitwise networks**: Understanding what bitwise networks learn (vs black-box deep networks)
6. **Walsh-Hadamard Transform for Hopfield Networks**: Storing patterns in Walsh domain for better separation
7. **Bitwise Fast Hadamard Transform**: Optimizing WHT using bit-packing for 64× parallelism
8. **Frequency-domain pattern matching**: Using Walsh spectrum for O(N log N) pattern retrieval
9. **Modern Hopfield with bitwise approximations**: Approximating exponential energy function with bit shifts

---

## Notes

- **Last Updated**: 2025-01-20
- **Focus Areas for v2 Development**:
  1. XNOR + popcount as fundamental operations
  2. Ternary weights {-1, 0, +1} for decision granularity
  3. Hamming distance for pattern matching
  4. Fixed-point arithmetic for deterministic behavior
  5. Associative memory for state recall
  6. Walsh-Hadamard Transform for pattern analysis and storage
  7. Frequency-domain pattern matching (potentially novel research)

---

*This document will be continuously updated as new relevant research is discovered.*
