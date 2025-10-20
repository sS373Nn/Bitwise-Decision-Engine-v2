# Bitwise Decision Engine v2 - Project Goals

## Vision
Build a decision-making engine for game AI using pure bitwise operations, combining Hopfield Networks (associative memory) with Walsh-Hadamard Transform (frequency-domain pattern analysis) to create an efficient, novel approach to pattern-based decision making.

---

## Core Hypothesis

**Can we improve Hopfield network capacity and retrieval speed by leveraging Walsh-Hadamard Transform for pattern analysis and storage optimization?**

### Expected Benefits:
1. **Better pattern separation** - Walsh domain analysis helps choose orthogonal patterns
2. **Faster retrieval** - O(N log N) pattern matching via Fast Hadamard Transform
3. **Extreme efficiency** - Pure bitwise operations (no floating-point)
4. **Novel contribution** - This specific combination appears unexplored in literature

---

## Success Criteria

### Minimum Viable Product (MVP):
- [ ] Bitmath library with core operations (popcount, XNOR, Hamming distance)
- [ ] Classical Hopfield network (bitwise implementation)
- [ ] Fast Walsh-Hadamard Transform (bitwise implementation)
- [ ] Basic pattern storage and retrieval working
- [ ] Can store and retrieve TicTacToe board states

### Success Metrics:
1. **Capacity**: Store at least N/20 patterns (perfect retrieval) or N/7 (good retrieval)
2. **Accuracy**: >95% correct pattern retrieval from noisy inputs
3. **Speed**: Faster than baseline Hopfield for pattern matching
4. **Efficiency**: Memory usage in bits documented and optimized

### Stretch Goals:
- [ ] Walsh-domain pattern storage (store patterns as Walsh coefficients)
- [ ] Automatic pattern selection (choose patterns with orthogonal Walsh spectra)
- [ ] Comparison with modern Hopfield (2020 exponential energy function)
- [ ] Scale to more complex games (Connect 4, Chess endgames)

---

## Methodology

### Phase 1: Foundation (BitMath Library)
**Duration:** 2-4 weeks

**Deliverables:**
- Core bitwise operations (from Hacker's Delight)
  - Popcount (multiple implementations, benchmarked)
  - Parity
  - Leading/trailing zeros
  - Bit reversal
- Pattern operations
  - Hamming distance
  - XNOR operations
  - Bit packing/unpacking
- Test suite (TDD approach with GoogleTest)

**Learning Resources:**
- Hacker's Delight (Chapters 2, 5, 11)
- Bit Twiddling Hacks (Stanford)

---

### Phase 2: Classical Hopfield Network
**Duration:** 2-3 weeks

**Deliverables:**
- Hopfield network implementation (bitwise)
  - Pattern storage (weight matrix generation)
  - Pattern retrieval (iterative update rule)
  - Energy function calculation
- XNOR + popcount optimization
- Capacity testing (how many patterns before degradation?)
- Test suite with known patterns

**Benchmarks:**
- Pattern capacity vs theoretical limit (0.05N and 0.15N)
- Retrieval accuracy with noise
- Speed (patterns retrieved per second)

**Learning Resources:**
- Hopfield 1982 paper
- neuronaldynamics-exercises.readthedocs.io tutorial

---

### Phase 3: Walsh-Hadamard Transform
**Duration:** 2-3 weeks

**Deliverables:**
- Fast Hadamard Transform implementation
  - Recursive algorithm
  - Iterative (butterfly) algorithm
  - Bitwise optimizations
- Walsh function generation
- Pattern spectrum analysis
  - Decompose patterns into Walsh basis
  - Measure spectral overlap between patterns
- Test suite with known transforms

**Benchmarks:**
- Transform speed (elements transformed per second)
- Verify correctness (forward + inverse = identity)
- Compare to naive O(N²) approach

**Learning Resources:**
- Wikipedia: Fast Walsh-Hadamard Transform
- CMU lecture notes
- 3Blue1Brown Fourier video (for intuition)

---

### Phase 4: Integration - Hopfield + Walsh-Hadamard
**Duration:** 3-4 weeks

**Deliverables:**
- Pattern analysis tool
  - Compute Walsh spectrum of candidate patterns
  - Select patterns with minimal spectral overlap
- Hybrid storage experiments
  - Store patterns directly vs Walsh coefficients
  - Compare capacity and speed
- Frequency-domain retrieval
  - Match in Walsh domain
  - Inverse transform to get result
- Comprehensive benchmarks

**Experiments:**
1. **Baseline**: Classical Hopfield with random patterns
2. **Optimized**: Hopfield with Walsh-selected patterns
3. **Hybrid**: Walsh-domain storage
4. **Frequency**: Pattern matching in Walsh domain

**Metrics to Compare:**
- Maximum patterns stored (at 95% accuracy)
- Retrieval speed
- Memory usage
- Robustness to noise

---

### Phase 5: Game AI Application
**Duration:** 2-3 weeks

**Deliverables:**
- TicTacToe implementation
  - Board state encoding (18 bits: 2 bits per square)
  - Win pattern storage
  - Move recommendation system
- Benchmark against baselines
  - Minimax algorithm
  - Random play
  - Simple heuristics
- Playable demo

**Success Criteria:**
- Engine plays valid moves
- Engine recognizes winning positions
- Competitive with simple heuristics
- Runs efficiently (< 1ms per decision)

---

### Phase 6: Documentation & Analysis
**Duration:** 2-3 weeks

**Deliverables:**
- Technical report
  - Introduction & motivation
  - Methodology
  - Results & benchmarks
  - Analysis (what worked, what didn't)
  - Limitations & future work
- Code documentation
  - Well-commented implementations
  - API documentation
  - Usage examples
- GitHub repository
  - Clear README
  - Build instructions
  - Example programs
  - Benchmark results

**Optional (if results warrant):**
- arXiv preprint
- Blog post series
- Conference workshop submission

---

## Measurable Outcomes

### Primary Questions to Answer:

1. **Does Walsh-Hadamard analysis improve pattern selection?**
   - Measure: Pattern capacity with vs without Walsh selection
   - Hypothesis: 20-50% improvement in capacity

2. **Is Walsh-domain pattern matching faster?**
   - Measure: Time to retrieve pattern (microseconds)
   - Hypothesis: O(N log N) vs O(N²) should show speedup for N > 64

3. **Does storing patterns as Walsh coefficients reduce memory?**
   - Measure: Bits per pattern stored
   - Hypothesis: Sparse Walsh spectrum could enable compression

4. **Is the combined approach practical for game AI?**
   - Measure: Decision speed, win rate, code complexity
   - Hypothesis: Competitive with heuristics, faster than minimax

### Secondary Questions:

5. **What is the bitwise implementation overhead?**
   - Compare to floating-point reference implementations

6. **How does it scale to larger state spaces?**
   - Test with board sizes from 3×3 to 8×8

7. **What patterns are learned/stored?**
   - Interpret the stored patterns (explainability)

---

## Technical Architecture

### Module Structure:
```
bitwise-engine/
├── bitmath/           # Core bitwise operations
│   ├── popcount.cpp
│   ├── hamming.cpp
│   ├── xnor.cpp
│   └── bit_utils.cpp
├── walsh/             # Walsh-Hadamard Transform
│   ├── fwht.cpp       # Fast Walsh-Hadamard Transform
│   ├── walsh_functions.cpp
│   └── spectrum.cpp
├── hopfield/          # Hopfield Network
│   ├── classical.cpp
│   ├── modern.cpp     # (stretch goal)
│   └── weights.cpp
├── hybrid/            # Combined system
│   ├── pattern_selector.cpp
│   ├── walsh_storage.cpp
│   └── freq_retrieval.cpp
├── games/             # Game AI applications
│   ├── tictactoe.cpp
│   ├── connect4.cpp   # (stretch goal)
│   └── board_encoding.cpp
├── benchmarks/        # Performance testing
└── tests/             # Unit tests (GoogleTest)
```

### Development Approach:
- **Test-Driven Development (TDD)** - Write tests first
- **Incremental** - Each module works independently before integration
- **Benchmarked** - Every optimization is measured
- **Documented** - Code comments and design decisions recorded

---

## Risk Assessment & Mitigation

### Potential Challenges:

**1. Walsh-Hadamard may not improve Hopfield**
- **Mitigation**: Even null results are valuable; document why
- **Fallback**: Focus on bitwise optimization of classical Hopfield

**2. Bitwise implementation complexity**
- **Mitigation**: Start simple, optimize incrementally
- **Fallback**: Compare to floating-point to validate correctness

**3. Limited capacity for complex games**
- **Mitigation**: Start with TicTacToe (small state space)
- **Fallback**: Focus on endgames or specific patterns

**4. Time constraints**
- **Mitigation**: Each phase delivers working module
- **Fallback**: MVP is still valuable learning project

---

## Timeline Estimate

**Total Duration:** 4-6 months (part-time work)

### Breakdown:
- Phase 1 (BitMath): 2-4 weeks
- Phase 2 (Hopfield): 2-3 weeks
- Phase 3 (Walsh-Hadamard): 2-3 weeks
- Phase 4 (Integration): 3-4 weeks
- Phase 5 (Game AI): 2-3 weeks
- Phase 6 (Documentation): 2-3 weeks

**Flexible schedule** - adjust based on findings and interest

---

## Learning Outcomes (Guaranteed!)

Even if the research hypothesis doesn't pan out, you will:
- ✅ Master bitwise operations (valuable skill)
- ✅ Understand neural networks at fundamental level
- ✅ Learn signal processing concepts (transforms)
- ✅ Practice rigorous benchmarking and analysis
- ✅ Build impressive portfolio project
- ✅ Gain experience with TDD and professional code structure
- ✅ Develop intuition for algorithm complexity and optimization

---

## Next Steps

1. **Set up project structure**
   - Initialize git repository (already done!)
   - Create folder structure
   - Set up build system (CMake)
   - Configure GoogleTest

2. **Start Phase 1: BitMath**
   - Read Hacker's Delight Chapter 2
   - Implement popcount (multiple versions)
   - Write comprehensive tests
   - Benchmark on your hardware

3. **Document as you go**
   - Keep research journal (markdown notes)
   - Track decisions and their rationale
   - Note interesting findings

4. **Stay flexible**
   - Adjust plan based on what you discover
   - Follow interesting tangents
   - Don't be afraid to pivot if needed

---

## Resources & References

See `references.md` for comprehensive list of papers, books, and resources.

**Key Resources for Each Phase:**
- Phase 1: Hacker's Delight, Bit Twiddling Hacks
- Phase 2: Hopfield 1982 paper, neuronaldynamics tutorial
- Phase 3: CMU FHT notes, 3Blue1Brown Fourier video
- Phase 4: Experimental - document your findings!
- Phase 5: Game AI textbooks, existing implementations
- Phase 6: Academic writing guides, technical blog examples

---

## Contact & Collaboration

If this project generates interesting results:
- Post to GitHub with permissive license (MIT/Apache)
- Share on Hacker News / Reddit
- Consider reaching out to researchers for collaboration
- Write blog posts explaining your approach
- Maybe submit to conference workshop

**Remember:** The journey matters as much as the destination!

---

*Last Updated: 2025-01-20*
*This is a living document - update as the project evolves!*
