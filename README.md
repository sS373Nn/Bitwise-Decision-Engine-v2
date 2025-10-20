# Bitwise Decision Engine v2

**A novel approach to game AI combining Hopfield Networks with Walsh-Hadamard Transform using pure bitwise operations.**

---

## 🎯 Project Vision

This project explores whether associative memory (Hopfield Networks) can be enhanced with frequency-domain pattern analysis (Walsh-Hadamard Transform) to create an efficient, interpretable decision engine for game AI.

### Core Innovation
- **Hopfield Networks** for pattern-based decision making
- **Walsh-Hadamard Transform** for pattern selection and optimization
- **Pure bitwise operations** for extreme efficiency
- **Novel combination** - This specific approach appears unexplored in current research

---

## 🔬 Research Hypothesis

**Can we improve Hopfield network capacity and retrieval speed by leveraging Walsh-Hadamard Transform for pattern analysis?**

Expected benefits:
- Better pattern separation through Walsh spectrum analysis
- Faster retrieval via O(N log N) frequency-domain matching
- Extreme efficiency through bitwise-only implementation
- Potential for novel contribution to associative memory research

---

## 📂 Project Structure

```
bitwise-engine-v2/
├── README.md              # This file
├── PROJECT_GOALS.md       # Detailed roadmap and methodology
├── references.md          # Research papers and resources
├── LICENSE                # Project license
│
├── bitmath/               # Core bitwise operations library
├── walsh/                 # Walsh-Hadamard Transform implementation
├── hopfield/              # Hopfield Network implementation
├── hybrid/                # Integration of Hopfield + Walsh-Hadamard
├── games/                 # Game AI applications
├── benchmarks/            # Performance testing
└── tests/                 # Unit tests (GoogleTest)
```

---

## 🚀 Current Status

**Project Phase:** Planning & Foundation

See [PROJECT_GOALS.md](PROJECT_GOALS.md) for detailed roadmap.

### Completed:
- ✅ Research and literature review
- ✅ Project goals defined
- ✅ Repository structure planned
- ✅ Reference library compiled

### Next Steps:
- [ ] Set up build system (CMake + GoogleTest)
- [ ] Begin Phase 1: BitMath Library
- [ ] Implement popcount with multiple algorithms
- [ ] Establish TDD workflow

---

## 🎓 Learning Objectives

Even if the research hypothesis doesn't yield breakthrough results, this project guarantees:
- Deep understanding of bitwise operations and optimization
- Hands-on experience with neural network fundamentals
- Knowledge of signal processing and transforms
- Practice with rigorous benchmarking and scientific methodology
- Professional software development practices (TDD, documentation)

---

## 📚 Key Resources

### Essential Reading:
- **Hacker's Delight** (Henry S. Warren Jr.) - Bitwise operations bible
- **Hopfield 1982 Paper** - Original associative memory paper
- **Walsh-Hadamard Transform** - Wikipedia and CMU lecture notes
- **Modern Hopfield Networks (2020)** - Recent advances connecting to transformers

See [references.md](references.md) for comprehensive bibliography.

---

## 🎮 Target Applications

**Primary:** TicTacToe (proof of concept)
**Stretch Goals:** Connect 4, Chess endgames, other perfect-information games

The engine should:
- Store "good" game positions as patterns
- Retrieve similar winning positions from current state
- Recommend moves based on pattern matching
- Run efficiently on resource-constrained devices

---

## 🔬 Methodology

1. **Test-Driven Development (TDD)** - Tests written before implementation
2. **Incremental Modules** - Each component works independently
3. **Rigorous Benchmarking** - Every optimization measured
4. **Scientific Documentation** - Design decisions and findings recorded

---

## 📊 Success Metrics

### Minimum Viable Product (MVP):
- Bitmath library with core operations
- Working Hopfield network (bitwise)
- Working Walsh-Hadamard transform (bitwise)
- Basic pattern storage and retrieval
- TicTacToe demo application

### Quantitative Goals:
- **Capacity:** Store N/20 patterns (perfect) or N/7 patterns (good)
- **Accuracy:** >95% pattern retrieval from noisy inputs
- **Speed:** Faster than baseline Hopfield for pattern matching
- **Efficiency:** Memory usage documented and optimized

---

## 🛠️ Technology Stack

- **Language:** C++ (for bitwise control and performance)
- **Testing:** GoogleTest
- **Build System:** CMake (cross-platform)
- **Development:** TDD approach
- **Platform:** Windows (WSL) + Linux compatible

---

## 📝 Documentation Philosophy

This project emphasizes learning and transparency:
- **Code comments** explain the "why," not just the "what"
- **Design decisions** documented in markdown files
- **Benchmark results** tracked over time
- **Research journal** maintained throughout development
- **Null results** documented (learning from what doesn't work)

---

## 🤝 Contributing

This is currently a personal learning project, but feedback and suggestions are welcome!

If you find this work interesting:
- ⭐ Star the repository
- 💬 Open issues for discussion
- 📧 Reach out for collaboration opportunities

---

## 📖 Project Timeline

**Estimated Duration:** 4-6 months (part-time development)

### Phase Breakdown:
1. **BitMath Library** - 2-4 weeks
2. **Hopfield Network** - 2-3 weeks
3. **Walsh-Hadamard Transform** - 2-3 weeks
4. **Integration** - 3-4 weeks
5. **Game AI Application** - 2-3 weeks
6. **Documentation & Analysis** - 2-3 weeks

See [PROJECT_GOALS.md](PROJECT_GOALS.md) for detailed phase descriptions.

---

## 🔍 Research Gaps Addressed

Based on literature review, this project explores under-researched areas:
- Walsh-Hadamard Transform for Hopfield pattern optimization
- Bitwise implementation of Fast Hadamard Transform
- Frequency-domain pattern matching for associative memory
- Game-specific pattern encoding and decision-making
- Bitwise approximations of modern Hopfield energy functions

---

## 📜 License

MIT License - See [LICENSE](LICENSE) file for details.

This project is intended for research and educational purposes.

---

## 🙏 Acknowledgments

### Inspiration:
- John Hopfield (1982) - Associative memory networks
- Henry S. Warren Jr. - Hacker's Delight
- Ramsauer et al. (2020) - Modern Hopfield Networks
- The binary neural networks research community

### Tools:
- GoogleTest for rigorous testing
- CMake for cross-platform builds
- Claude (Anthropic) for research assistance and pair programming

---

## 📫 Contact

For questions, collaboration, or discussion:
- Open an issue on GitHub
- See profile for contact information

---

## 🎯 Quick Start (Coming Soon!)

```bash
# Clone the repository
git clone https://github.com/yourusername/Bitwise-Decision-Engine-v2.git
cd Bitwise-Decision-Engine-v2

# Build
mkdir build && cd build
cmake ..
make

# Run tests
./run_tests

# Run TicTacToe demo
./tictactoe_demo
```

*(Build system to be implemented in Phase 1)*

---

## 📊 Project Status Updates

**2025-01-20:** Project initialized, research phase completed, roadmap established.

More updates as development progresses!

---

**Remember:** The journey of learning matters as much as the destination of results! 🚀

---

*This README will be updated as the project evolves.*
