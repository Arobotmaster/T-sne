<!-- Sync Impact Report -->
<!-- Version: 0.0.0 → 1.0.0 -->
<!-- Modified Principles: All 9 SDD principles established -->
<!-- Added Sections: Data Science Constraints, Web Application Standards -->
<!-- Removed Sections: None -->
<!-- Templates Updated: ✅ plan-template.md, ✅ spec-template.md -->
<!-- Follow-up TODOs: None -->

# MOF数据t-SNE交互式可视化 Constitution

## Core Principles

### I. Library-First Principle
Every data processing and visualization component MUST start as a standalone library. Libraries MUST be self-contained, independently testable, and documented. Clear scientific purpose required - no organizational-only libraries. Each algorithm (PCA, t-SNE, data preprocessing) MUST be extractable as a separate library for reuse in other scientific computing projects.

### II. CLI Interface Mandate
Every data processing library MUST expose functionality via CLI. Text in/out protocol MUST be followed: stdin/args → stdout, errors → stderr. MUST support both JSON (for programmatic use) and human-readable formats (for scientific users). Visualization parameters MUST be configurable via command-line arguments.

### III. Test-First Imperative (NON-NEGOTIABLE)
TDD is MANDATORY for all data processing algorithms: Tests MUST be written → Scientific validation approved → Tests MUST fail → Only then implement. Red-Green-Refactor cycle MUST be strictly enforced. All statistical calculations and dimensionality reductions MUST have corresponding test cases with known expected outputs.

### IV. Integration-First Testing
Integration tests MUST cover: End-to-end data pipeline (CSV → PCA → t-SNE → visualization), Web application API contracts, Parameter validation across components, Export functionality (PNG/SVG), and Real-time interaction responsiveness. Each algorithm integration point MUST have dedicated integration tests.

### V. Scientific Observability
All data processing steps MUST emit structured logs suitable for scientific reproducibility. Intermediate results (PCA variance explained, t-SNE convergence metrics) MUST be logged. Debug mode MUST provide detailed algorithmic state information. Performance metrics (computation time, memory usage) MUST be tracked for each processing stage.

### VI. Semantic Versioning for Scientific Software
MAJOR.MINOR.BUILD format MUST be used. MAJOR version for algorithmic changes that affect scientific results, MINOR for new features without changing existing outputs, BUILD for bug fixes and documentation. All data processing algorithms MUST include version information in their output.

### VII. Simplicity in Scientific Computing
Start with the simplest scientifically valid approach. YAGNI principles MUST be applied to avoid over-engineering. Complex statistical methods MUST be justified by scientific necessity. Each component MUST have a clear, single responsibility in the data processing pipeline.

### VIII. Anti-Abstraction for Scientific Code
Avoid unnecessary abstractions that obscure the scientific methodology. Data transformations MUST be traceable and understandable. Mathematical operations MUST be directly visible in the code. Scientific algorithms MUST not be hidden behind multiple layers of abstraction.

### IX. Web Application Integration Testing
The complete web application MUST be tested as an integrated system: Frontend-backend communication, Real-time parameter updates, Export functionality with consistent state, Cross-browser compatibility for visualization, and Performance under typical dataset sizes.

## Data Science Constraints

### Algorithmic Implementation Requirements
- PCA implementation MUST support configurable variance retention (typically 95% or fixed component count)
- t-SNE implementation MUST support multiple perplexity values and convergence criteria
- Data preprocessing MUST handle missing values and normalize appropriately
- All numerical computations MUST use appropriate precision (float64 recommended)

### Scientific Validation
- Algorithm outputs MUST be validated against known datasets or theoretical expectations
- Visualization MUST accurately represent the underlying mathematical transformations
- Parameter changes MUST produce scientifically consistent results
- Performance MUST be documented for typical dataset sizes (1K-100K data points)

### Data Pipeline Integrity
- Each processing stage MUST validate input data format and quality
- Error handling MUST provide clear scientific context for failures
- Intermediate results MUST be serializable for debugging and reproducibility
- Pipeline MUST be restartable from any stage for debugging purposes

## Web Application Standards

### Frontend Requirements
- Visualization MUST be responsive and work on common screen sizes
- Interactive controls MUST provide immediate feedback without page reload
- Export functionality MUST preserve current visualization state exactly
- UI MUST be intuitive for scientific users without extensive training

### Backend API Standards
- API endpoints MUST follow REST conventions
- All endpoints MUST validate input parameters and return appropriate HTTP status codes
- Computation endpoints MUST support asynchronous processing for large datasets
- API responses MUST include sufficient metadata for scientific reproducibility

### Performance Standards
- Initial page load MUST complete within 3 seconds on typical hardware
- Parameter adjustments MUST update visualization within 1 second
- Data processing for 10K points MUST complete within 30 seconds
- Memory usage MUST remain below 2GB for typical datasets

## Governance

### Constitution Supremacy
This constitution supersedes all other development practices and technical decisions. All code reviews, pull requests, and architectural decisions MUST verify compliance with these principles. The scientific integrity of the data processing pipeline is the highest priority.

### Amendment Process
- Amendments MUST be proposed with clear scientific justification
- Changes MUST be documented with reasoning and impact analysis
- All amendments MUST maintain backward compatibility for existing validated results
- Major changes MUST be approved by project stakeholders after review

### Compliance Review
- All pull requests MUST include constitution compliance checks
- Code reviewers MUST explicitly verify adherence to relevant principles
- Automated tests MUST validate constitutional constraints where possible
- Non-compliance MUST be documented with justification and exception approval

### Documentation Requirements
- All algorithms MUST have clear mathematical documentation
- API endpoints MUST be documented with examples and expected outputs
- User-facing documentation MUST include scientific methodology explanations
- Implementation decisions MUST be justified in technical documentation

**Version**: 1.0.0 | **Ratified**: 2025-09-19 | **Last Amended**: 2025-09-19