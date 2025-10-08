# Contributing to RetriVex

Thank you for your interest in contributing to RetriVex! This document provides guidelines and instructions for contributing.

## Code of Conduct

Be respectful, inclusive, and constructive in all interactions.

## Getting Started

### Development Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Sushanth-reddyD/retrivex.git
   cd retrivex
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install in development mode**:
   ```bash
   pip install -e ".[dev]"
   ```

4. **Run tests to verify setup**:
   ```bash
   pytest tests/ -v
   ```

## Development Workflow

### Making Changes

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**:
   - Follow existing code style
   - Add tests for new functionality
   - Update documentation as needed

3. **Run tests**:
   ```bash
   pytest tests/ -v --cov=retrivex
   ```

4. **Format code**:
   ```bash
   black src/ tests/
   ruff check src/ tests/ --fix
   ```

5. **Commit changes**:
   ```bash
   git add .
   git commit -m "feat: add new feature"
   ```

6. **Push and create PR**:
   ```bash
   git push origin feature/your-feature-name
   ```

### Commit Message Convention

Use conventional commit format:

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `test:` Test additions or changes
- `refactor:` Code refactoring
- `perf:` Performance improvements
- `chore:` Maintenance tasks

## Testing Guidelines

### Writing Tests

- Use pytest for all tests
- Organize tests in `tests/unit/` and `tests/integration/`
- Aim for >80% code coverage
- Include both positive and negative test cases

### Test Structure

```python
class TestFeature:
    """Tests for Feature."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return ...
    
    def test_basic_functionality(self, sample_data):
        """Test basic use case."""
        result = feature(sample_data)
        assert result == expected
    
    def test_edge_case(self):
        """Test edge case."""
        with pytest.raises(ValueError):
            feature(invalid_input)
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=retrivex --cov-report=html

# Run specific test file
pytest tests/unit/test_models.py

# Run specific test
pytest tests/unit/test_models.py::TestChunk::test_valid_chunk
```

## Code Style

### Python Style

- Follow PEP 8
- Use Black for formatting (line length: 100)
- Use type hints for function signatures
- Write docstrings for public APIs

### Example

```python
def retrieve_spans(
    seed_hits: List[SeedHit],
    config: RetrievalConfig,
) -> List[Span]:
    """
    Retrieve and compose spans from seed hits.
    
    Args:
        seed_hits: Base kNN hits from vector search
        config: Retrieval configuration
        
    Returns:
        List of composed spans ordered for context
        
    Raises:
        ValueError: If seed_hits is empty
    """
    if not seed_hits:
        raise ValueError("seed_hits cannot be empty")
    
    # Implementation...
    return spans
```

## Documentation

### Docstring Format

Use Google-style docstrings:

```python
def function_name(param1: str, param2: int) -> bool:
    """
    Brief description.
    
    Longer description if needed.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When this happens
    """
```

### Updating Documentation

- Update README.md for user-facing changes
- Update docs/ for conceptual changes
- Add examples for new features
- Keep docstrings in sync with code

## Pull Request Process

1. **Ensure tests pass**: All tests must pass before merging
2. **Add tests**: New features must include tests
3. **Update docs**: Update relevant documentation
4. **Small PRs**: Keep PRs focused and manageable
5. **Descriptive title**: Use clear, descriptive PR titles
6. **Link issues**: Reference related issues in PR description

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests added/updated
- [ ] All tests pass
- [ ] Manual testing performed

## Checklist
- [ ] Code follows project style
- [ ] Documentation updated
- [ ] CHANGELOG updated (if applicable)
```

## Areas for Contribution

### High Priority

- [ ] LangChain adapter implementation
- [ ] LlamaIndex adapter implementation
- [ ] Performance benchmarks
- [ ] More evaluation metrics

### Medium Priority

- [ ] Haystack adapter
- [ ] Additional chunking strategies
- [ ] Visualization tools for spans
- [ ] More comprehensive examples

### Low Priority

- [ ] Additional ordering strategies
- [ ] CLI tool
- [ ] Jupyter notebook examples
- [ ] Integration with popular vector DBs

## Questions?

- Open an issue for bugs or feature requests
- Start a discussion for questions or ideas
- Check existing issues before creating new ones

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
