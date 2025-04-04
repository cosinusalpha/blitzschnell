# Contributing to Blitzschnell

Thank you for considering contributing to Blitzschnell! We welcome contributions of all kinds. Please follow the guidelines below to ensure a smooth contribution process.

---

## Getting Started

1. **Fork the Repository**  
   Fork the repository on GitHub: [Blitzschnell Repository](https://github.com/cosinusalpha/blitzschnell).

2. **Clone Your Fork**  
   Clone your forked repository to your local machine:

   ```bash
   git clone https://github.com/<your-username>/blitzschnell.git
   cd blitzschnell
   ```

3. **Set Up the Development Environment**  
   Install the development dependencies:

   ```bash
   pip install -e .[dev]
   ```

4. **Run Tests**  
   Ensure all tests pass before making changes:

   ```bash
   pytest
   ```

---

## Making Changes

1. **Create a Branch**  
   Create a new branch for your changes:

   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Your Changes**  
   Implement your changes, ensuring they align with the project's coding standards.

3. **Run Linters and Type Checkers**  
   Ensure your code passes all checks:

   ```bash
   ruff check .
   mypy src/
   ```

4. **Add Tests**  
   Add or update tests to cover your changes. Use `pytest` to verify:

   ```bash
   pytest
   ```

5. **Commit Your Changes**  
   Write clear and concise commit messages:

   ```bash
   git add .
   git commit -m "Add feature: your-feature-name"
   ```

6. **Push Your Branch**  
   Push your branch to your fork:

   ```bash
   git push origin feature/your-feature-name
   ```

---

## Submitting a Merge Request

1. **Open a Merge Request**  
   Go to the [Blitzschnell Repository](https://github.com/cosinusalpha/blitzschnell) and open a merge request (MR) to the `main` branch.

2. **Ensure All Checks Pass**  
   Your MR will be reviewed only if all automated checks (e.g., tests, linters) succeed.

3. **Address Feedback**  
   Be responsive to feedback from maintainers and reviewers.

---

## Code Style

- Follow [PEP 8](https://peps.python.org/pep-0008/) for Python code.
- Use `ruff` and `mypy` to ensure code quality and type safety.

---

## Reporting Issues

If you encounter a bug or have a feature request, please open an issue on GitHub: [Bug Tracker](https://github.com/cosinusalpha/blitzschnell/issues).

---

Thank you for contributing to Blitzschnell!
