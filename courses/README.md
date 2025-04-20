# EGen Security AI Educational Courses

This directory contains educational resources for learning about AI-based security concepts, from basic fundamentals to expert-level topics. The courses are organized into three difficulty levels to help you progress in your learning journey.

## Course Structure

The courses are divided into three categories:

### Basics

Introductory courses for those new to the field of security AI:

- **Introduction to Security AI**: Fundamental concepts of AI in security applications
- **Creating an AI Security Model**: Step-by-step guide to building your first security model

### Advanced

More complex topics for users with some experience:

- **Using the Lily-Cybersecurity-7B Model**: Detailed guide to working with our specialized security model
- **Evaluating Security Models**: Techniques and metrics for assessing model performance
- **Adversarial Machine Learning**: Understanding and defending against adversarial attacks
- **Deploying Security Models**: Best practices for deploying models to production environments

### Expert

Advanced topics for experienced practitioners:

- **Building Robust Security Models**: Advanced techniques for model robustness
- **Custom Security Model Architectures**: Designing specialized neural networks for security applications

## How to Use These Courses

### Web Interface

You can access these courses through the EGen Security AI web interface:

1. Start the application using `python run.py`
2. Navigate to http://localhost:3000/courses in your browser
3. Select a course category and then a specific course
4. Progress through the content at your own pace

### Direct Access

You can also read the course materials directly as Markdown files:

```bash
# View a list of available courses
ls courses/basics
ls courses/advanced
ls courses/expert

# Open a specific course (example with VS Code)
code courses/basics/introduction_to_security_ai.md
```

### API Access

The courses can be accessed programmatically via the API:

```python
import requests

# Get a list of all courses
response = requests.get("http://localhost:5000/courses")
courses = response.json()

# Get a specific course by category and ID
response = requests.get("http://localhost:5000/courses/basics/introduction_to_security_ai")
course = response.json()

# Get course in HTML format
response = requests.get("http://localhost:5000/courses/basics/introduction_to_security_ai?format=html")
html_content = response.text
```

## Adding New Courses

To add a new course:

1. Determine the appropriate category (basics, advanced, or expert)
2. Create a new Markdown file in the corresponding directory
3. Follow the structure of existing courses:
   - Start with a # Title
   - Include a brief introduction
   - Use ## for main sections
   - Include code examples where appropriate
   - Add references or further reading at the end

Example:

```markdown
# My New Security Course

This course covers important aspects of security AI...

## First Section

Content goes here...

## Second Section

More content...

## References

- [Reference 1](https://example.com)
- [Reference 2](https://example.com)
```

4. Update the courses section in `docs/TECHNOLOGIES.md` to include your new course

## Course Development Roadmap

Future courses planned for development:

- **Security Model Evaluation Metrics**: Understanding precision, recall, and security-specific metrics
- **Privacy-Preserving AI**: Techniques for maintaining privacy in security AI systems
- **Graph Neural Networks for Threat Hunting**: Using graph structures to detect complex threats
- **Multimodal Security Models**: Combining different data types for enhanced threat detection

## Contributing

We welcome contributions to the course materials! Please see the main [CONTRIBUTING.md](../docs/CONTRIBUTING.md) file for guidelines on how to contribute. 