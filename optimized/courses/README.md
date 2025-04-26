# EGen Security AI Educational Courses

This directory contains educational courses for learning about cybersecurity and security AI concepts. The courses range from beginner to expert level and are designed to be accessible for all ages and learning styles.

## Course Organization

Courses are organized into four difficulty levels:

### Basics (For Complete Beginners)
Simple courses that explain cybersecurity topics using everyday language and examples. These courses are designed to be understandable by anyone, including children and those with no technical background.

### Intermediate 
More detailed courses for those who understand the basics and want to learn more practical skills. These courses introduce more technical concepts but still explain them clearly.

### Advanced
Technical courses for those with good cybersecurity knowledge. These dive deeper into complex topics and include hands-on exercises.

### Expert
Highly specialized courses for security professionals. These cover cutting-edge techniques and research-level concepts.

## Course Structure

Each course follows this structure:

1. **Header Information**:
   - Title
   - Brief description
   - Tags for easy searching
   - Author
   - Last updated date
   - Estimated completion time

2. **Introduction Section**:
   - Simple explanation of the topic
   - Real-world comparisons
   - Relevance to everyday life

3. **Main Content**:
   - Key concepts explained with examples
   - Visual aids when possible
   - Step-by-step instructions where appropriate
   - Practical tips

4. **Interactive Elements**:
   - Activities to reinforce learning
   - Scenarios to apply knowledge
   - Self-check questions

5. **Summary**:
   - Review of main points
   - Next steps for learning

6. **Quiz**:
   - 5-10 questions to test understanding
   - Mix of recall and application questions
   - Answer key provided

## Creating New Courses

Follow these guidelines when creating new courses:

1. **Choose the Right Level**: Make sure your content matches the difficulty level

2. **Use Simple Language**: Even advanced topics can be explained clearly
   - Define all technical terms
   - Use everyday examples
   - Break down complex ideas into smaller parts

3. **Follow the Template**: Use the template in `docs/COURSE_TEMPLATE.md`

4. **Name Your File**: Use lowercase with underscores (e.g., `web_security_basics.md`)

5. **Include Activities**: Every course should have at least one hands-on activity

6. **Add Visual Elements**: Suggest diagrams, charts, or images where helpful

7. **Review for Accuracy**: Double-check all technical information

8. **Test with Real Users**: If possible, have someone from the target age group review

## Available Courses

### Basics
- **Cybersecurity Fundamentals**: Introduction to key security concepts
- **Digital Privacy**: Controlling your information online
- *[More courses to be added]*

### Intermediate
- **Network Security Fundamentals**: Understanding how to secure computer networks
- *[More courses to be added]*

### Advanced
- **Ethical Hacking Fundamentals**: Introduction to security testing techniques
- *[More courses to be added]*

### Expert
- *[More courses to be added]*

## Accessing Courses

### Web Interface
1. Start the application using `python main.py`
2. Go to http://localhost:8000 in your browser
3. Navigate to the "Courses" section
4. Browse by difficulty level or search by topic

### API Access
Courses can be accessed programmatically via the API:

```python
import requests

# Get all courses
response = requests.get("http://localhost:8000/api/courses")
courses = response.json()

# Get courses by level
response = requests.get("http://localhost:8000/api/courses?level=basics")
basic_courses = response.json()

# Get a specific course
response = requests.get("http://localhost:8000/api/courses/cybersecurity_fundamentals")
course = response.json()
```

## Contributing New Courses

We welcome contributions! To add a new course:

1. Fork the repository
2. Create your course following the template
3. Place it in the appropriate level directory
4. Submit a pull request with a brief description

Please ensure your content is:
- Accurate and up-to-date
- Free from technical errors
- Written in clear, accessible language
- Properly formatted in Markdown

## Course Development Roadmap

We plan to develop the following new courses:

### Basics (In Development)
- Password Security Basics
- Safe Social Media Use
- Protecting Your Personal Information
- Recognizing Phishing Attempts
- Safe Web Browsing

### Intermediate (In Development)
- Understanding Malware
- Wireless Network Security
- Personal Device Security
- Security for Online Shopping
- Introduction to Encryption

### Advanced (In Development)
- Penetration Testing Fundamentals
- Secure Coding Practices
- Incident Response Basics
- Threat Hunting
- Security Tool Development

### Expert (In Development)
- Advanced Malware Analysis
- Machine Learning for Security
- Security Architecture Design
- Advanced Network Defense
- APT Detection and Response

## Questions or Suggestions?

If you have ideas for new courses or questions about existing ones, please open an issue in the repository or contact us at courses@egensecurity.ai. 