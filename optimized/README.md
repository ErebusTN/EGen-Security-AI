# EGen Security AI

EGen Security AI is a fun and educational cybersecurity platform that helps you learn about staying safe online while also providing powerful tools to check if your files are secure.

## Features

- **Cybersecurity Courses**: Learn about online safety with courses for all ages and skill levels
- **Security Scanner**: Check your files for viruses and security problems
- **AI-Powered Analysis**: Smart computer programs that can detect security threats
- **Interactive Dashboard**: See your progress and security status with colorful charts
- **User-Friendly Interface**: Simple design that's easy for everyone to use

## Who Is This For?

- **Kids and Students**: Learn cybersecurity basics with fun, easy-to-understand courses
- **Parents and Teachers**: Help children understand online safety
- **Beginners**: Start learning about cybersecurity with no technical background
- **IT Professionals**: Advanced courses for security experts and penetration testers
- **Developers**: Learn secure coding practices and scan your code for vulnerabilities

## Project Structure

```
├── client/                # Frontend website code
│   ├── public/            # Images and other files
│   └── src/               # Website code
│       ├── components/    # Reusable parts of the website
│       ├── pages/         # Different pages you can visit
│       ├── hooks/         # Special functions for the website
│       ├── services/      # Code that talks to the server
│       ├── store/         # Where data is stored
│       └── utils/         # Helper functions
│
├── src/                   # Backend server code
│   ├── ai/                # Artificial intelligence models
│   ├── api/               # How the website talks to the server
│   ├── config/            # Settings for the application
│   ├── db/                # Database code
│   ├── security/          # Security scanning tools
│   └── utils/             # Helper functions
│
├── courses/               # Learning materials
│   ├── basics/            # Courses for beginners
│   ├── intermediate/      # Courses for people with some knowledge
│   ├── advanced/          # Courses for experienced users
│   └── expert/            # Courses for professionals
│
├── models/                # Smart AI models that help detect threats
├── datasets/              # Information used to train the AI
└── logs/                  # Records of what the program does
```

## Our Courses

We've created over 20 courses across different difficulty levels:

### Basics (For Everyone)
- **Cybersecurity Fundamentals**: Learn the basics of staying safe online
- **Digital Privacy**: Controlling what information you share online
- **Password Security Basics**: Create strong passwords that are hard to guess
- **Safe Web Browsing**: How to explore the internet safely
- **Protecting Your Personal Information**: Keep your private details secure
- **Recognizing Phishing Attempts**: Spot scammers trying to trick you
- **Social Media Safety**: Stay safe while using social networks

### Intermediate (Some Experience Needed)
- **Network Security Fundamentals**: Understanding how computer networks work
- **Understanding Malware**: Learn about viruses and how they spread
- **Wireless Network Security**: Keep your WiFi network safe
- **Personal Device Security**: Protect your phones and tablets
- **Security for Online Shopping**: Shop online without getting scammed
- **Introduction to Encryption**: How secret codes protect your information

### Advanced (For Tech-Savvy Users)
- **Ethical Hacking Fundamentals**: Testing security the right way
- **Secure Coding Practices**: Writing computer code that's safe from hackers
- **Incident Response Basics**: What to do when something goes wrong
- **Threat Hunting**: Finding hidden dangers in computer systems
- **Security Tool Development**: Building your own security tools
- **Cloud Security**: Keeping data safe in the cloud

### Expert (For Professionals)
- **Machine Learning for Security**: Using AI to detect threats
- **Advanced Malware Analysis**: Taking apart dangerous software
- **Security Architecture Design**: Building secure computer systems
- **Advanced Network Defense**: Protecting networks from sophisticated attacks
- **APT Detection and Response**: Finding and stopping advanced hackers

## Getting Started

### What You'll Need

- Python 3.8 or higher (a programming language)
- Node.js 16 or higher (another programming language)
- npm or yarn (tools for installing software)

### Installation

1. Get the code:
```bash
git clone https://github.com/your-username/egen-security-ai.git
cd egen-security-ai
```

2. Set up a special environment for the code to run in:
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

3. Install the programs the server needs:
```bash
pip install -r requirements.txt
```

4. Install the programs the website needs:
```bash
cd client
npm install
cd ..
```

5. Create a settings file:
   Copy the `.env.example` file to a new file called `.env` and change the settings.

### Running the Program

#### For Testing and Development

1. Start the server:
```bash
python main.py --debug --reload
```

2. In a new window, start the website:
```bash
cd client
npm start
```

3. Visit http://localhost:3000 in your web browser

#### For Regular Use

1. Build the website files:
```bash
cd client
npm run build
cd ..
```

2. Start everything:
```bash
python main.py
```

3. Visit http://localhost:8000 in your web browser

## Using the Security Scanner

You can check files to see if they have security problems:

```python
from src.security import scan_file

# Check a file
result = scan_file('/path/to/file.py')
print(f"Threats found: {result['threats_detected']}")
print(f"Is it dangerous: {result['is_malicious']}")

# See details about any problems found
for match in result['signature_matches']:
    print(f"Problem: {match['name']} (How bad: {match['severity']})")
    print(f"What it means: {match['description']}")
```

## For Developers

### Running Tests

Make sure everything is working correctly:

```bash
pytest tests/
```

### Code Style

We keep our code neat and tidy:

```bash
# Format Python code
black src/ tests/

# Check JavaScript code
cd client
npm run lint
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The EGen Security AI Team for creating this project
- Open-source security tools and datasets that helped make this possible
- All the students and teachers who provided feedback on our courses 