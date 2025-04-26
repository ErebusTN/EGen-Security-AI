# Secure Coding Practices

Learn how to write code that's resistant to common security attacks and vulnerabilities.

Tags: coding, programming, security, vulnerabilities, development
Author: EGen Security AI Team
Last Updated: 2023-07-05
Estimated Time: 60 minutes

## What is Secure Coding?

Secure coding means writing software in a way that protects it from attacks and prevents security vulnerabilities. It's like building a house with strong locks, alarm systems, and reinforced windows instead of leaving the doors wide open.

When programmers don't follow secure coding practices, they might accidentally create "holes" in their programs that attackers can use to steal data, take control of systems, or cause other damage.

## Why Should You Care?

If you're learning to code or already writing programs, understanding secure coding is important because:

- Your programs might handle sensitive information (like passwords or personal data)
- Insecure code can be exploited to harm users or systems
- Security vulnerabilities can damage your reputation as a developer
- Fixing security issues after a program is released is much harder than building it securely from the start
- Many coding jobs require security knowledge

## Common Security Vulnerabilities

Let's look at some of the most common security weaknesses in code and how to avoid them:

### 1. Injection Attacks

**The Problem:**
When user input is directly included in commands or queries without proper checking, attackers can "inject" malicious code.

**Example: SQL Injection**

Unsafe code (in Python):
```python
username = input("Enter username: ")
query = "SELECT * FROM users WHERE username = '" + username + "'"
```

If someone enters: `' OR '1'='1`, the query becomes:
```sql
SELECT * FROM users WHERE username = '' OR '1'='1'
```

This would return ALL users because `1=1` is always true!

**The Solution:** Use parameterized queries or prepared statements:

```python
username = input("Enter username: ")
cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
```

### 2. Broken Authentication

**The Problem:**
Weak password systems, poor session management, and other authentication failures let attackers pretend to be legitimate users.

**Example: Weak Password Storage**

Unsafe code:
```python
user_password = input("Create password: ")
# Storing password as plain text
save_to_database(username, user_password)
```

**The Solution:** Use secure password hashing:

```python
import hashlib, os

user_password = input("Create password: ")
salt = os.urandom(16)
hashed_password = hashlib.pbkdf2_hmac('sha256', user_password.encode(), salt, 100000)
save_to_database(username, hashed_password, salt)
```

### 3. Sensitive Data Exposure

**The Problem:**
Accidentally revealing sensitive information like passwords, credit card numbers, or personal data.

**Example: Insecure Logging**

Unsafe code:
```python
user_credit_card = input("Enter credit card: ")
print("Processing payment for card: " + user_credit_card)
log_file.write("Payment processed for card: " + user_credit_card)
```

**The Solution:** Mask sensitive data:

```python
user_credit_card = input("Enter credit card: ")
masked_card = "XXXX-XXXX-XXXX-" + user_credit_card[-4:]
print("Processing payment for card: " + masked_card)
log_file.write("Payment processed for card: " + masked_card)
```

### 4. XML External Entities (XXE)

**The Problem:**
When processing XML input, improperly configured XML parsers might process dangerous external entity references.

**Example: XXE Vulnerability**

Unsafe code:
```python
import xml.etree.ElementTree as ET

data = request.get_data()
tree = ET.parse(data)  # Vulnerable to XXE
```

**The Solution:** Disable external entities:

```python
import defusedxml.ElementTree as ET

data = request.get_data()
tree = ET.parse(data)  # Safe from XXE
```

### 5. Broken Access Control

**The Problem:**
When users can access functionality or data they shouldn't be allowed to.

**Example: Missing Authorization Check**

Unsafe code:
```python
@app.route('/user/<user_id>/settings')
def user_settings(user_id):
    # No check if logged-in user has permission to access this user's settings
    return get_user_settings(user_id)
```

**The Solution:** Add proper authorization:

```python
@app.route('/user/<user_id>/settings')
@login_required
def user_settings(user_id):
    if current_user.id != user_id and not current_user.is_admin:
        return "Unauthorized", 403
    return get_user_settings(user_id)
```

## Secure Coding Principles

Here are key principles to follow when writing secure code:

### 1. Never Trust User Input

Always assume that any data coming from users (or external systems) could be malicious:
- Validate all input against strict rules
- Sanitize data before using it
- Use whitelists (allow only known-good input) rather than blacklists

### 2. Defense in Depth

Don't rely on just one security measure:
- Add multiple layers of security
- Assume each protection might fail
- Make attackers break through multiple barriers

### 3. Principle of Least Privilege

Give code only the permissions it absolutely needs:
- Run processes with minimal required permissions
- Limit database user access rights
- Restrict API access to necessary functions only

### 4. Fail Securely

When errors happen, make sure they don't create security holes:
- Don't reveal sensitive information in error messages
- Return to a safe state after failures
- Log errors properly (but securely)

### 5. Keep Security Simple

Complex security systems often have unexpected weaknesses:
- Use established, well-tested libraries and frameworks
- Avoid inventing your own encryption or authentication
- Write clear, maintainable code

## Let's Try It Out!

Let's practice identifying and fixing a security vulnerability. Look at this code and see if you can spot the problem:

```python
def reset_password(user_email):
    # Generate a 4-digit reset code
    reset_code = str(random.randint(1000, 9999))
    
    # Store the reset code
    user = find_user_by_email(user_email)
    user.reset_code = reset_code
    user.save()
    
    # Send email with the reset code
    send_email(user_email, "Your password reset code is: " + reset_code)
    
    return "Reset code sent to your email"
```

**Security Issues:**
1. The reset code is too short (4 digits = only 9,000 possible codes)
2. Simple random numbers aren't cryptographically secure
3. No expiration time for the reset code

**Improved Version:**

```python
import secrets
import time

def reset_password(user_email):
    # Generate a secure, random 16-character reset code
    reset_code = secrets.token_urlsafe(16)
    
    # Store the reset code with expiration (1 hour from now)
    user = find_user_by_email(user_email)
    user.reset_code = reset_code
    user.reset_code_expiry = time.time() + 3600  # Current time + 1 hour in seconds
    user.save()
    
    # Send email with the reset code
    send_email(user_email, "Your password reset code is: " + reset_code)
    
    return "Reset code sent to your email"
```

## Secure Coding Tools

Here are some tools that can help you write more secure code:

### 1. Static Analysis Tools

These scan your code without running it to find potential security issues:
- **For Python**: Bandit, PyLint
- **For JavaScript**: ESLint with security plugins, SonarQube
- **For Java**: FindBugs, SpotBugs

### 2. Dependency Checkers

These check if your project uses libraries with known vulnerabilities:
- **For Python**: safety, pip-audit
- **For JavaScript**: npm audit, Snyk
- **For Java**: OWASP Dependency Check

### 3. Security Headers Analyzers

For web applications, these check if your HTTP security headers are properly set:
- SecurityHeaders.com
- OWASP ZAP's header analyzer

### 4. Web Application Scanners

These test running web applications for common vulnerabilities:
- OWASP ZAP (Zed Attack Proxy)
- Burp Suite Community Edition

## Secure Coding by Language

Different programming languages have their own security considerations:

### Python
- Use the `secrets` module instead of `random` for security purposes
- Be cautious with `pickle` and `eval()`
- Consider using `defusedxml` for XML processing

### JavaScript
- Use `===` instead of `==` for comparisons
- Avoid `eval()` and `innerHTML`
- Set proper Content Security Policy (CSP) headers

### Java
- Use prepared statements for database queries
- Avoid serialization when possible
- Use security manager for untrusted code

## Fun Facts

Did you know?
- The average cost of a data breach in 2023 is over $4 million
- About 70% of applications have security vulnerabilities when first tested
- The first computer worm that spread through the internet (The Morris Worm, 1988) was accidentally created by a student trying to measure the size of the internet
- Some major companies offer "bug bounties" - rewards for finding security vulnerabilities in their software

## Summary

Secure coding is a critical skill for any programmer. By understanding common vulnerabilities like injection attacks, broken authentication, and data exposure, you can write code that resists attacks. Remember the key principles: never trust user input, implement defense in depth, follow the principle of least privilege, fail securely, and keep security simple. Use security tools to help identify vulnerabilities, and remember that security is an ongoing process, not a one-time task.

## Quiz

1. Which of these is an example of an injection attack?
   a) Guessing a user's password through repeated attempts
   b) Adding malicious code to a database query through a form
   c) Intercepting network traffic between servers
   d) Uploading an oversized file to crash a server

2. Why should you use parameterized queries instead of string concatenation for database operations?
   a) They run faster
   b) They make the code more readable
   c) They prevent SQL injection attacks
   d) They use less memory

3. What's wrong with storing user passwords as plain text?
   a) It uses too much database space
   b) If the database is compromised, all passwords are exposed
   c) Users might forget their passwords
   d) The database might get corrupted

4. What does the "principle of least privilege" mean?
   a) Only administrators should have access to the system
   b) Users should have minimal knowledge of the security systems
   c) Code should only have the permissions it absolutely needs to function
   d) Security should be as simple as possible

5. Why is the `random` module in Python not suitable for security-sensitive operations?
   a) It's too slow for practical use
   b) It's not included in all Python installations
   c) It's not truly random and can be predicted
   d) It only works with certain data types

Answers: 1b, 2c, 3b, 4c, 5c 