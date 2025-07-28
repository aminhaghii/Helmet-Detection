# Security Policy

## Supported Versions

We actively support the following versions of HSE Vision with security updates:

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

We take the security of HSE Vision seriously. If you believe you have found a security vulnerability, please report it to us as described below.

### How to Report

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please report them via email to: [your-email@domain.com]

You should receive a response within 48 hours. If for some reason you do not, please follow up via email to ensure we received your original message.

### What to Include

Please include the following information in your report:

- Type of issue (e.g. buffer overflow, SQL injection, cross-site scripting, etc.)
- Full paths of source file(s) related to the manifestation of the issue
- The location of the affected source code (tag/branch/commit or direct URL)
- Any special configuration required to reproduce the issue
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the issue, including how an attacker might exploit the issue

### Response Process

1. **Acknowledgment**: We will acknowledge receipt of your vulnerability report within 48 hours.

2. **Investigation**: We will investigate the issue and determine its severity and impact.

3. **Resolution**: We will work on a fix and coordinate the release timeline with you.

4. **Disclosure**: We will publicly disclose the vulnerability after a fix is available, giving appropriate credit to the reporter (unless anonymity is requested).

## Security Best Practices

When using HSE Vision:

- Keep your installation up to date with the latest version
- Use strong authentication mechanisms
- Regularly review access logs
- Follow the principle of least privilege
- Keep your system and dependencies updated
- Use HTTPS when deploying web interfaces
- Validate all input data
- Implement proper error handling that doesn't expose sensitive information

## Dependencies

We regularly monitor our dependencies for known vulnerabilities and update them as needed. If you discover a vulnerability in one of our dependencies, please report it to us as well as the upstream maintainer.

## Contact

For any security-related questions or concerns, please contact us at [your-email@domain.com].