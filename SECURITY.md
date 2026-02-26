# Security Policy

kindpath-analyser is the KindPath Creative Analyser — it processes audio files, analyses creative signatures, and manages a creative seedbank. It handles potentially sensitive creative works.

## Reporting a Vulnerability

**Do not open a public issue for security vulnerabilities.**

Email: security@kindpathcollective.org

We will acknowledge receipt within 48 hours and coordinate responsible disclosure.

## Scope

Security concerns relevant to this repository include:
- Arbitrary file read/write via unsanitised audio file paths
- Dependency vulnerabilities (librosa, numpy, scipy, etc.)
- Seedbank data integrity — protecting the creative records from corruption or unauthorised modification
- API key or external service credential exposure

## Supported Versions

| Branch | Supported |
|--------|-----------|
| `main` | ✅ Active |
