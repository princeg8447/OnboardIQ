# TechCorp IT Setup Guide for New Employees

## Equipment

You will receive:
- Laptop (MacBook Pro 14" or Dell XPS depending on role)
- Power adapter and accessories
- Access card for office entry

Engineering roles additionally receive a second monitor and docking station.
Request via IT portal if not provided on Day 1.

---

## Accounts & Access (Day 1 Priority)

### Corporate Email (Google Workspace)
- Format: firstname.lastname@techcorp.com
- Activated by IT before your joining date
- If not active, raise a ticket at it-helpdesk.techcorp.internal

### Mandatory: Enable Multi-Factor Authentication (MFA)
1. Go to myaccount.google.com
2. Security → 2-Step Verification → Enable
3. Use Google Authenticator or hardware key (engineering roles get a YubiKey)

### Slack
- Invite sent to corporate email
- Key channels to join: #general, #announcements, #your-team, #random
- Engineering: also join #engineering, #deployments, #on-call

### VPN (Cisco AnyConnect)
1. Download from: software.techcorp.internal/vpn
2. Server: vpn.techcorp.com
3. Login with corporate email credentials + MFA

---

## Software Installation

Access the Software Portal at software.techcorp.internal for licensed tools.

### All Employees
- Google Chrome (preferred browser for internal tools)
- Slack Desktop
- Zoom (video calls)
- 1Password (password manager — company-licensed)

### Engineering Roles
- VS Code or JetBrains IDE (license in Software Portal)
- Docker Desktop
- kubectl + Helm (for K8s access)
- AWS CLI / GCP CLI (depending on your team)
- Postman

### Design Roles
- Figma (company account — join via link in #design Slack channel)
- Adobe Creative Cloud (request via IT ticket)

---

## Security Policies

1. **Screen lock**: Auto-lock after 5 minutes of inactivity (enforced by MDM)
2. **Disk encryption**: FileVault (Mac) / BitLocker (Windows) enabled automatically
3. **Password policy**: Minimum 12 characters, rotated every 90 days
4. **No personal USB drives**: Prohibited by security policy
5. **Public WiFi**: Always use VPN when on public/home networks

Violation of security policies may result in disciplinary action.

---

## Common IT Issues & Solutions

| Issue | Solution |
|-------|----------|
| Can't access HR portal | Must be on VPN or office network |
| Email not syncing | Re-authenticate in Gmail settings |
| Slack 2FA issue | Contact IT on +91-80-XXXX-XXXX |
| Laptop slow / overheating | Raise ticket; IT will run diagnostics |
| Lost access card | Report to admin@techcorp.com immediately |

---

## IT Support

- **Portal**: it-helpdesk.techcorp.internal (raise tickets here)
- **Email**: it-support@techcorp.com
- **Phone**: +91-80-XXXX-XXXX (9 AM – 8 PM, Mon–Fri)
- **Urgent/after-hours**: Slack DM @it-oncall
- **SLA**: P1 (system down) = 2 hrs | P2 (partial issue) = 4 hrs | P3 (general) = 1 business day
