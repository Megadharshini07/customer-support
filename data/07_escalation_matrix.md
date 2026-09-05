# Policy: Escalation Matrix (POL-EM-07)

Use this matrix whenever a ticket matches one of the triggers below. When a ticket matches an
escalation trigger, the resolution draft must recommend escalation rather than a direct
resolution, even if another policy would otherwise allow a direct answer.

| Trigger                                             | Escalate to        |
|------------------------------------------------------|---------------------|
| Suspected account compromise / unauthorized access    | Security team        |
| Billing dispute $50+ or multi-charge dispute           | Billing team          |
| Chargeback already filed with bank                     | Billing team          |
| Account merge / purchase-history transfer request      | Tier 2 support        |
| Ambiguous warranty claim (defect vs. accidental damage)| Warranty specialist   |
| Threats of legal action or regulatory complaint         | Legal/Compliance team |
| Customer requests deletion of all personal data         | Data Privacy team     |
| Any request not covered by an existing policy document  | Tier 2 support (human)|

General agents (and this assistant) should never attempt to resolve tickets matching these
triggers on their own. The correct output in these cases is a routing recommendation, not a
final resolution.
