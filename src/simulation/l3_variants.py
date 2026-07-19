"""Chapter 3 L3 correct and misspecified standalone estimators.

The variants share the replication-level DGP and differ only in their locked
measurement assumptions:

- l3_correct uses source accuracy and channel dependence from the DGP;
- l3_ignore_dependence forces conditional independence;
- l3_wrong_fixed_pi fixes prevalence to a prespecified incorrect value;
- l3_clean_anchor incorrectly