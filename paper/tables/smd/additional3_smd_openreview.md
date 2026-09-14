## SMD results for the three additional systems
Direct-audio condition, 1%-winsorized scenario-level SMD; 95% participant-clustered bootstrap CI. Sign convention: Race = Black − White, Gender = Female − Male. “—” = scenario not estimable under the per-prompt validity filter.

### Option A — single combined table

| Scenario | Gemini-3.5-Flash · Race | Gemini-3.5-Flash · Gender | Gemini-3.1-Pro · Race | Gemini-3.1-Pro · Gender | Kimi-Audio-7B · Race | Kimi-Audio-7B · Gender |
|---|---|---|---|---|---|---|
| Caring Household Members | -0.10 [-0.39, +0.21] | +0.02 [-0.27, +0.33] | +0.00 [-0.10, +0.09] | -0.14 [-0.24, -0.04] | -0.00 [-0.16, +0.14] | +0.02 [-0.13, +0.16] |
| Civic and Religious Activities | +0.00 [-0.15, +0.15] | +0.06 [-0.09, +0.22] | -0.07 [-0.16, +0.03] | +0.01 [-0.09, +0.10] | +0.05 [-0.06, +0.16] | -0.03 [-0.15, +0.09] |
| Educational Activities | — | — | -0.08 [-0.20, +0.03] | -0.11 [-0.23, +0.01] | +0.13 [-0.02, +0.27] | +0.04 [-0.10, +0.19] |
| Finance | — | — | — | — | +0.16 [+0.03, +0.29] | -0.12 [-0.26, +0.03] |
| Health | -0.03 [-0.28, +0.22] | +0.09 [-0.18, +0.30] | -0.00 [-0.16, +0.15] | -0.05 [-0.18, +0.10] | -0.23 [-0.38, -0.08] | +0.00 [-0.19, +0.20] |
| Household Activities | — | — | +0.14 [+0.04, +0.26] | -0.02 [-0.12, +0.09] | +0.04 [-0.12, +0.21] | -0.08 [-0.24, +0.07] |
| Leisure and Sports | -0.06 [-0.21, +0.09] | -0.05 [-0.20, +0.09] | +0.04 [-0.05, +0.14] | -0.01 [-0.11, +0.07] | +0.08 [-0.03, +0.18] | -0.01 [-0.11, +0.11] |
| Personal Care | +0.10 [-0.06, +0.25] | +0.12 [-0.03, +0.28] | -0.02 [-0.12, +0.07] | +0.02 [-0.08, +0.10] | +0.05 [-0.07, +0.16] | -0.01 [-0.14, +0.10] |
| Politics | — | — | -0.10 [-0.24, +0.04] | -0.05 [-0.20, +0.08] | -0.07 [-0.22, +0.07] | +0.04 [-0.14, +0.21] |
| Purchasing Goods and Services | +0.03 [-0.09, +0.14] | -0.04 [-0.16, +0.08] | -0.03 [-0.13, +0.06] | -0.01 [-0.12, +0.07] | -0.29 [-0.40, -0.17] | -0.13 [-0.26, -0.01] |
| Work-related Activities | — | — | +0.01 [-0.08, +0.11] | +0.02 [-0.07, +0.13] | -0.09 [-0.25, +0.06] | -0.06 [-0.21, +0.09] |

### Option B — three per-model tables

**Gemini-3.5-Flash**

| Scenario | Race SMD [95% CI] | Gender SMD [95% CI] |
|---|---|---|
| Caring Household Members | -0.10 [-0.39, +0.21] | +0.02 [-0.27, +0.33] |
| Civic and Religious Activities | +0.00 [-0.15, +0.15] | +0.06 [-0.09, +0.22] |
| Health | -0.03 [-0.28, +0.22] | +0.09 [-0.18, +0.30] |
| Leisure and Sports | -0.06 [-0.21, +0.09] | -0.05 [-0.20, +0.09] |
| Personal Care | +0.10 [-0.06, +0.25] | +0.12 [-0.03, +0.28] |
| Purchasing Goods and Services | +0.03 [-0.09, +0.14] | -0.04 [-0.16, +0.08] |

**Gemini-3.1-Pro**

| Scenario | Race SMD [95% CI] | Gender SMD [95% CI] |
|---|---|---|
| Caring Household Members | +0.00 [-0.10, +0.09] | -0.14 [-0.24, -0.04] |
| Civic and Religious Activities | -0.07 [-0.16, +0.03] | +0.01 [-0.09, +0.10] |
| Educational Activities | -0.08 [-0.20, +0.03] | -0.11 [-0.23, +0.01] |
| Health | -0.00 [-0.16, +0.15] | -0.05 [-0.18, +0.10] |
| Household Activities | +0.14 [+0.04, +0.26] | -0.02 [-0.12, +0.09] |
| Leisure and Sports | +0.04 [-0.05, +0.14] | -0.01 [-0.11, +0.07] |
| Personal Care | -0.02 [-0.12, +0.07] | +0.02 [-0.08, +0.10] |
| Politics | -0.10 [-0.24, +0.04] | -0.05 [-0.20, +0.08] |
| Purchasing Goods and Services | -0.03 [-0.13, +0.06] | -0.01 [-0.12, +0.07] |
| Work-related Activities | +0.01 [-0.08, +0.11] | +0.02 [-0.07, +0.13] |

**Kimi-Audio-7B**

| Scenario | Race SMD [95% CI] | Gender SMD [95% CI] |
|---|---|---|
| Caring Household Members | -0.00 [-0.16, +0.14] | +0.02 [-0.13, +0.16] |
| Civic and Religious Activities | +0.05 [-0.06, +0.16] | -0.03 [-0.15, +0.09] |
| Educational Activities | +0.13 [-0.02, +0.27] | +0.04 [-0.10, +0.19] |
| Finance | +0.16 [+0.03, +0.29] | -0.12 [-0.26, +0.03] |
| Health | -0.23 [-0.38, -0.08] | +0.00 [-0.19, +0.20] |
| Household Activities | +0.04 [-0.12, +0.21] | -0.08 [-0.24, +0.07] |
| Leisure and Sports | +0.08 [-0.03, +0.18] | -0.01 [-0.11, +0.11] |
| Personal Care | +0.05 [-0.07, +0.16] | -0.01 [-0.14, +0.10] |
| Politics | -0.07 [-0.22, +0.07] | +0.04 [-0.14, +0.21] |
| Purchasing Goods and Services | -0.29 [-0.40, -0.17] | -0.13 [-0.26, -0.01] |
| Work-related Activities | -0.09 [-0.25, +0.06] | -0.06 [-0.21, +0.09] |

