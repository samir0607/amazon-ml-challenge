# Dataset report

## train

| source | rows | countries | addr missing | non-ASCII name | non-ASCII addr | name len (median) | addr len (median) |
|---|---|---|---|---|---|---|---|
| source1 | 2,206,821 | US 1,323,633, India 883,188 | 0.00% | 0.00% | 0.03% | 24 | 41 |
| source2 | 5,034,616 | US 3,016,817, India 2,017,799 | 3.36% | 15.19% | 9.50% | 25 | 37 |
| source3 | 5,285,603 | US 3,170,056, India 2,115,547 | 3.33% | 11.48% | 9.02% | 25 | 42 |

## test

| source | rows | countries | addr missing | non-ASCII name | non-ASCII addr | name len (median) | addr len (median) |
|---|---|---|---|---|---|---|---|
| source1 | 1,732,544 | India 809,986, US 663,106, France 259,452 | 0.00% | 2.35% | 4.26% | 24 | 50 |
| source2 | 4,887,273 | India 2,312,565, US 1,871,330, France 703,378 | 2.65% | 18.99% | 14.75% | 25 | 43 |
| source3 | 5,082,316 | India 2,405,000, US 1,945,701, France 731,615 | 2.68% | 14.51% | 14.35% | 25 | 44 |

## Ground truth (train)

- S1 entities: 2,206,821; singletons (no match): 123,247 (5.58%)
- true pairs: 7,638,365; mean matches per S1: 3.46; max 11
- every S2/S3 record matches at most one S1 (exclusivity holds for 100% of labelled records)
- matched share of S2 / S3 records: 73.4% / 74.6%
- country label agrees on 100% of true pairs (used to scope blocking; open set: France appears only in test)

### Matches per S1

| matches | S1 count |
|---|---|
| 0 | 123,247 |
| 1 | 119,157 |
| 2 | 375,212 |
| 3 | 530,841 |
| 4 | 484,115 |
| 5 | 321,957 |
| 6 | 164,868 |
| 7 | 63,968 |
| 8 | 18,680 |
| 9 | 4,205 |
| 10 | 534 |
| 11 | 37 |
