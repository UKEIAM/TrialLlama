# TREC evaulation system requires
- TOPIC_NO in the first row
- Q0 constant in the second row (0)
- NCT ID in the third row
- The RANK for the document
- The SCORE for the document (probability)

## How to achieve correct output
- First run evaluation and calculate the probability for patient topic to be eligible for certain clinical trial. That is the only probability of interest, since if he is not-eligible or the trial is not relevant for him, 