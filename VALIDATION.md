# Validation record — 3 November 2026

- Windows / Python 3.12: 41 tests passed, including 13 problem/learner app combinations.
- Full seven-cell notebook executed locally; `outputs/LearnLens_executed.ipynb` retains the outputs. Hosted Colab unverified.
- Controlled benchmark: 108 fits completed; GF2 zero error on all test draws; mean SGD test error 0.328648 on the specified grid. See source/config hashes in JSON.
- Streamlit localhost server started and health endpoint returned `ok`.
- Browser screenshot QA of localhost was blocked: in-app browser cannot access the host localhost, and local Chrome app access timed out. Automated Streamlit checks execute the application logic but do not establish pixel-level layout correctness.
- Docker unavailable; Dockerfile supplied but image build unverified.
- GitHub CI / public deployment status is reported in the repository Actions and Pages UI; local success alone is not a remote deployment test.
- Inherited paper and old results are superseded. They are preserved locally but excluded from the public release.
