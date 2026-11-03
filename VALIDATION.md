# Validation record — 3 November 2026

- Windows / Python 3.12: 43 tests passed, including 13 problem/learner app combinations and two single-class fallback regressions.
- Full seven-cell notebook executed locally; `outputs/LearnLens_executed.ipynb` retains the outputs. Hosted Colab unverified.
- Controlled benchmark: 108 fits completed; GF2 zero error on all test draws; mean SGD test error 0.328648 on the specified grid. See source/config hashes in JSON.
- Streamlit localhost server started and health endpoint returned `ok`.
- Public GitHub Pages demo verified in-browser: benchmark filters update results, live neural training completed with test error 0.000 on the fixed demo test draw, methods tab checked, and desktop screenshots inspected. Local Streamlit layout has not been visually checked; its logic was exercised by AppTest.
- Docker unavailable; Dockerfile supplied but image build unverified.
- GitHub Linux CI passed: https://github.com/naren-srinivas/learnlens/actions/runs/41605095654
- Pages deployment succeeded: https://github.com/naren-srinivas/learnlens/actions/runs/41604953233
- Public site returned HTTP 200; its served benchmark JSON matched the local file exactly. All 43 repository files matched local Git blob hashes before this status update.
- Public demo: https://naren-srinivas.github.io/learnlens/
- Python app is local/deployment-ready, not deployed to a hosted Streamlit backend. Docker and hosted Google Colab remain unverified.
- Inherited paper and old results are superseded. They are preserved locally but excluded from the public release.
