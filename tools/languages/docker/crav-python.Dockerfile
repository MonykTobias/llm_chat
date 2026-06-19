# Python analysis image: the official slim base + the linters/test/type tools
# baked in, so a review never needs them on the host. Project deps are installed
# at run time into /work (cached via the pip volume mounted at /root/.cache/pip).
FROM python:3.12-slim
RUN pip install --no-cache-dir pylint mypy pytest pytest-cov
