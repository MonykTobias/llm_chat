# Java (Maven) analysis image: the official Maven+JDK base + a Checkstyle CLI
# (linter). Compilation IS the type check, so the build tool covers compile/tests.
# The local Maven repo is mounted at run time (/root/.m2).
FROM maven:3-eclipse-temurin-21
USER root
ARG CHECKSTYLE_VERSION=10.17.0
RUN set -eux; \
    apt-get update; apt-get install -y --no-install-recommends curl; \
    rm -rf /var/lib/apt/lists/*; \
    curl -fL -o /opt/checkstyle.jar \
      "https://github.com/checkstyle/checkstyle/releases/download/checkstyle-${CHECKSTYLE_VERSION}/checkstyle-${CHECKSTYLE_VERSION}-all.jar"; \
    printf '#!/bin/sh\nexec java -jar /opt/checkstyle.jar "$@"\n' > /usr/local/bin/checkstyle; \
    chmod +x /usr/local/bin/checkstyle
