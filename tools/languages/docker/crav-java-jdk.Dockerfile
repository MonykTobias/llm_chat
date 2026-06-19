# Java (no build file) analysis image: a plain JDK (javac/java) + a Checkstyle
# CLI, for projects that have neither Maven nor Gradle. javac compiles every
# .java into a throwaway dir; compilation IS the type check.
FROM eclipse-temurin:21-jdk
ARG CHECKSTYLE_VERSION=10.17.0
RUN set -eux; \
    apt-get update; apt-get install -y --no-install-recommends curl; \
    rm -rf /var/lib/apt/lists/*; \
    curl -fL -o /opt/checkstyle.jar \
      "https://github.com/checkstyle/checkstyle/releases/download/checkstyle-${CHECKSTYLE_VERSION}/checkstyle-${CHECKSTYLE_VERSION}-all.jar"; \
    printf '#!/bin/sh\nexec java -jar /opt/checkstyle.jar "$@"\n' > /usr/local/bin/checkstyle; \
    chmod +x /usr/local/bin/checkstyle
